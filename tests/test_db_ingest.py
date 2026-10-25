import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, inspect, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ingest as ing
from src.db import (dataset_summary, get_engine, ingested_files, load_dataframe, records, rejected_records,
                    snapshots, pipeline_runs)

GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "opd_visits.csv").exists(), reason="run scripts/generate_fake_data.py first")


def count(engine, table, **where):
    with engine.connect() as c:
        q = select(func.count()).select_from(table)
        for k, v in where.items():
            q = q.where(table.c[k] == v)
        return c.execute(q).scalar()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


BASE = "visit_date,patient,dx,vas\n2026-10-01,p1,flu,1.5\n2026-10-02,p2,cold,2.5\n2026-10-03,p3,flu,\n"


def test_tables_are_created(engine):
    assert {"datasets", "ingested_files", "records", "rejected_records", "snapshots", "pipeline_runs"} <= set(inspect(engine).get_table_names())


def test_ingest_and_read_back(engine, tmp_path):
    r = ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    assert (r.status, r.rows_in, r.rows_new, r.rows_duplicate) == ("loaded", 3, 3, 0)
    df = load_dataframe(engine, "opd")
    assert list(df.columns) == ["visit_date", "patient", "dx", "vas"] and len(df) == 3
    assert str(df["visit_date"].dtype).startswith("datetime64") and df["vas"].isna().sum() == 1     # blank stays blank
    assert df["vas"].dtype.kind == "f" and df["dx"].tolist() == ["flu", "cold", "flu"]


def test_same_file_is_never_loaded_twice_even_if_renamed(engine, tmp_path):
    ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    r = ing.ingest_file(engine, write(tmp_path / "renamed.csv", BASE), "opd")
    assert r.status == "skipped" and count(engine, records) == 3 and count(engine, ingested_files) == 1


def test_overlapping_file_adds_only_new_rows(engine, tmp_path):
    ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    r = ing.ingest_file(engine, write(tmp_path / "b.csv", BASE + "2026-10-04,p4,flu,3.0\n"), "opd")
    assert (r.rows_in, r.rows_new, r.rows_duplicate) == (4, 1, 3) and count(engine, records) == 4


def test_datasets_are_independent(engine, tmp_path):
    f = write(tmp_path / "a.csv", BASE)
    assert ing.ingest_file(engine, f, "opd").status == "loaded" and ing.ingest_file(engine, f, "icu").status == "loaded"
    assert count(engine, records, dataset="opd") == count(engine, records, dataset="icu") == 3


def test_window_filter_is_inclusive_of_the_end_day(engine, tmp_path):
    ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    assert len(load_dataframe(engine, "opd", "2026-10-02", "2026-10-03")) == 2
    assert len(load_dataframe(engine, "opd", end="2026-10-01")) == 1 and load_dataframe(engine, "opd", "2027-01-01").empty


def test_unreadable_dates_are_rejected_with_a_trail(engine, tmp_path):
    f = write(tmp_path / "a.csv", BASE + "not-a-date,p9,flu,1.0\n,p8,flu,1.0\n")
    r = ing.ingest_file(engine, f, "opd")
    assert (r.rows_new, r.rows_rejected) == (3, 2) and count(engine, rejected_records) == 2
    with engine.connect() as c:
        assert c.execute(select(rejected_records.c.reason)).scalars().first() == "Unreadable or missing date"


def test_schema_drift_is_rejected_cleanly(engine, tmp_path):
    ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    r = ing.ingest_file(engine, write(tmp_path / "b.csv", "visit_date,patient,dx\n2026-10-09,p1,flu\n"), "opd")
    assert r.status == "failed" and "vas" in r.message and count(engine, records) == 3
    assert ing.ingest_file(engine, tmp_path / "b.csv", "opd").status == "skipped"       # not retried forever


def test_extra_columns_are_ignored_and_reported(engine, tmp_path):
    ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    r = ing.ingest_file(engine, write(tmp_path / "b.csv", "visit_date,patient,dx,vas,ward\n2026-10-09,p7,flu,1.0,W1\n"), "opd")
    assert r.status == "loaded" and "ward" in r.message and "ward" not in load_dataframe(engine, "opd").columns


def test_bad_files_fail_with_a_message_not_a_crash(engine, tmp_path):
    assert "empty" in ing.ingest_file(engine, write(tmp_path / "e.csv", "visit_date,dx\n"), "opd").message.lower()
    assert ing.ingest_file(engine, write(tmp_path / "z.csv", ""), "opd").status == "failed"
    assert ing.ingest_file(engine, write(tmp_path / "n.csv", "a,b\n1,2\n"), "opd").message.startswith("No date column")
    assert ing.ingest_file(engine, write(tmp_path / "t.txt", "hello"), "opd").status == "failed"
    assert count(engine, records) == 0


def test_a_failure_part_way_keeps_nothing(engine, tmp_path, monkeypatch):
    real = ing.insert_ignore
    def flaky(conn, table, rows):
        if table is records:
            raise RuntimeError("disk full")
        return real(conn, table, rows)
    monkeypatch.setattr(ing, "insert_ignore", flaky)
    r = ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    assert r.status == "failed" and "disk full" in r.message
    assert count(engine, records) == 0 and count(engine, ing.datasets) == 0        # rolled back, dataset not registered
    assert count(engine, ingested_files, status="loaded") == 0 and count(engine, ingested_files, status="failed") == 1


def test_a_database_that_cannot_log_the_failure_does_not_crash_the_scan(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "insert_ignore", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database gone")))
    r = ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")             # must return, not raise
    assert r.status == "failed" and "database gone" in r.message and "could not be logged" in r.message


def test_unknown_dataset_gives_a_clear_error(engine):
    with pytest.raises(KeyError, match="Ingest a file"):
        load_dataframe(engine, "nope")


def test_scan_folder_uses_subfolders_as_datasets_and_skips_done_files(engine, tmp_path):
    inc = tmp_path / "incoming"
    write(inc / "opd" / "w1.csv", BASE)
    write(inc / "icu" / "w1.csv", BASE.replace("flu", "sepsis"))
    write(inc / "loose.csv", BASE.replace("flu", "x"))
    first = ing.scan_folder(engine, inc, min_age_seconds=0)
    assert sorted((r.dataset, r.status) for r in first) == [("default", "loaded"), ("icu", "loaded"), ("opd", "loaded")]
    assert {r.status for r in ing.scan_folder(engine, inc, min_age_seconds=0)} == {"skipped"}
    write(inc / "opd" / "w2.csv", BASE + "2026-10-05,p5,cold,2.0\n")
    again = {r.file: r for r in ing.scan_folder(engine, inc, min_age_seconds=0)}
    assert again["w2.csv"].rows_new == 1 and count(engine, records, dataset="opd") == 4


def test_files_still_being_written_wait_for_the_next_scan(engine, tmp_path):
    inc = tmp_path / "incoming"
    write(inc / "opd" / "w1.csv", BASE)
    r = ing.scan_folder(engine, inc, min_age_seconds=3600)
    assert [x.status for x in r] == ["pending"] and count(engine, records) == 0
    old = time.time() - 7200
    os.utime(inc / "opd" / "w1.csv", (old, old))
    assert [x.status for x in ing.scan_folder(engine, inc, min_age_seconds=3600)] == ["loaded"]


def test_office_temp_and_hidden_files_are_ignored(engine, tmp_path):
    inc = tmp_path / "incoming"
    write(inc / "opd" / "~$w1.xlsx", "lock"); write(inc / "opd" / ".hidden.csv", BASE); write(inc / "opd" / "notes.md", "x")
    assert ing.scan_folder(engine, inc, min_age_seconds=0) == []


def test_summary_reports_what_is_stored(engine, tmp_path):
    ing.ingest_file(engine, write(tmp_path / "a.csv", BASE), "opd")
    s = dataset_summary(engine)
    assert len(s) == 1 and s[0]["dataset"] == "opd" and s[0]["rows"] == 3 and s[0]["files"] == 1
    assert str(s[0]["first"]).startswith("2026-10-01") and str(s[0]["last"]).startswith("2026-10-03")


@needs_data
def test_generated_opd_round_trips_through_the_database(engine):
    from src.cleaning import clean_data, load_file
    expected, _ = clean_data(load_file(GEN / "opd_visits.csv"))
    r = ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    assert r.status == "loaded" and r.rows_new == len(expected)
    back = load_dataframe(engine, "opd")
    assert len(back) == len(expected) and back["visit_date"].min() == expected["visit_date"].min()
    assert sorted(back["visit_id"]) == sorted(expected["visit_id"])
    assert back["follow_up_vas"].round(2).sum() == pytest.approx(expected["follow_up_vas"].round(2).sum())


@needs_data
def test_dirty_file_is_cleaned_and_rejects_are_kept(engine):
    r = ing.ingest_file(engine, GEN / "opd_visits_dirty.csv", "opd")
    assert r.status == "loaded" and r.rows_rejected == 10 and count(engine, rejected_records) == 10
    assert r.rows_new < r.rows_in                                   # duplicates and rejects were not stored


@needs_data
def test_dashboard_pipeline_runs_on_database_rows(engine, tmp_path):
    """The existing engine works unchanged on data that came back out of the database."""
    import json
    from src.pipeline import make_dashboard, prepare
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    prep = prepare(load_dataframe(engine, "opd"))
    spec = json.loads((ROOT / "specs/opd_example.json").read_text())
    out = make_dashboard(prep, current=("2026-10-01", "2026-10-31"), baseline=("2026-09-01", "2026-09-30"), spec=spec)
    assert out["insights"]["findings"] and "Total OPD visits" in out["html"]