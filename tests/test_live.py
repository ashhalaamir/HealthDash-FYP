import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ingest as ing
from src import live
from src.db import dataset_specs, get_engine, pipeline_runs, snapshots

GEN = ROOT / "data/generated"
pytestmark = pytest.mark.skipif(not (GEN / "opd_visits.csv").exists(), reason="run scripts/generate_fake_data.py first")
OPD_SPEC = json.loads((ROOT / "specs/opd_example.json").read_text()) if (ROOT / "specs/opd_example.json").exists() else None


@pytest.fixture
def parts(tmp_path):
    """The generated OPD data split in two: everything up to Oct 13, and everything after (which holds the Oct 14 spike)."""
    df = pd.read_csv(GEN / "opd_visits.csv")
    a, b = tmp_path / "part1.csv", tmp_path / "part2.csv"
    df[df.visit_date <= "2026-10-13"].to_csv(a, index=False)
    df[df.visit_date > "2026-10-13"].to_csv(b, index=False)
    return a, b


def n(engine, table):
    with engine.connect() as c:
        return c.execute(select(func.count()).select_from(table)).scalar()


def fake_llm(calls):
    def llm(prompt):
        kind = "spec" if "design analytics dashboards" in prompt else "summary"
        calls.append(kind)
        return json.dumps(OPD_SPEC) if kind == "spec" else json.dumps({"summary": "Visits changed over the period."})
    return llm


def test_first_refresh_builds_and_stores_a_snapshot(engine):
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    r = live.run_once(engine, "opd", "30d")
    assert r.status == "ok" and r.snapshot_id and r.rows_new == 372 and r.summary_source == "template"
    snap = live.latest_snapshot(engine, "opd")
    meta = snap["payload"]["meta"]
    assert meta["current"] == ["2026-10-02", "2026-10-31"] and meta["baseline"] == ["2026-09-02", "2026-10-01"]
    assert snap["payload"]["insights"]["findings"] and snap["payload"]["since_last"] is None
    assert "<html" in live.snapshot_html(snap)
    with engine.connect() as c:
        assert c.execute(select(pipeline_runs.c.status)).scalar() == "ok"


def test_nothing_new_means_no_new_snapshot_and_no_llm_call(engine):
    calls = []
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    live.run_once(engine, "opd", "30d", llm=fake_llm(calls))
    before = list(calls)
    r = live.run_once(engine, "opd", "30d", llm=fake_llm(calls))
    assert r.status == "no_change" and n(engine, snapshots) == 1 and n(engine, pipeline_runs) == 2 and calls == before


def test_new_data_gives_a_new_snapshot_that_sees_the_spike_and_reports_the_change(engine, parts):
    a, b = parts
    ing.ingest_file(engine, a, "opd")
    first = live.run_once(engine, "opd", "30d")
    before = live.latest_snapshot(engine, "opd")["payload"]
    assert not any("Oct 14" in f["text"] for f in before["insights"]["findings"])
    added = ing.ingest_file(engine, b, "opd").rows_new
    r = live.run_once(engine, "opd", "30d")
    assert r.status == "ok" and r.rows_new == added and r.snapshot_id != first.snapshot_id
    after = live.latest_snapshot(engine, "opd")["payload"]
    assert any(f["kind"] == "spike" and "Oct 14" in f["text"] for f in after["insights"]["findings"])
    assert after["meta"]["current"][1] == "2026-10-31" and before["meta"]["current"][1] == "2026-10-13"
    assert after["since_last"]["rows_new"] == added and isinstance(after["since_last"]["findings"], list)
    assert after["since_last"]["window_change"] != 0
    assert live.latest_snapshot(engine, "opd", offset=1)["id"] == first.snapshot_id


def test_llm_is_asked_once_and_the_summary_is_reused_while_findings_are_unchanged(engine):
    calls = []
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    first = live.run_once(engine, "opd", "30d", llm=fake_llm(calls))
    assert first.summary_source == "llm" and calls.count("summary") == 1
    again = live.run_once(engine, "opd", "30d", llm=fake_llm(calls), force=True)
    assert again.status == "ok" and again.summary_source == "reused" and calls.count("summary") == 1
    assert n(engine, snapshots) == 2
    assert live.latest_snapshot(engine, "opd")["payload"]["insights"]["summary"] == "Visits changed over the period."


def test_layout_is_chosen_once_and_kept(engine, parts):
    a, b = parts
    calls = []
    ing.ingest_file(engine, a, "opd")
    live.run_once(engine, "opd", "30d", llm=fake_llm(calls))
    assert calls.count("spec") == 1
    ing.ingest_file(engine, b, "opd")
    live.run_once(engine, "opd", "30d", llm=fake_llm(calls))
    assert calls.count("spec") == 1 and n(engine, dataset_specs) == 1
    ids = lambda s: [w["id"] for w in s["payload"]["spec"]["widgets"]]
    assert ids(live.latest_snapshot(engine, "opd")) == ids(live.latest_snapshot(engine, "opd", 1))


def test_failing_llm_still_produces_a_snapshot_with_an_explanation(engine):
    def broken(_): raise RuntimeError("quota exceeded")
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    r = live.run_once(engine, "opd", "30d", llm=broken)
    assert r.status == "ok" and r.summary_source == "template"
    ins = live.latest_snapshot(engine, "opd")["payload"]["insights"]
    assert ins["source"] == "template" and "quota exceeded" in ins["notes"][0]
    with engine.connect() as c:
        assert c.execute(select(dataset_specs.c.source)).scalar() == "fallback"


def test_spec_from_a_file_and_an_explicit_spec(engine, tmp_path, monkeypatch):
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    (tmp_path / "opd.json").write_text(json.dumps({**OPD_SPEC, "title": "FROM FILE"}))
    monkeypatch.setattr(live, "SPEC_DIR", tmp_path)
    live.run_once(engine, "opd", "30d")
    assert live.latest_snapshot(engine, "opd")["payload"]["spec"]["title"] == "FROM FILE"
    live.run_once(engine, "opd", "30d", spec={**OPD_SPEC, "title": "EXPLICIT"})
    assert live.latest_snapshot(engine, "opd")["payload"]["spec"]["title"] == "EXPLICIT"


def test_problems_are_logged_and_returned_not_raised(engine):
    assert live.run_once(engine, "ghost", "7d").status == "failed"
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    bad = live.run_once(engine, "opd", "7x")
    assert bad.status == "failed" and "Window must look like" in bad.message
    empty = live.run_once(engine, "opd", "7d", as_of="2030-01-01")
    assert empty.status == "no_data" and n(engine, snapshots) == 0
    with engine.connect() as c:
        assert set(c.execute(select(pipeline_runs.c.status)).scalars()) == {"failed", "no_data"}


def test_an_error_inside_the_engine_is_contained(engine, monkeypatch):
    ing.ingest_file(engine, GEN / "opd_visits.csv", "opd")
    monkeypatch.setattr(live, "make_dashboard", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine broke")))
    r = live.run_once(engine, "opd", "30d")
    assert r.status == "failed" and "engine broke" in r.message and n(engine, snapshots) == 0


def test_window_helpers():
    assert live.parse_window("7d") == 7 and live.parse_window("2w") == 14 and live.parse_window(" 30D ") == 30
    for bad in ("0d", "7", "d7", "400d", "1h", ""):
        with pytest.raises(ValueError):
            live.parse_window(bad)
    assert live.rolling_windows("2026-10-31", 7) == (("2026-10-25", "2026-10-31"), ("2026-10-18", "2026-10-24"))