import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import worker
from src.db import get_engine, snapshots

GEN = ROOT / "data/generated"
pytestmark = pytest.mark.skipif(not (GEN / "opd_visits.csv").exists(), reason="run scripts/generate_fake_data.py first")


@pytest.fixture
def inbox(tmp_path):
    """An incoming folder with the OPD data split in two files, the second held back until the test wants it."""
    df = pd.read_csv(GEN / "opd_visits.csv")
    first = tmp_path / "incoming" / "opd"
    first.mkdir(parents=True)
    df[df.visit_date <= "2026-10-13"].to_csv(first / "part1.csv", index=False)
    df[df.visit_date > "2026-10-13"].to_csv(tmp_path / "part2.csv", index=False)
    return tmp_path / "incoming", tmp_path / "part2.csv"


def snaps(engine):
    with engine.connect() as c:
        return c.execute(select(func.count()).select_from(snapshots)).scalar()


def test_parse_interval():
    assert [worker.parse_interval(x) for x in ("30m", "2h", "1h30m", "90s", " 5 M ")] == [1800, 7200, 5400, 90, 300]
    for bad in ("", "30", "m", "0s", "1d", "abc", "-5m"):
        with pytest.raises(ValueError):
            worker.parse_interval(bad)


def test_a_cycle_loads_files_and_builds_the_first_snapshot(engine, inbox):
    folder, _ = inbox
    r = worker.cycle(engine, folder, "30d", min_age_seconds=0)
    assert r.ok and [x.status for x in r.ingested] == ["loaded"] and [x.status for x in r.refreshed] == ["ok"]
    assert snaps(engine) == 1 and "1 file(s) loaded" in r.line() and "OK" in r.line()


def test_an_idle_cycle_is_cheap_and_a_new_file_triggers_a_new_snapshot(engine, inbox):
    folder, part2 = inbox
    worker.cycle(engine, folder, "30d", min_age_seconds=0)
    idle = worker.cycle(engine, folder, "30d", min_age_seconds=0)
    assert [x.status for x in idle.ingested] == ["skipped"] and [x.status for x in idle.refreshed] == ["no_change"]
    assert snaps(engine) == 1
    shutil.move(part2, folder / "opd" / "part2.csv")
    busy = worker.cycle(engine, folder, "30d", min_age_seconds=0)
    assert [x.status for x in busy.refreshed] == ["ok"] and snaps(engine) == 2


def test_a_file_still_being_written_is_picked_up_next_cycle(engine, inbox):
    folder, _ = inbox
    first = worker.cycle(engine, folder, "30d", min_age_seconds=3600)
    assert [x.status for x in first.ingested] == ["pending"] and first.refreshed == [] and "still being written" in first.line()
    old = time.time() - 7200
    os.utime(folder / "opd" / "part1.csv", (old, old))
    assert [x.status for x in worker.cycle(engine, folder, "30d", min_age_seconds=3600).refreshed] == ["ok"]


def test_only_filter_limits_which_datasets_refresh(engine, inbox):
    folder, _ = inbox
    shutil.copy(folder / "opd" / "part1.csv", (folder / "icu").__class__(folder / "icu_part.csv"))   # lands in 'default'
    worker.cycle(engine, folder, "30d", min_age_seconds=0)
    r = worker.cycle(engine, folder, "30d", only=["opd"], min_age_seconds=0, force=True)
    assert [x.dataset for x in r.refreshed] == ["opd"]


def test_a_broken_folder_does_not_stop_the_refresh(engine, inbox, monkeypatch):
    folder, _ = inbox
    worker.cycle(engine, folder, "30d", min_age_seconds=0)
    monkeypatch.setattr(worker, "ingest_step", lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")))
    r = worker.cycle(engine, folder, "30d", force=True)
    assert not r.ok and "denied" in r.errors[0] and [x.status for x in r.refreshed] == ["ok"]


def test_a_failed_refresh_is_reported_not_raised(engine, inbox):
    folder, _ = inbox
    r = worker.cycle(engine, folder, "bogus", min_age_seconds=0)
    assert not r.ok and r.refreshed[0].status == "failed" and "Window must look like" in r.line()


def test_run_forever_waits_between_cycles_and_stops_cleanly(engine, inbox):
    folder, _ = inbox
    waits, lines = [], []
    n = worker.run_forever(engine, 1800, max_cycles=3, sleep=waits.append, log=lines.append,
                           folder=folder, window="30d", min_age_seconds=0)
    assert n == 3 and waits == [1800, 1800] and len(lines) == 3 and "OK" in lines[0] and "NO_CHANGE" in lines[1]


def test_ctrl_c_stops_the_worker_without_a_traceback(engine, inbox):
    folder, _ = inbox
    lines = []
    def interrupt(_): raise KeyboardInterrupt
    n = worker.run_forever(engine, 60, sleep=interrupt, log=lines.append, folder=folder, window="30d", min_age_seconds=0)
    assert n == 1 and lines[-1] == "Worker stopped."