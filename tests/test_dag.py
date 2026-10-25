"""The Airflow DAG file: its logic is tested without Airflow; a real Airflow parse runs if Airflow is installed."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DAG_FILE = ROOT / "airflow" / "dags" / "dashboard_dag.py"
GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "opd_visits.csv").exists(), reason="run scripts/generate_fake_data.py first")


@pytest.fixture
def dag_module(monkeypatch):
    """Import the DAG file against a stand-in 'airflow' that records how the DAG is declared instead of running it."""
    seen = {"dag": {}, "calls": []}
    def dag(**kw):
        seen["dag"] = kw
        return lambda fn: fn
    def task(fn):
        return lambda *a: (seen["calls"].append((fn.__name__, a)), f"<{fn.__name__}>")[1]
    sdk = types.ModuleType("airflow.sdk"); sdk.dag, sdk.task = dag, task
    monkeypatch.setitem(sys.modules, "airflow", types.ModuleType("airflow"))
    monkeypatch.setitem(sys.modules, "airflow.sdk", sdk)
    spec = importlib.util.spec_from_file_location("dashboard_dag_under_test", DAG_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, seen


def test_dag_declaration(dag_module):
    _, seen = dag_module
    kw = seen["dag"]
    assert kw["dag_id"] == "healthdash_refresh" and kw["schedule"] == "*/30 * * * *"
    assert kw["catchup"] is False and kw["max_active_runs"] == 1 and kw["default_args"]["retries"] == 2


def test_tasks_run_in_order_ingest_then_refresh(dag_module):
    _, seen = dag_module
    assert seen["calls"] == [("ingest_new_files", ()), ("refresh_snapshots", ("<ingest_new_files>",))]


def test_schedule_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("HEALTHDASH_SCHEDULE", "*/2 * * * *")
    seen = {}
    sdk = types.ModuleType("airflow.sdk"); sdk.dag = lambda **kw: (seen.update(kw), (lambda fn: fn))[1]; sdk.task = lambda fn: (lambda *a: None)
    monkeypatch.setitem(sys.modules, "airflow", types.ModuleType("airflow")); monkeypatch.setitem(sys.modules, "airflow.sdk", sdk)
    spec = importlib.util.spec_from_file_location("dag_sched", DAG_FILE); spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert seen["schedule"] == "*/2 * * * *"


@needs_data
def test_task_functions_do_the_real_work(dag_module, tmp_path, monkeypatch):
    mod, _ = dag_module
    monkeypatch.setattr(mod, "HOME", tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'af.db').as_posix()}")
    monkeypatch.setenv("HEALTHDASH_WINDOW", "30d")
    (tmp_path / "data" / "incoming" / "opd").mkdir(parents=True)
    pd.read_csv(GEN / "opd_visits.csv").to_csv(tmp_path / "data" / "incoming" / "opd" / "w1.csv", index=False)
    import os, time
    old = time.time() - 600
    os.utime(tmp_path / "data" / "incoming" / "opd" / "w1.csv", (old, old))
    summary = mod.ingest_new_files()
    assert summary == {"loaded": 1, "failed": 0, "new_rows": 372}
    json.dumps(summary)                                           # must be storable in Airflow's task results
    assert mod.refresh_snapshots(summary) == {"opd": "ok"}
    assert mod.refresh_snapshots(summary) == {"opd": "no_change"}


def test_a_failed_refresh_fails_the_task_so_airflow_shows_it(dag_module, tmp_path, monkeypatch):
    mod, _ = dag_module
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'af.db').as_posix()}")
    monkeypatch.setenv("HEALTHDASH_WINDOW", "bogus")
    from src.db import get_engine
    from src.ingest import ingest_file
    f = tmp_path / "a.csv"; f.write_text("visit_date,dx\n2026-10-01,a\n2026-10-02,b\n")
    ingest_file(get_engine(), f, "opd")
    with pytest.raises(RuntimeError, match="Refresh failed.*Window must look like"):
        mod.refresh_snapshots()


def test_real_airflow_parses_the_dag(monkeypatch):
    try:                                    # not importorskip("airflow"): our own airflow/ folder also imports as 'airflow'
        from airflow.models import DagBag
    except ImportError:
        pytest.skip("Apache Airflow is not installed")
    bag = DagBag(dag_folder=str(DAG_FILE.parent))           # no include_examples: removed in Airflow 3
    assert {k: v for k, v in bag.import_errors.items() if "dashboard_dag" in k} == {}
    d = bag.get_dag("healthdash_refresh")
    assert sorted(t.task_id for t in d.tasks) == ["ingest_new_files", "refresh_snapshots"]
    assert d.catchup is False and d.max_active_runs == 1
    assert {t.task_id for t in d.get_task("ingest_new_files").downstream_list} == {"refresh_snapshots"}