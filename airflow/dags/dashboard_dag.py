"""Airflow DAG: every 30 minutes, load new files, then refresh every dataset's dashboard snapshot.

    ingest_new_files  ->  refresh_snapshots

It calls the same functions as `python -m src.worker`, so behaviour is identical with or without Airflow.
Settings come from environment variables (set them on the Airflow containers):
    HEALTHDASH_HOME      folder containing src/ (default: two levels above this file)
    HEALTHDASH_SCHEDULE  cron expression (default */30 * * * *)
    HEALTHDASH_WINDOW    rolling window, e.g. 7d (default 7d)
    HEALTHDASH_USE_LLM   1 to let the LLM write layouts and summaries (needs the key in the same environment)
    DATABASE_URL         the database (Airflow's containers need a URL they can reach)
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

HOME = Path(os.environ.get("HEALTHDASH_HOME") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(HOME))

try:                                                    # Airflow 3
    from airflow.sdk import dag, task
except ImportError:                                     # Airflow 2
    from airflow.decorators import dag, task


def ingest_new_files():
    """Load every new file from data/incoming. A bad file is logged in the database, it does not fail the task."""
    from src.db import get_engine
    from src.worker import ingest_step
    results = ingest_step(get_engine(), HOME / "data" / "incoming")
    for r in results:
        print(r.line())
    return {"loaded": sum(r.status == "loaded" for r in results), "failed": sum(r.status == "failed" for r in results),
            "new_rows": sum(r.rows_new for r in results)}


def refresh_snapshots(ingest_summary=None):
    """Refresh every dataset. If any refresh fails the task fails, so it shows red in Airflow and is retried."""
    from src.db import get_engine
    from src.worker import refresh_step
    llm = None
    if os.environ.get("HEALTHDASH_USE_LLM") == "1":
        from src.llm import call_llm
        llm = call_llm
    results = refresh_step(get_engine(), os.environ.get("HEALTHDASH_WINDOW", "7d"), llm)
    for r in results:
        print(r.line())
    failed = [r for r in results if r.status == "failed"]
    if failed:
        raise RuntimeError("Refresh failed: " + "; ".join(f"{r.dataset}: {r.message}" for r in failed))
    return {r.dataset: r.status for r in results}


@dag(dag_id="healthdash_refresh", schedule=os.environ.get("HEALTHDASH_SCHEDULE", "*/30 * * * *"),
     start_date=datetime(2026, 1, 1), catchup=False, max_active_runs=1,       # never two refreshes at once
     default_args={"retries": 2, "retry_delay": timedelta(minutes=2)}, tags=["healthdash"])
def healthdash_refresh():
    refresh = task(refresh_snapshots)(task(ingest_new_files)())
    return refresh


healthdash_refresh()