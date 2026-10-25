"""Phase 4: the scheduler. Scan for new files, load them, refresh dashboards, repeat.

    python -m src.worker --every 30m              keep running, one cycle every 30 minutes
    python -m src.worker --once                   one cycle, then exit (for Task Scheduler, cron or a test)
    python -m src.worker --every 2m --llm         faster demo cycle, LLM writes layout (first time) and summaries

The two steps are separate functions so Airflow can run them as separate tasks:
    ingest_step(engine)  -> files into the database
    refresh_step(engine) -> a new snapshot for every dataset that changed
A cycle never raises: problems are logged and returned, and the next cycle simply tries again.
Run ONE worker per database. A second one is harmless (it finds nothing changed) but wasteful.
"""
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select

from .db import datasets, get_engine
from .ingest import INCOMING, scan_folder
from .live import run_once


def parse_interval(text):
    """'30m' -> 1800, '2h' -> 7200, '1h30m' -> 5400, '90s' -> 90 (seconds)."""
    t = str(text).lower().replace(" ", "")
    if not t or not re.fullmatch(r"(\d+[smh])+", t):
        raise ValueError(f"Interval must look like 30s, 15m, 2h or 1h30m; got '{text}'.")
    seconds = sum(int(n) * {"s": 1, "m": 60, "h": 3600}[u] for n, u in re.findall(r"(\d+)([smh])", t))
    if seconds < 1:
        raise ValueError("Interval must be at least 1 second.")
    return seconds


@dataclass
class CycleResult:
    started: datetime
    ingested: list = field(default_factory=list)       # IngestResult per file seen
    refreshed: list = field(default_factory=list)      # RunResult per dataset
    errors: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.errors and all(r.status != "failed" for r in self.refreshed)

    def line(self):
        loaded = [r for r in self.ingested if r.status == "loaded"]
        failed = [r for r in self.ingested if r.status == "failed"]
        pending = sum(r.status == "pending" for r in self.ingested)
        parts = [f"{len(loaded)} file(s) loaded ({sum(r.rows_new for r in loaded)} new rows)"]
        if failed:
            parts.append(f"{len(failed)} file(s) FAILED: " + "; ".join(f"{r.file}: {r.message}" for r in failed))
        if pending:
            parts.append(f"{pending} still being written")
        parts += [r.line() for r in self.refreshed] + [f"ERROR {e}" for e in self.errors]
        return f"[{self.started:%Y-%m-%d %H:%M:%S}] " + " | ".join(parts)


def ingest_step(engine, folder=INCOMING, min_age_seconds=2):
    return scan_folder(engine, folder, min_age_seconds)


def refresh_step(engine, window="7d", llm=None, only=None, force=False):
    """A snapshot for every dataset in the database (or just `only`); unchanged ones are skipped cheaply."""
    with engine.connect() as c:
        names = list(c.execute(select(datasets.c.name).order_by(datasets.c.name)).scalars())
    return [run_once(engine, n, window, llm=llm, force=force) for n in names if not only or n in only]


def cycle(engine, folder=INCOMING, window="7d", llm=None, only=None, min_age_seconds=2, force=False):
    res = CycleResult(started=datetime.now())
    try:
        res.ingested = ingest_step(engine, folder, min_age_seconds)
    except Exception as e:                              # e.g. folder permissions: still refresh what we have
        res.errors.append(f"scanning {folder}: {type(e).__name__}: {e}")
    try:
        res.refreshed = refresh_step(engine, window, llm, only, force)
    except Exception as e:
        res.errors.append(f"refreshing: {type(e).__name__}: {e}")
    return res


def run_forever(engine, every_seconds, max_cycles=None, sleep=time.sleep, log=print, **kwargs):
    """Cycle, wait, repeat until Ctrl+C (or max_cycles, used by tests). Returns the number of cycles run."""
    n = 0
    try:
        while max_cycles is None or n < max_cycles:
            log(cycle(engine, **kwargs).line())
            n += 1
            if max_cycles is None or n < max_cycles:
                sleep(every_seconds)
    except KeyboardInterrupt:
        log("Worker stopped.")
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Scan for new data files and keep dashboard snapshots fresh.")
    ap.add_argument("--every", default="30m", help="time between cycles: 30s, 15m, 2h, 1h30m (default 30m)")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    ap.add_argument("--window", default="7d", help="rolling window: 7d, 30d, 2w (default 7d)")
    ap.add_argument("--folder", default=str(INCOMING), help="folder to watch (default data/incoming)")
    ap.add_argument("--datasets", nargs="*", help="only refresh these datasets")
    ap.add_argument("--llm", action="store_true", help="use the LLM from .env (layout once, summary when findings change)")
    ap.add_argument("--force", action="store_true", help="build snapshots even when nothing changed")
    a = ap.parse_args()
    llm = None
    if a.llm:
        from .llm import call_llm
        llm = call_llm
    engine = get_engine()
    print(f"Database: {engine.url.render_as_string(hide_password=True)} | watching {a.folder} | window {a.window}")
    kw = dict(folder=a.folder, window=a.window, llm=llm, only=a.datasets, force=a.force)
    if a.once:
        print(cycle(engine, **kw).line())
    else:
        secs = parse_interval(a.every)
        print(f"Cycle every {a.every}. Press Ctrl+C to stop.")
        run_forever(engine, secs, **kw)


if __name__ == "__main__":
    main()