"""Database layer for live mode. SQLite by default (no setup), PostgreSQL by changing one setting:

    DATABASE_URL=sqlite:///data/healthdash.db                       (default)
    DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/healthdash

Tables
    datasets          one row per dataset: its date column and column list (a new file must match)
    ingested_files    every file seen, with a content hash so no file is loaded twice
    records           the cleaned rows, one JSON document each, with arrival time; unique per row content
    rejected_records  rows that could not be used (for example an unreadable date), with the reason
    dataset_specs     the dashboard layout chosen for each dataset (kept stable between refreshes)
    snapshots         each refresh's dashboard numbers, findings and summary
    pipeline_runs     the run log, one row per refresh attempt

    python -m src.db init      create the tables
    python -m src.db status    show what is stored
"""
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import (JSON, Column, DateTime, ForeignKey, Index, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, event, func, select)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import make_url

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = f"sqlite:///{(ROOT / 'data' / 'healthdash.db').as_posix()}"

metadata = MetaData()
Json = JSON().with_variant(postgresql.JSONB(), "postgresql")
utcnow = lambda: datetime.now(timezone.utc).replace(tzinfo=None)

datasets = Table("datasets", metadata,
    Column("name", String(100), primary_key=True),
    Column("date_column", String(200), nullable=False),
    Column("columns", Json, nullable=False),
    Column("created_at", DateTime, default=utcnow))

ingested_files = Table("ingested_files", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(100), nullable=False),
    Column("filename", String(500)),
    Column("file_hash", String(64), nullable=False),
    Column("status", String(20), nullable=False),            # loaded | failed
    Column("rows_in", Integer, default=0), Column("rows_new", Integer, default=0),
    Column("rows_duplicate", Integer, default=0), Column("rows_rejected", Integer, default=0),
    Column("message", Text),
    Column("ingested_at", DateTime, default=utcnow),
    UniqueConstraint("dataset", "file_hash", name="uq_file_per_dataset"))

records = Table("records", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(100), nullable=False),
    Column("row_hash", String(64), nullable=False),
    Column("event_ts", DateTime, nullable=False),
    Column("data", Json, nullable=False),
    Column("ingested_at", DateTime, default=utcnow),
    Column("file_id", Integer, ForeignKey("ingested_files.id")),
    UniqueConstraint("dataset", "row_hash", name="uq_row_per_dataset"),
    Index("ix_records_dataset_ts", "dataset", "event_ts"))

rejected_records = Table("rejected_records", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(100), nullable=False),
    Column("file_id", Integer, ForeignKey("ingested_files.id")),
    Column("reason", String(200)),
    Column("data", Json),
    Column("created_at", DateTime, default=utcnow))

snapshots = Table("snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(100), nullable=False),
    Column("created_at", DateTime, default=utcnow),
    Column("window_label", String(50)),
    Column("payload", Json, nullable=False),
    Column("findings_hash", String(64)),
    Index("ix_snapshots_dataset_created", "dataset", "created_at"))

dataset_specs = Table("dataset_specs", metadata,           # the dashboard layout is chosen once and kept, so it never jumps between refreshes
    Column("dataset", String(100), primary_key=True),
    Column("spec", Json, nullable=False),
    Column("source", String(20)),                         # given | file | llm | fallback
    Column("updated_at", DateTime, default=utcnow))

pipeline_runs = Table("pipeline_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(100), nullable=False),
    Column("started_at", DateTime, default=utcnow),
    Column("finished_at", DateTime),
    Column("status", String(20)),
    Column("rows_new", Integer, default=0),
    Column("message", Text),
    Column("snapshot_id", Integer, ForeignKey("snapshots.id")))


def get_engine(url=None, **engine_options):
    """Engine for DATABASE_URL (or the default SQLite file). Creates the tables if they do not exist."""
    url = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite") and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **engine_options)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):              # WAL lets the worker write while the app reads
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    metadata.create_all(engine)
    return engine


def insert_ignore(conn, table, rows):
    """INSERT ... ON CONFLICT DO NOTHING on either database."""
    if rows:
        ins = (postgresql.insert if conn.dialect.name == "postgresql" else sqlite.insert)(table).on_conflict_do_nothing()
        conn.execute(ins, rows)


def get_dataset(conn, name):
    row = conn.execute(select(datasets).where(datasets.c.name == name)).mappings().first()
    return dict(row) if row else None


def load_dataframe(engine, dataset, start=None, end=None):
    """Clean rows of a dataset as a DataFrame (date column parsed), optionally for start..end (days, inclusive)."""
    with engine.connect() as conn:
        ds = get_dataset(conn, dataset)
        if ds is None:
            raise KeyError(f"Unknown dataset '{dataset}'. Ingest a file for it first.")
        q = select(records.c.data).where(records.c.dataset == dataset).order_by(records.c.event_ts, records.c.id)
        if start is not None:
            q = q.where(records.c.event_ts >= pd.Timestamp(start).to_pydatetime())
        if end is not None:
            q = q.where(records.c.event_ts < (pd.Timestamp(end) + timedelta(days=1)).to_pydatetime())
        rows = [r[0] for r in conn.execute(q)]
    df = pd.DataFrame(rows, columns=ds["columns"])
    df[ds["date_column"]] = pd.to_datetime(df[ds["date_column"]])
    return df


def dataset_summary(engine):
    """One dict per dataset: rows stored, date range, files loaded, last ingest time."""
    with engine.connect() as conn:
        out = []
        for ds in conn.execute(select(datasets.c.name, datasets.c.date_column)).all():
            n, first, last = conn.execute(select(func.count(), func.min(records.c.event_ts), func.max(records.c.event_ts))
                                          .where(records.c.dataset == ds.name)).one()
            files, seen = conn.execute(select(func.count(), func.max(ingested_files.c.ingested_at))
                                       .where(ingested_files.c.dataset == ds.name, ingested_files.c.status == "loaded")).one()
            out.append({"dataset": ds.name, "rows": n, "first": first, "last": last, "files": files, "last_ingest": seen})
        return out


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1:]
    if cmd not in (["init"], ["status"]):
        sys.exit("Usage: python -m src.db init | status")
    eng = get_engine()
    print(f"Database: {eng.url.render_as_string(hide_password=True)}")
    if cmd == ["status"]:
        rows = dataset_summary(eng)
        for r in rows or []:
            print(f"  {r['dataset']}: {r['rows']} rows, {r['first']} to {r['last']}, {r['files']} files, last ingest {r['last_ingest']}")
        if not rows:
            print("  (no datasets yet; put a CSV in data/incoming/ and run: python -m src.ingest)")