"""Phase 2: files in, clean rows in the database. Reuses the same cleaner as the upload flow.

    ingest_file(engine, "data/incoming/opd/week1.csv")     one file
    scan_folder(engine, "data/incoming")                    every new file (subfolder name = dataset name)

Guarantees
  * A file is loaded once: it is recognised by a hash of its content, whatever it is called.
  * A row is stored once per dataset, even if it arrives in a later file (unique row hash).
  * A file is all-or-nothing: if anything fails part way, nothing from it is kept.
  * A new file must match the dataset's columns; one that does not is rejected with a clear message.
  * Rows with an unreadable date are not lost silently: they go to rejected_records with the reason.
  * Files still being written (modified a moment ago) are left for the next scan.

    python -m src.ingest                         scan data/incoming once
    python -m src.ingest path/to/file.csv --dataset opd
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from .cleaning import DataValidationError, clean_data, load_file, parse_dates
from .db import (ROOT, datasets, get_dataset, get_engine, ingested_files, insert_ignore, records, rejected_records, utcnow)

INCOMING = ROOT / "data" / "incoming"
SUFFIXES = {".csv", ".xlsx", ".xls"}
MAX_REJECTS_STORED = 1000
CHUNK = 500


@dataclass
class IngestResult:
    file: str
    dataset: str
    status: str                 # loaded | skipped | failed | pending
    rows_in: int = 0
    rows_new: int = 0
    rows_duplicate: int = 0
    rows_rejected: int = 0
    message: str = ""

    def line(self):
        extra = f" ({self.message})" if self.message else ""
        return (f"{self.status.upper():8} {self.dataset}/{self.file}: in {self.rows_in}, new {self.rows_new}, "
                f"duplicate {self.rows_duplicate}, rejected {self.rows_rejected}{extra}")


def _row_hash(rec):
    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()


def _existing_hashes(conn, dataset, hashes):
    found = set()
    for i in range(0, len(hashes), CHUNK):
        part = hashes[i:i + CHUNK]
        found |= {r[0] for r in conn.execute(select(records.c.row_hash).where(
            records.c.dataset == dataset, records.c.row_hash.in_(part)))}
    return found


def _record_failure(engine, dataset, name, file_hash, rows_in, message):
    try:
        with engine.begin() as conn:
            insert_ignore(conn, ingested_files, [dict(dataset=dataset, filename=name, file_hash=file_hash, status="failed",
                                                      rows_in=rows_in, message=message, ingested_at=utcnow())])
    except Exception as e:                              # a database problem must not crash a whole folder scan
        message += f" (the failure could not be logged: {type(e).__name__}: {e})"
    return IngestResult(name, dataset, "failed", rows_in=rows_in, message=message)


def _store(engine, raw, dataset, name, file_hash, date_col):
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    with engine.connect() as conn:
        ds = get_dataset(conn, dataset)
    clean, rep = clean_data(raw, date_col=ds["date_column"] if ds else date_col,
                            required_columns=ds["columns"] if ds else None)
    date_col = rep["date_column"]

    notes = []
    if ds:                                              # keep every file in the dataset's shape
        extra = [c for c in clean.columns if c not in ds["columns"]]
        if extra:
            clean = clean.drop(columns=extra)
            notes.append(f"ignored new column(s): {', '.join(extra)}")
        clean = clean[ds["columns"]]

    docs = json.loads(clean.to_json(orient="records", date_format="iso", date_unit="s"))
    stamps = [t.to_pydatetime() for t in clean[date_col]]
    hashes = [_row_hash(d) for d in docs]

    bad = parse_dates(raw[date_col]).isna()
    reject_docs = json.loads(raw[bad].head(MAX_REJECTS_STORED).to_json(orient="records", date_format="iso"))
    rejects = [dict(dataset=dataset, reason="Unreadable or missing date", created_at=utcnow(), data=d) for d in reject_docs]

    with engine.begin() as conn:                        # all or nothing
        if ds is None:
            conn.execute(insert(datasets).values(name=dataset, date_column=date_col,
                                                 columns=list(clean.columns), created_at=utcnow()))
        seen = _existing_hashes(conn, dataset, hashes)
        fresh = [(h, d, t) for h, d, t in zip(hashes, docs, stamps) if h not in seen]
        dupes = len(docs) - len(fresh)
        file_id = conn.execute(insert(ingested_files).values(
            dataset=dataset, filename=name, file_hash=file_hash, status="loaded", rows_in=len(raw), rows_new=len(fresh),
            rows_duplicate=dupes, rows_rejected=int(bad.sum()), message="; ".join(notes) or None,
            ingested_at=utcnow())).inserted_primary_key[0]
        insert_ignore(conn, records, [dict(dataset=dataset, row_hash=h, event_ts=t, data=d, ingested_at=utcnow(), file_id=file_id)
                                      for h, d, t in fresh])
        for r in rejects:
            r["file_id"] = file_id
        insert_ignore(conn, rejected_records, rejects)
    return IngestResult(name, dataset, "loaded", len(raw), len(fresh), dupes, int(bad.sum()), "; ".join(notes))


def ingest_frame(engine, raw, dataset, name, file_hash, date_col=None):
    """Ingest an in-memory DataFrame (used by the app's 'save to database' button). Same rules as files."""
    with engine.connect() as conn:
        prior = conn.execute(select(ingested_files.c.status).where(
            ingested_files.c.dataset == dataset, ingested_files.c.file_hash == file_hash)).first()
    if prior:
        return IngestResult(name, dataset, "skipped", message=f"already processed ({prior[0]})")
    try:
        return _store(engine, raw, dataset, name, file_hash, date_col)
    except IntegrityError:                              # another worker loaded the same file a moment ago
        return IngestResult(name, dataset, "skipped", message="already processed")
    except DataValidationError as e:
        return _record_failure(engine, dataset, name, file_hash, len(raw), str(e))
    except Exception as e:
        return _record_failure(engine, dataset, name, file_hash, len(raw), f"Unexpected error: {type(e).__name__}: {e}")


def dataset_for(path, root=INCOMING):
    """Subfolder name under the incoming folder is the dataset; files directly inside it are 'default'."""
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return "default"
    return parts[0] if len(parts) > 1 else "default"


def ingest_file(engine, path, dataset=None, date_col=None):
    path = Path(path)
    dataset = dataset or dataset_for(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        raw = load_file(path)
    except DataValidationError as e:
        return _record_failure(engine, dataset, path.name, file_hash, 0, str(e))
    except Exception as e:                              # unreadable, empty, or not really a CSV
        return _record_failure(engine, dataset, path.name, file_hash, 0, f"Could not read the file: {e}")
    return ingest_frame(engine, raw, dataset, path.name, file_hash, date_col)


def ingest_bytes(engine, filename, content, dataset):
    """Ingest an uploaded file held in memory. Same rules as a file on disk."""
    import tempfile
    path = Path(tempfile.mkdtemp()) / Path(filename).name
    path.write_bytes(content)
    return ingest_file(engine, path, dataset)


def scan_folder(engine, folder=INCOMING, min_age_seconds=2):
    """Ingest every new file under `folder`, oldest first. Returns one IngestResult per file seen."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES
                    and not p.name.startswith(("~$", "."))), key=lambda p: (p.stat().st_mtime, p.name))
    results = []
    for p in files:
        if time.time() - p.stat().st_mtime < min_age_seconds:
            results.append(IngestResult(p.name, dataset_for(p, folder), "pending", message="still being written"))
            continue
        results.append(ingest_file(engine, p, dataset_for(p, folder)))
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Load data files into the database.")
    ap.add_argument("path", nargs="?", default=str(INCOMING), help="a file, or a folder to scan (default: data/incoming)")
    ap.add_argument("--dataset", help="dataset name (default: the subfolder name, or 'default')")
    ap.add_argument("--min-age", type=float, default=2, help="skip files modified more recently than this many seconds")
    a = ap.parse_args()
    engine, p = get_engine(), Path(a.path)
    results = [ingest_file(engine, p, a.dataset)] if p.is_file() else scan_folder(engine, p, a.min_age)
    for r in results or []:
        print(r.line())
    if not results:
        print(f"No data files found in {p}. Put a .csv or .xlsx file there (one subfolder per dataset).")


if __name__ == "__main__":
    main()