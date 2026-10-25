"""Phase 3: one live refresh of a dataset, from rows in the database to a stored dashboard snapshot.

    result = run_once(engine, "opd", window="7d", llm=call_llm)

What a refresh does
  1. Finds the newest data day (the window is anchored to the data, not the wall clock) and builds a rolling
     window: the last N days against the N days before.
  2. Skips everything if nothing changed since the last snapshot (no new rows, same window, same layout).
  3. Uses the dataset's saved dashboard layout, creating and saving one the first time only.
  4. Runs the same engine as upload mode: aggregate, compare, findings.
  5. Asks the LLM for the plain-English summary only when the findings changed; otherwise reuses the last one.
  6. Compares with the previous snapshot ("since the last refresh") using the same findings code.
  7. Stores the snapshot and logs the run. A failure is logged and returned, never raised, so a scheduler survives.

    python -m src.live opd --window 7d [--llm] [--force]
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from datetime import timedelta

import pandas as pd
from sqlalchemy import delete, func, insert, select, update

from .aggregator import previous_period
from .db import (ROOT, dataset_specs, get_dataset, get_engine, ingested_files, load_dataframe, pipeline_runs, records, snapshots, utcnow)
from .filters import filter_options
from .insights import build_findings, top_findings
from .pipeline import make_dashboard, prepare
from .renderer import render_html
from .spec_generator import generate_spec
from .summariser import make_context, summarise

SPEC_DIR = ROOT / "specs"
SINCE_LAST_KINDS = {"kpi_change", "category_shift", "combo_shift"}


@dataclass
class RunResult:
    dataset: str
    status: str                       # ok | no_change | no_data | failed
    snapshot_id: int = None
    rows_new: int = 0
    summary_source: str = None        # llm | reused | template
    message: str = ""
    run_id: int = None

    def line(self):
        return f"{self.status.upper():9} {self.dataset}: {self.message}" + (f" (snapshot {self.snapshot_id})" if self.snapshot_id else "")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def parse_window(text):
    """'7d' -> 7, '2w' -> 14."""
    m = re.fullmatch(r"\s*(\d+)\s*([dw])\s*", str(text).lower())
    days = int(m.group(1)) * (7 if m and m.group(2) == "w" else 1) if m else 0
    if not 1 <= days <= 366:
        raise ValueError(f"Window must look like 7d, 30d or 2w (1 to 366 days); got '{text}'.")
    return days


def rolling_windows(as_of, days):
    """(current, baseline): the last `days` days up to as_of, and the same number of days before that."""
    end = pd.Timestamp(as_of).normalize()
    current = ((end - timedelta(days=days - 1)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return current, previous_period(*current)


# ------------------------------------------------------------------ stored things
def latest_snapshot(engine, dataset, offset=0):
    """Newest snapshot (offset=1 is the one before it), or None."""
    with engine.connect() as c:
        row = c.execute(select(snapshots).where(snapshots.c.dataset == dataset)
                        .order_by(snapshots.c.id.desc()).limit(1).offset(offset)).mappings().first()
    return dict(row) if row else None


def snapshot_html(snapshot):
    return render_html(snapshot["payload"])


def resolve_spec(engine, dataset, title, llm=None, spec=None):
    """(spec, source). Order: explicit, saved, specs/<dataset>.json, then generated once (LLM if given, else rules)."""
    with engine.connect() as c:
        saved = c.execute(select(dataset_specs).where(dataset_specs.c.dataset == dataset)).mappings().first()
    if spec is None and saved:
        return saved["spec"], saved["source"]
    if spec is not None:
        source = "given"
    elif (SPEC_DIR / f"{dataset}.json").exists():
        spec, source = json.loads((SPEC_DIR / f"{dataset}.json").read_text()), "file"
    else:
        prep = prepare(load_dataframe(engine, dataset))
        spec, info = generate_spec(prep.profile, prep.df, title, llm)
        source = info["source"]
    with engine.begin() as c:
        c.execute(delete(dataset_specs).where(dataset_specs.c.dataset == dataset))
        c.execute(insert(dataset_specs).values(dataset=dataset, spec=spec, source=source, updated_at=utcnow()))
    return spec, source


def _since_last(new, old, rows_new):
    """What changed since the previous snapshot, reusing the findings code: the old numbers become the baseline."""
    ids = [w for w in new["spec"]["widgets"] if w["type"] in ("kpi", "breakdown", "combo") and w["id"] in old["data"]["widgets"]]
    if not ids:
        return None
    diff = {"spec": {"widgets": ids},
            "data": {"dates": {"current": "now", "baseline": "previous refresh"},
                     "rows": {"current": new["data"]["rows"]["current"], "baseline": old["data"]["rows"]["current"]},
                     "widgets": {w["id"]: {"current": new["data"]["widgets"][w["id"]]["current"],
                                           "baseline": old["data"]["widgets"][w["id"]]["current"]} for w in ids}}}
    found = [f for f in build_findings(diff) if f["kind"] in SINCE_LAST_KINDS]
    return {"rows_new": rows_new,                                             # rows that arrived since the last snapshot
            "window_change": new["data"]["rows"]["current"] - old["data"]["rows"]["current"],   # change in records inside the window
            "findings": [{"kind": f["kind"], "severity": f["severity"], "text": f["text"]} for f in top_findings(found, 5)]}


def pipeline_status(engine, dataset, stale_after_minutes=90, now=None):
    """Is the pipeline healthy? 'stale' = no successful (ok or no_change) refresh within the allowed time."""
    now = now or utcnow()
    with engine.connect() as c:
        last = c.execute(select(pipeline_runs).where(pipeline_runs.c.dataset == dataset)
                         .order_by(pipeline_runs.c.id.desc()).limit(1)).mappings().first()
        ok = c.execute(select(pipeline_runs).where(pipeline_runs.c.dataset == dataset, pipeline_runs.c.status.in_(("ok", "no_change")))
                       .order_by(pipeline_runs.c.id.desc()).limit(1)).mappings().first()
    ok_at = ok["finished_at"] if ok else None
    age = (now - ok_at).total_seconds() / 60 if ok_at else None
    stale = age is None or age > stale_after_minutes
    failed = bool(last and last["status"] == "failed")
    if not last:
        message = "No refresh has run yet."
    elif failed:
        message = f"The last refresh failed: {last['message']}"
    elif stale:
        message = f"No successful refresh for {int(age)} minutes."
    else:
        message = "Healthy."
    return {"last": dict(last) if last else None, "last_ok_at": ok_at, "age_minutes": age,
            "stale": stale, "failed": failed, "message": message}


def recent_files(engine, dataset, limit=10):
    with engine.connect() as c:
        rows = c.execute(select(ingested_files).where(ingested_files.c.dataset == dataset)
                         .order_by(ingested_files.c.id.desc()).limit(limit)).mappings().all()
    return [dict(r) for r in rows]


def snapshot_history(engine, dataset, limit=10):
    with engine.connect() as c:
        rows = c.execute(select(snapshots.c.id, snapshots.c.created_at, snapshots.c.window_label, snapshots.c.payload)
                         .where(snapshots.c.dataset == dataset).order_by(snapshots.c.id.desc()).limit(limit)).all()
    return [{"id": r.id, "created_at": r.created_at, "window": r.window_label, "summary": r.payload["insights"]["source"],
             "findings": len(r.payload["insights"]["findings"])} for r in rows]


def view_filters(engine, dataset, snapshot):
    """{column: values} for the filter boxes, from the snapshot's own window and the layout's filter list."""
    meta, spec = snapshot["payload"]["meta"], snapshot["payload"]["spec"]
    df = load_dataframe(engine, dataset, meta["baseline"][0], meta["current"][1])
    return filter_options(df, [c for c in spec.get("filters", []) if c in df.columns])


def view_dashboard(engine, dataset, snapshot, filters=None, llm=None, narrate=False, cache=None, stale_after=None):
    """The dashboard to show for a snapshot. No filters: the stored snapshot itself. With filters: recomputed from the
    database over the snapshot's windows, so every number and sentence describes exactly the selected slice."""
    payload, meta = snapshot["payload"], snapshot["payload"]["meta"]
    title = payload["spec"].get("title") or dataset
    active = {c: v for c, v in (filters or {}).items() if v}
    live = {"snapshot_id": snapshot["id"], "updated_at": snapshot["created_at"].isoformat() + "Z", "stale_minutes": stale_after}
    if active:
        df = load_dataframe(engine, dataset, meta["baseline"][0], meta["current"][1])
        out = make_dashboard(prepare(df), current=tuple(meta["current"]), baseline=tuple(meta["baseline"]), spec=payload["spec"],
                             title=title, filters=active, llm=llm, narrate=narrate, cache=cache)
        p = out["payload"]
        p["since_last"] = None                      # the comparison with the previous refresh is for the full dataset
        live["view"] = "filtered view"
    else:
        p = json.loads(json.dumps(payload))
        if narrate and llm is not None:
            r = summarise(build_findings(p), make_context(p, None, title), llm, cache=cache)
            p["insights"].update(summary=r["summary"], source=r["source"], notes=r["notes"])
    p["live"] = live
    return {"payload": p, "html": render_html(p), "filtered": bool(active)}


# ------------------------------------------------------------------ the refresh
def _refresh(engine, dataset, window, as_of, spec, llm, force, title):
    days = parse_window(window)
    with engine.connect() as c:
        if get_dataset(c, dataset) is None:
            return RunResult(dataset, "failed", message=f"Unknown dataset '{dataset}'. Ingest a file for it first.")
        last_id, last_ts, total = c.execute(select(func.max(records.c.id), func.max(records.c.event_ts), func.count())
                                            .where(records.c.dataset == dataset)).one()
    if last_id is None:
        return RunResult(dataset, "no_data", message="No records yet.")

    title = title or dataset
    spec, spec_source = resolve_spec(engine, dataset, title, llm, spec)
    prev = latest_snapshot(engine, dataset)
    spec = json.loads(json.dumps(spec))                       # exactly what a later run will read back
    current, baseline = rolling_windows(as_of or last_ts, days)
    meta = json.loads(json.dumps({"dataset": dataset, "window": window, "current": current, "baseline": baseline,   # as stored: lists, not tuples
                                  "last_record_id": int(last_id), "spec_hash": _sha(spec), "spec_source": spec_source}))

    prev_meta = (prev or {}).get("payload", {}).get("meta", {})
    new_rows = int(total) if not prev else int(last_id) - int(prev_meta.get("last_record_id", 0))
    if prev and not force and all(prev_meta.get(k) == meta[k] for k in ("last_record_id", "window", "spec_hash", "current")):
        return RunResult(dataset, "no_change", rows_new=0, message="Nothing new since the last snapshot.")

    df = load_dataframe(engine, dataset, baseline[0], current[1])
    if df.empty:
        return RunResult(dataset, "no_data", message=f"No records between {baseline[0]} and {current[1]}.")
    out = make_dashboard(prepare(df), current=current, baseline=baseline, spec=spec, title=title, narrate=False)
    payload = out["payload"]

    findings = build_findings(payload)
    ctx = make_context(payload, None, title)
    fhash = _sha([f["text"] for f in top_findings(findings, 8)] + [ctx["filters"]])
    ins = payload["insights"]
    if llm is None:
        summary_source = "template"
    elif prev and prev["findings_hash"] == fhash and prev["payload"]["insights"]["source"] == "llm":
        ins.update(summary=prev["payload"]["insights"]["summary"], source="llm")           # same findings, same words
        summary_source = "reused"
    else:
        r = summarise(findings, ctx, llm)
        ins.update(summary=r["summary"], source=r["source"], notes=r["notes"])
        summary_source = r["source"]

    payload["meta"] = meta
    payload["since_last"] = _since_last(payload, prev["payload"], new_rows) if prev else None
    payload = json.loads(json.dumps(payload, default=str))
    with engine.begin() as c:
        snap_id = c.execute(insert(snapshots).values(dataset=dataset, created_at=utcnow(), window_label=window,
                                                      payload=payload, findings_hash=fhash)).inserted_primary_key[0]
    return RunResult(dataset, "ok", snap_id, new_rows, summary_source,
                     f"{new_rows} new rows; summary: {summary_source}; window {current[0]} to {current[1]}")


def run_once(engine, dataset, window="7d", as_of=None, spec=None, llm=None, force=False, title=None):
    """One refresh. Always returns a RunResult and always logs a pipeline_runs row; never raises."""
    with engine.begin() as c:
        run_id = c.execute(insert(pipeline_runs).values(dataset=dataset, started_at=utcnow(), status="running")).inserted_primary_key[0]
    try:
        res = _refresh(engine, dataset, window, as_of, spec, llm, force, title)
    except Exception as e:
        res = RunResult(dataset, "failed", message=f"{type(e).__name__}: {e}")
    with engine.begin() as c:
        c.execute(update(pipeline_runs).where(pipeline_runs.c.id == run_id).values(
            finished_at=utcnow(), status=res.status, rows_new=res.rows_new, message=res.message, snapshot_id=res.snapshot_id))
    res.run_id = run_id
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Refresh one dataset's live dashboard snapshot.")
    ap.add_argument("dataset")
    ap.add_argument("--window", default="7d", help="rolling window, e.g. 7d, 30d, 2w")
    ap.add_argument("--as-of", help="anchor date (default: the newest data day)")
    ap.add_argument("--llm", action="store_true", help="use the LLM from .env for the layout (first time) and the summary")
    ap.add_argument("--force", action="store_true", help="build a snapshot even if nothing changed")
    a = ap.parse_args()
    llm = None
    if a.llm:
        from .llm import call_llm
        llm = call_llm
    print(run_once(get_engine(), a.dataset, a.window, a.as_of, llm=llm, force=a.force).line())


if __name__ == "__main__":
    main()