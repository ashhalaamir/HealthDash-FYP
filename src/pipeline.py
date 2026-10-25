"""Entry points for building a dashboard. The Streamlit app and the Airflow DAG both call these,
so the logic lives in one place.

    prep = prepare("data/generated/opd_visits.csv")          # load + clean + profile: do once per upload
    out  = make_dashboard(prep, filters={"gender": ["Female"]}, narrate=True, llm=call_llm)
    out["html"], out["insights"], out["payload"]              # everything the UI needs

    run(path, out_path, ...)                                  # one-shot version of the two calls above

narrate=False gives instant findings plus a template summary (cheap: use it while filters change).
narrate=True also asks the LLM for the written summary (use it for the Summary button and scheduled runs).

CLI:
  python -m src.pipeline data/generated/opd_visits.csv --spec specs/opd_example.json \
      --current 2026-10-01 2026-10-31 --baseline 2026-09-01 2026-09-30 \
      --filter gender=Female --filter age_group=40-59,60+ --llm --narrate --out data/output/opd.html
"""
import argparse
from datetime import timedelta
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .aggregator import build_payload, previous_period
from .cleaning import clean_data, load_file
from .filters import apply_filters
from .insights import build_findings
from .llm import call_llm
from .profiler import profile_data
from .renderer import render_dashboard, render_html
from .spec_generator import generate_spec
from .summariser import make_context, summarise


@dataclass
class Prepared:
    df: pd.DataFrame
    report: dict
    date_col: str
    profile: dict


def prepare(source, required_columns=None):
    """Load, clean and profile a file (or DataFrame). Raises DataValidationError for unusable files."""
    raw = load_file(source) if isinstance(source, (str, Path)) else source
    df, report = clean_data(raw, required_columns=required_columns)
    return Prepared(df, report, report["date_column"], profile_data(df, report["date_column"]))


def default_periods(prep):
    """Last 30 days of data against the 30 days before."""
    last = prep.df[prep.date_col].max()
    current = ((last - timedelta(days=29)).strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
    return current, previous_period(*current)


def make_dashboard(prep, out_path=None, current=None, baseline=None, spec=None, title=None,
                   llm=None, focus=None, filters=None, narrate=False, cache=None):
    """Build one dashboard view. Returns a dict with html, payload, insights, spec, periods and warnings.
    spec: use as-is; None means the LLM proposes one (if llm given) or the rule-based spec is used."""
    if current is None:
        current, default_base = default_periods(prep)
        baseline = baseline or default_base
    baseline = baseline or previous_period(*current)
    name = title or "Dashboard"

    spec_source, notes = "given", []
    if spec is None:
        spec, info = generate_spec(prep.profile, prep.df, name, llm, focus)
        spec_source, notes = info["source"], info["warnings"]

    view = apply_filters(prep.df, filters)
    payload = build_payload(view, spec, prep.date_col, current, baseline)
    findings = build_findings(payload)
    ctx = make_context(payload, filters, name)
    result = summarise(findings, ctx, llm if narrate else None, cache=cache)
    payload["insights"] = {"summary": result["summary"], "source": result["source"], "scope": ctx["filters"],
                           "findings": result["findings"], "notes": result["notes"]}

    path = render_dashboard(payload, out_path) if out_path else None
    return {"path": path, "html": render_html(payload), "payload": payload, "insights": payload["insights"],
            "spec": payload["spec"], "spec_source": spec_source, "current": current, "baseline": baseline,
            "clean_report": prep.report, "warnings": notes + payload["warnings"]}


def run(source, out_path, current=None, baseline=None, spec=None, title=None, required_columns=None,
        llm=None, focus=None, filters=None, narrate=False, cache=None):
    """One-shot: file in, dashboard file out."""
    prep = prepare(source, required_columns)
    return make_dashboard(prep, out_path, current, baseline, spec, title or Path(str(source)).stem,
                          llm, focus, filters, narrate, cache)


def parse_filters(pairs):
    """['gender=Female', 'age_group=40-59,60+'] -> {'gender': ['Female'], 'age_group': ['40-59', '60+']}"""
    out = {}
    for p in pairs or []:
        col, _, vals = p.partition("=")
        if not vals:
            raise SystemExit(f"Bad --filter '{p}'. Use COLUMN=VALUE or COLUMN=VALUE1,VALUE2.")
        out[col.strip()] = [v.strip() for v in vals.split(",")]
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate a dashboard HTML file from a data file.")
    ap.add_argument("source")
    ap.add_argument("--out", default="data/output/dashboard.html")
    ap.add_argument("--spec", help="spec JSON file; omit to use the rule-based fallback (or --llm)")
    ap.add_argument("--llm", action="store_true", help="let the LLM in .env propose the spec")
    ap.add_argument("--narrate", action="store_true", help="also ask the LLM for the written summary")
    ap.add_argument("--focus", help='optional hint for the LLM, e.g. "pharmacy stock and expiry"')
    ap.add_argument("--filter", action="append", metavar="COL=VAL[,VAL]", help="repeatable")
    ap.add_argument("--current", nargs=2, metavar=("START", "END"))
    ap.add_argument("--baseline", nargs=2, metavar=("START", "END"))
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text()) if a.spec else None
    llm = call_llm if (a.llm or a.narrate) else None
    r = run(a.source, a.out, a.current, a.baseline, spec, llm=llm, focus=a.focus,
            filters=parse_filters(a.filter), narrate=a.narrate)
    print(f"Wrote {r['path']}  spec={r['spec_source']}  summary={r['insights']['source']}  current={r['current']}  baseline={r['baseline']}")
    print("Summary:", r["insights"]["summary"])
    for w in r["warnings"] + r["insights"]["notes"]:
        print("Note:", w)


if __name__ == "__main__":
    main()