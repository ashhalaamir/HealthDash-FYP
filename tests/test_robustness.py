"""Regression tests for the review findings: truncated totals, vanished categories, malformed LLM specs,
and empty-view detection that must not depend on which KPIs the spec happens to contain."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import aggregator
from src.aggregator import build_payload, validate_spec
from src.insights import build_findings
from src.profiler import profile_data
from src.spec_generator import generate_spec
from src.summariser import make_context

CUR, BASE = ("2026-10-10", "2026-10-10"), ("2026-10-01", "2026-10-01")


def frame(cur, base=None):
    """One row per count: cur/base are {category: n} for 2026-10-10 / 2026-10-01."""
    rows = [("2026-10-10", k) for k, n in cur.items() for _ in range(n)]
    rows += [("2026-10-01", k) for k, n in (base or {}).items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["d", "dx"]).assign(d=lambda x: pd.to_datetime(x["d"]), v=1.5)


def spec(top=None, **extra):
    by = {"id": "dx", "type": "breakdown", "label": "Visits by diagnosis", "by": "dx", **extra}
    if top:
        by["top"] = top
    return {"widgets": [{"id": "n", "type": "kpi", "label": "Total", "agg": "count"}, by]}


# ---- 1. shares and insight percentages use the whole period, not just the bars on screen
def test_shares_use_full_period_totals_when_top_n_truncates():
    df = frame({"A": 40, "B": 30, "C": 20, "D": 10}, {"A": 40, "B": 30, "C": 20, "D": 10})
    w = build_payload(df, spec(top=2), "d", CUR, BASE)["data"]["widgets"]["dx"]
    assert w["current"]["labels"] == ["A", "B"]
    assert w["totals"] == {"current": 100, "baseline": 100}
    assert [r["share_cur"] for r in w["compare"]] == [40.0, 30.0]            # not 57.1 and 42.9


def test_top_share_finding_uses_the_true_total():
    df = frame({"A": 45, "B": 40, "C": 10, "D": 5})                            # A is 45% of all rows (53% of A+B)
    f = build_findings(build_payload(df, spec(top=2), "d", CUR))
    assert not [x for x in f if x["kind"] == "top_share"]


def test_blank_share_uses_the_true_total():
    df = frame({"A": 50, "Not recorded": 8, "C": 42})                           # 8% of all rows, but 14% of A + blank
    f = build_findings(build_payload(df, spec(order=["A", "Not recorded"]), "d", CUR))
    assert not [x for x in f if x["kind"] == "data_quality"]


# ---- 2. a category that disappears must still be compared
def test_vanished_category_is_shown_and_reported():
    df = frame({"A": 20, "B": 20}, {"A": 20, "B": 20, "C": 30, "D": 10})
    p = build_payload(df, spec(top=3), "d", CUR, BASE)
    w = p["data"]["widgets"]["dx"]
    assert "C" in w["current"]["labels"]
    row = next(r for r in w["compare"] if r["label"] == "C")
    assert (row["current"], row["baseline"]) == (0, 30)
    assert any(f["kind"] == "category_shift" and "'C' fell from 30 to 0" in f["text"] for f in build_findings(p))


# ---- 3. malformed LLM specs never crash
BAD_SPECS = [
    {"widgets": ["oops", {"id": "a"}]},
    {"widgets": [{"id": "a", "type": "breakdown", "label": "x", "by": ["dx"]}]},
    {"widgets": [{"id": "a", "type": "combo", "label": "x", "by": "dx", "bar": "count"}]},
    {"widgets": [{"id": "a", "type": "breakdown", "label": "x", "by": "dx", "top": "many"}]},
    {"widgets": [{"id": "a", "type": "kpi", "label": "x", "agg": "count", "where": ["dx"]}]},
    {"widgets": [{"id": "a", "type": "breakdown", "label": "x", "by": "dx", "order": "A,B"}]},
    {"widgets": [{"id": ["a"], "type": "kpi", "label": "x", "agg": ["sum"]}]},
    {"widgets": [None, 5, [], {}]},
    {"widgets": "none"},
    ["not", "an", "object"],
]


@pytest.mark.parametrize("bad", BAD_SPECS)
def test_validate_spec_never_raises_on_malformed_input(bad):
    clean, problems = validate_spec(bad, frame({"A": 5}))
    assert clean["widgets"] == [] and problems


@pytest.mark.parametrize("bad", BAD_SPECS)
def test_generate_spec_falls_back_instead_of_crashing(bad):
    df = frame({"A": 30, "B": 30})
    s, info = generate_spec(profile_data(df, "d"), df, llm=lambda _: json.dumps(bad))
    assert info["source"] == "fallback" and s["widgets"]


def test_a_widget_that_fails_while_computing_is_dropped_not_fatal(monkeypatch):
    def boom(*a, **k): raise RuntimeError("bad data")
    monkeypatch.setattr(aggregator, "_kpi", boom)
    p = build_payload(frame({"A": 30}), spec(), "d", CUR)
    assert [w["id"] for w in p["spec"]["widgets"]] == ["dx"] and "dx" in p["data"]["widgets"]
    assert any("'n' dropped" in w and "bad data" in w for w in p["warnings"])


# ---- 4. an empty or small view is recognised whatever the spec contains
NO_COUNT_KPI = {"widgets": [{"id": "m", "type": "kpi", "label": "Average", "agg": "mean", "column": "v"},
                            {"id": "dx", "type": "breakdown", "label": "By dx", "by": "dx"}]}


def test_empty_view_detected_without_a_count_kpi():
    df = frame({"A": 30}, {"A": 30})
    empty = df[df.dx == "Z"]
    p = build_payload(empty, NO_COUNT_KPI, "d", CUR, BASE)
    assert p["data"]["rows"] == {"current": 0, "baseline": 0}
    assert [f["kind"] for f in build_findings(p)] == ["no_data"]


def test_small_sample_and_tiny_baseline_detected_without_a_count_kpi():
    p = build_payload(frame({"A": 12}, {"A": 40}), NO_COUNT_KPI, "d", CUR, BASE)
    assert build_findings(p)[0]["kind"] == "small_sample"
    p = build_payload(frame({"A": 60}, {"A": 3}), NO_COUNT_KPI, "d", CUR, BASE)
    assert build_findings(p)[0]["kind"] == "baseline_small"


def test_context_rows_come_from_the_payload():
    p = build_payload(frame({"A": 60}, {"A": 3}), NO_COUNT_KPI, "d", CUR, BASE)
    assert make_context(p)["rows"] == 60