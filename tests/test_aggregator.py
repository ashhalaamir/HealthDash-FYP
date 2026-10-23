import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aggregator import build_payload, cumulative_windows, previous_period, validate_spec, window
from src.cleaning import clean_data, load_file


@pytest.fixture
def df():
    return pd.DataFrame({
        "d": pd.to_datetime(["2026-10-01", "2026-10-01", "2026-10-02", "2026-10-04", "2026-10-05"]),
        "who": ["a", "b", "a", "c", "a"],
        "dx": ["flu", "flu", "cold", "flu", "cold"],
        "vas": [1.0, 3.0, 2.0, 4.0, 6.0],
        "flag": ["Yes", "No", "Yes", "No", "No"],
    })


SPEC = {"title": "T", "widgets": [
    {"id": "n", "type": "kpi", "label": "n", "agg": "count"},
    {"id": "u", "type": "kpi", "label": "u", "agg": "nunique", "column": "who"},
    {"id": "m", "type": "kpi", "label": "m", "agg": "mean", "column": "vas"},
    {"id": "f", "type": "kpi", "label": "f", "agg": "count", "where": {"flag": "Yes"}},
    {"id": "t", "type": "trend", "label": "t", "agg": "count"},
    {"id": "b", "type": "breakdown", "label": "b", "by": "dx"},
    {"id": "c", "type": "combo", "label": "c", "by": "dx", "bar": {"agg": "count"}, "line": {"agg": "mean", "column": "vas"}},
]}


def test_window_is_inclusive(df):
    assert len(window(df, "d", "2026-10-01", "2026-10-02")) == 3


def test_kpis_for_both_periods(df):
    p = build_payload(df, SPEC, "d", ("2026-10-03", "2026-10-05"), ("2026-10-01", "2026-10-02"))
    w = p["data"]["widgets"]
    assert w["n"] == {"current": {"value": 2}, "baseline": {"value": 3}}
    assert w["u"]["baseline"]["value"] == 2
    assert w["m"]["baseline"]["value"] == 2 and w["m"]["current"]["value"] == 5
    assert w["f"]["baseline"]["value"] == 2 and w["f"]["current"]["value"] == 0


def test_trend_fills_empty_days(df):
    p = build_payload(df, SPEC, "d", ("2026-10-01", "2026-10-05"))
    t = p["data"]["widgets"]["t"]["current"]
    assert t["values"] == [2, 1, 0, 1, 1] and t["labels"][0] == "Oct 01"


def test_breakdown_aligns_labels_across_periods(df):
    p = build_payload(df, SPEC, "d", ("2026-10-04", "2026-10-05"), ("2026-10-01", "2026-10-02"))
    b = p["data"]["widgets"]["b"]
    assert b["current"]["labels"] == b["baseline"]["labels"]
    assert dict(zip(b["baseline"]["labels"], b["baseline"]["values"])) == {"flu": 2, "cold": 1}
    assert dict(zip(b["current"]["labels"], b["current"]["values"])) == {"flu": 1, "cold": 1}


def test_combo_has_bar_and_line(df):
    c = build_payload(df, SPEC, "d", ("2026-10-01", "2026-10-05"))["data"]["widgets"]["c"]["current"]
    i = c["labels"].index("flu")
    assert c["bar"][i] == 3 and c["line"][i] == round((1 + 3 + 4) / 3, 2)


def test_no_baseline_omits_baseline_keys(df):
    w = build_payload(df, SPEC, "d", ("2026-10-01", "2026-10-05"))["data"]["widgets"]
    assert all("baseline" not in v for v in w.values())


def test_empty_window_does_not_crash(df):
    p = build_payload(df, SPEC, "d", ("2026-10-01", "2026-10-05"), ("2025-01-01", "2025-01-31"))
    assert p["data"]["widgets"]["n"]["baseline"]["value"] == 0
    assert p["data"]["widgets"]["b"]["baseline"]["values"] == [0, 0]


def test_validation_drops_bad_widgets_and_reports(df):
    bad = {"widgets": [
        {"id": "ok", "type": "kpi", "label": "x", "agg": "count"},
        {"id": "ghost", "type": "breakdown", "label": "x", "by": "nope"},
        {"id": "txt", "type": "kpi", "label": "x", "agg": "mean", "column": "dx"},
        {"id": "ok", "type": "kpi", "label": "dup", "agg": "count"},
        {"id": "pie", "type": "pie", "label": "x"},
        {"id": "nocol", "type": "kpi", "label": "x", "agg": "sum"},
    ]}
    clean, problems = validate_spec(bad, df)
    assert [w["id"] for w in clean["widgets"]] == ["ok"] and len(problems) == 5


def test_window_helpers(df):
    assert previous_period("2026-10-11", "2026-10-20") == ("2026-10-01", "2026-10-10")
    cur, base = cumulative_windows(df, "d", "2026-10-02", "2026-10-05")
    assert cur == ("2026-10-01", "2026-10-05") and base == ("2026-10-01", "2026-10-02")


def test_payload_is_json_serialisable(df):
    json.dumps(build_payload(df, SPEC, "d", ("2026-10-01", "2026-10-05"), ("2026-10-01", "2026-10-02")))


# ----- end to end on the generated data: the numbers must match the planted facts
GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "planted_facts.json").exists(),
                                reason="run scripts/generate_fake_data.py first")


def _value(payload, wid, label, side="current", key="values"):
    w = payload["data"]["widgets"][wid][side]
    return w[key][w["labels"].index(label)]


@needs_data
def test_opd_matches_planted_facts():
    facts = json.loads((GEN / "planted_facts.json").read_text())["opd"]
    df, rep = clean_data(load_file(GEN / "opd_visits.csv"))
    spec = json.loads((ROOT / "specs/opd_example.json").read_text())
    p = build_payload(df, spec, rep["date_column"], ("2026-10-01", "2026-10-31"), ("2026-09-01", "2026-09-30"))
    assert p["warnings"] == []
    assert _value(p, "diag", "Lower back pain") == facts["lower_back_pain_by_month"]["2026-10"]
    assert _value(p, "diag", "Lower back pain", "baseline") == facts["lower_back_pain_by_month"]["2026-09"]
    assert p["data"]["widgets"]["vas"]["current"]["value"] == pytest.approx(facts["avg_vas_by_month"]["2026-10"], abs=0.01)
    peak = max(p["data"]["widgets"]["trend"]["current"]["values"])
    assert peak == facts["busiest_day_visits"]


@needs_data
def test_icu_matches_planted_facts_with_same_code():
    facts = json.loads((GEN / "planted_facts.json").read_text())["icu"]
    df, rep = clean_data(load_file(GEN / "icu_admissions.csv"))
    spec = json.loads((ROOT / "specs/icu_example.json").read_text())
    p = build_payload(df, spec, rep["date_column"], ("2026-10-01", "2026-10-31"), ("2026-09-01", "2026-09-30"))
    assert p["warnings"] == []
    assert _value(p, "diag", "Sepsis") == facts["sepsis_by_month"]["2026-10"]
    assert p["data"]["widgets"]["los"]["current"]["value"] == pytest.approx(facts["avg_los_by_month"]["2026-10"], abs=0.01)