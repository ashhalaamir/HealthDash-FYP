import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aggregator import build_payload, previous_period, validate_spec
from src.cleaning import clean_data, load_file
from src.filters import apply_filters, describe_filters, filter_options, filterable_columns
from src.insights import build_findings
from src.profiler import profile_data


@pytest.fixture
def df():
    return pd.DataFrame({"d": pd.to_datetime(["2026-10-01"] * 6), "gender": ["F", "F", "M", "M", "M", "F"],
                         "age": ["a", "b", "a", "b", "a", "a"], "dx": ["x", "y", "x", "y", "x", "y"]})


def test_single_multi_and_combined_filters(df):
    assert len(apply_filters(df, {"gender": ["F"]})) == 3
    assert len(apply_filters(df, {"gender": ["F", "M"]})) == 6
    assert len(apply_filters(df, {"gender": ["F"], "age": ["a"]})) == 2          # AND across columns


def test_empty_selection_means_no_filter(df):
    assert len(apply_filters(df, {"gender": []})) == 6 and len(apply_filters(df, None)) == 6


def test_unknown_column_raises_and_unknown_value_is_empty(df):
    with pytest.raises(ValueError, match="unknown column"):
        apply_filters(df, {"ward": ["1"]})
    assert apply_filters(df, {"gender": ["Z"]}).empty


def test_describe_filters():
    assert describe_filters({}) == "All records" and describe_filters({"gender": []}) == "All records"
    assert describe_filters({"gender": ["Female"], "age_group": ["40-59", "60+"]}) == "Gender: Female; Age group: 40-59, 60+"


def test_filter_options_use_all_values(df):
    assert filter_options(df, ["gender", "age"]) == {"gender": ["F", "M"], "age": ["a", "b"]}


def test_spec_filters_are_validated(df):
    spec, problems = validate_spec({"widgets": [], "filters": ["gender", "ghost"]}, df)
    assert spec["filters"] == ["gender"] and "ghost" in problems[0]


GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "planted_facts.json").exists(), reason="run scripts/generate_fake_data.py first")


@needs_data
def test_filterable_columns_from_spec_and_profile():
    df, rep = clean_data(load_file(GEN / "opd_visits.csv"))
    profile = profile_data(df, rep["date_column"])
    spec = json.loads((ROOT / "specs/opd_example.json").read_text())
    assert filterable_columns(profile, spec) == ["gender", "age_group", "chief_complaint"]
    auto = filterable_columns(profile)
    assert "patient_id" not in auto and "visit_id" not in auto and 1 <= len(auto) <= 4


@needs_data
def test_filters_flow_through_to_numbers_and_findings():
    df, rep = clean_data(load_file(GEN / "opd_visits.csv"))
    spec = json.loads((ROOT / "specs/opd_example.json").read_text())
    cur = ("2026-10-01", "2026-10-31")
    run = lambda sel: build_payload(apply_filters(df, sel), spec, rep["date_column"], cur, previous_period(*cur))
    whole, women = run({}), run({"gender": ["Female"]})
    n = lambda p: p["data"]["widgets"]["visits"]["current"]["value"]
    assert 0 < n(women) < n(whole)
    assert women["data"]["widgets"]["gender"]["current"]["labels"] == ["Female"]
    narrow = run({"gender": ["Female"], "age_group": ["60+"], "chief_complaint": ["Wrist pain"]})
    assert build_findings(narrow)[0]["kind"] in ("small_sample", "no_data")