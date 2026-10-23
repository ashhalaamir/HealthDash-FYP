import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cleaning import clean_data, load_file
from src.profiler import profile_data


def test_roles_on_small_frame():
    df = pd.DataFrame({
        "visit_date": pd.date_range("2026-10-01", periods=40),
        "patient_id": [f"P{i}" for i in range(40)],
        "gender": ["M", "F"] * 20,
        "ward": [1, 2, 3, 4] * 10,                       # small int -> category
        "cost": [float(i) * 1.5 for i in range(40)],     # numeric
        "note": [f"free text {i}" for i in range(40)],   # high-cardinality text
    })
    p = profile_data(df, "visit_date")
    roles = {c["name"]: c["role"] for c in p["columns"]}
    assert roles == {"visit_date": "date", "patient_id": "id", "gender": "category",
                     "ward": "category", "cost": "numeric", "note": "text"}
    assert next(c for c in p["columns"] if c["name"] == "gender")["is_binary"] is True
    assert p["date_range"] == {"start": "2026-10-01", "end": "2026-11-09"}


def test_unknown_date_column():
    with pytest.raises(ValueError):
        profile_data(pd.DataFrame({"a": [1]}), "nope")


def _profile(csv):
    df, rep = clean_data(load_file(ROOT / "data/generated" / csv))
    return profile_data(df, rep["date_column"])


needs_data = pytest.mark.skipif(not (ROOT / "data/generated/opd_visits.csv").exists(),
                                reason="run scripts/generate_fake_data.py first")


@needs_data
def test_opd_profile():
    p = _profile("opd_visits.csv")
    r = p["by_role"]
    assert p["date_column"] == "visit_date"
    assert {"gender", "age_group", "diagnosis", "treatment", "investigation"} <= set(r["category"])
    assert {"age", "follow_up_vas"} <= set(r["numeric"])
    assert {"visit_id", "patient_id"} <= set(r["id"])


@needs_data
def test_icu_profile_same_code_different_data():
    p = _profile("icu_admissions.csv")
    r = p["by_role"]
    assert p["date_column"] == "admit_date"
    assert {"diagnosis", "ventilated", "outcome"} <= set(r["category"])
    assert {"length_of_stay_days", "apache_score"} <= set(r["numeric"])