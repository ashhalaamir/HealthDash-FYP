import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aggregator import build_payload
from src.compare import compare_rows
from tests.test_aggregator import SPEC, df  # noqa: F401  (df is a fixture)


def by_label(rows):
    return {r["label"]: r for r in rows}


def test_real_mix_shift_is_notable():
    r = by_label(compare_rows({"A": 30, "B": 70}, {"A": 10, "B": 90}))
    assert r["A"]["shift_pp"] == 20.0 and r["A"]["significant"] and r["A"]["delta"] == 20 and r["A"]["pct"] == 200.0


def test_noise_is_not_notable():
    assert not any(r["significant"] for r in compare_rows({"A": 52, "B": 48}, {"A": 50, "B": 50}))


def test_tiny_counts_never_notable():
    assert not any(r["significant"] for r in compare_rows({"A": 3, "B": 0}, {"A": 0, "B": 3}))


def test_same_mix_at_double_volume_shows_no_shift():
    a = by_label(compare_rows({"A": 60, "B": 140}, {"A": 30, "B": 70}))["A"]
    assert a["shift_pp"] == 0.0 and a["pct"] == 100.0 and not a["significant"]


def test_new_category_and_non_count_values():
    r = by_label(compare_rows({"A": 5, "NEW": 5}, {"A": 10}))
    assert r["NEW"]["baseline"] == 0 and r["NEW"]["pct"] is None
    v = compare_rows({"A": 1.5}, {"A": 2.0}, counts=False)[0]
    assert v["delta"] == -0.5 and v["shift_pp"] is None and v["z"] is None


def test_more_categories_means_a_stricter_bar():
    few = compare_rows({"A": 135, "R": 865}, {"A": 100, "R": 900})
    many = compare_rows({"A": 135, "R": 865, **{f"Z{i}": 0 for i in range(38)}}, {"A": 100, "R": 900})
    assert by_label(few)["A"]["significant"] and not by_label(many)["A"]["significant"]


def test_payload_carries_compare_rows(df):  # noqa: F811
    p = build_payload(df, SPEC, "d", ("2026-10-04", "2026-10-05"), ("2026-10-01", "2026-10-02"))
    w = p["data"]["widgets"]
    assert [r["label"] for r in w["b"]["compare"]] == w["b"]["current"]["labels"]
    assert w["c"]["compare"][0]["current"] == w["c"]["current"]["bar"][0]
    json.dumps(p)
    assert "compare" not in build_payload(df, SPEC, "d", ("2026-10-01", "2026-10-05"))["data"]["widgets"]["b"]