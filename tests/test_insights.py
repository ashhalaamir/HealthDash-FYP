import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aggregator import build_payload, previous_period
from src.cleaning import clean_data, load_file
from src.insights import SEVERITY, build_findings, template_summary, top_findings


def payload(**over):
    p = {
        "spec": {"widgets": [
            {"id": "n", "type": "kpi", "label": "Total visits", "agg": "count"},
            {"id": "vas", "type": "kpi", "label": "Average VAS", "agg": "mean", "column": "v", "higher_is_better": False},
            {"id": "wait", "type": "kpi", "label": "Average wait", "agg": "mean", "column": "w", "higher_is_better": False},
            {"id": "t", "type": "trend", "label": "Daily visits", "agg": "count"},
            {"id": "d", "type": "breakdown", "label": "Visits by diagnosis", "by": "dx"},
        ]},
        "data": {"dates": {"current": "c", "baseline": "b"}, "widgets": {
            "n": {"current": {"value": 150}, "baseline": {"value": 100}},
            "vas": {"current": {"value": 1.5}, "baseline": {"value": 2.0}},
            "wait": {"current": {"value": 30}, "baseline": {"value": 20}},
            "t": {"current": {"labels": [f"Oct {i:02d}" for i in range(1, 11)], "values": [3, 4, 3, 5, 4, 3, 30, 4, 3, 4]}},
            "d": {"current": {"labels": ["A", "B", "Not recorded"], "values": [30, 19, 20]},
                  "baseline": {"labels": ["A", "B", "Not recorded"], "values": [10, 20, 5]}},
        }},
    }
    for k, v in over.items():
        p["data"]["widgets"][k] = v
    return p


def by_kind(findings, kind):
    return [f for f in findings if f["kind"] == kind]


def test_kpi_change_text_and_severity():
    f = {x["subject"]: x for x in by_kind(build_findings(payload()), "kpi_change")}
    assert f["n"]["text"] == "Total visits rose 50% (100 to 150)." and f["n"]["severity"] == "high"
    assert f["vas"]["text"] == "Average VAS fell 25% (2 to 1.5)." and f["vas"]["severity"] == "medium"   # favourable: no bump
    assert f["wait"]["severity"] == "high"                                                                   # +50%: already high


def test_adverse_direction_is_bumped():
    p = payload(wait={"current": {"value": 26}, "baseline": {"value": 20}})        # +30% = medium, adverse -> high
    f = {x["subject"]: x for x in by_kind(build_findings(p), "kpi_change")}
    assert f["wait"]["severity"] == "high"


def test_tiny_absolute_count_moves_stay_low():
    p = payload(wait={"current": {"value": 13}, "baseline": {"value": 11}})
    p["spec"]["widgets"][2] = {"id": "wait", "type": "kpi", "label": "Extra visits", "agg": "count", "where": {"x": 1}, "higher_is_better": False}
    f = {x["subject"]: x for x in by_kind(build_findings(p), "kpi_change")}
    assert f["wait"]["severity"] == "low"                      # +18% but only 2 more records


def test_ordinary_count_noise_is_not_a_spike():
    p = payload(t={"current": {"labels": [f"Oct {i:02d}" for i in range(1, 11)], "values": [4, 3, 5, 4, 2, 4, 9, 3, 4, 5]}})
    assert by_kind(build_findings(p), "spike") == []


def test_small_changes_are_ignored():
    p = payload(n={"current": {"value": 102}, "baseline": {"value": 100}})
    assert not any(f["subject"] == "n" for f in build_findings(p))


def test_category_shift_skips_blank_label_and_picks_biggest_move():
    shifts = by_kind(build_findings(payload()), "category_shift")
    assert len(shifts) == 1 and "'A' rose from 10 to 30 (+200%)" in shifts[0]["text"] and shifts[0]["severity"] == "high"


def test_data_quality_flags_blank_share():
    dq = by_kind(build_findings(payload()), "data_quality")
    assert len(dq) == 1 and "29% of records" in dq[0]["text"] and dq[0]["severity"] == "high"


def test_spike_detected_and_located():
    sp = by_kind(build_findings(payload()), "spike")
    assert len(sp) == 1 and "Oct 07" in sp[0]["text"] and "30" in sp[0]["text"] and sp[0]["severity"] == "high"


def test_weekly_trend_wording():
    labels = ["Sep 01", "Sep 08", "Sep 15", "Sep 22", "Sep 29", "Oct 06"]
    p = payload(t={"current": {"labels": labels, "values": [20, 22, 21, 60, 23, 20]}})
    assert "the week of Sep 22" in by_kind(build_findings(p), "spike")[0]["text"]


def test_no_baseline_means_no_comparisons_but_spike_remains():
    p = payload()
    for w in p["data"]["widgets"].values():
        w.pop("baseline", None)
    kinds = {f["kind"] for f in build_findings(p)}
    assert "kpi_change" not in kinds and "category_shift" not in kinds and "spike" in kinds


def test_small_sample_warning_and_empty_view():
    small = payload(n={"current": {"value": 12}, "baseline": {"value": 100}})
    f = build_findings(small)
    assert f[0]["kind"] == "small_sample" and "Only 12 records" in f[0]["text"]
    empty = payload(n={"current": {"value": 0}, "baseline": {"value": 100}})
    assert [x["kind"] for x in build_findings(empty)] == ["no_data"]


def test_tiny_baseline_warns_and_suppresses_comparisons():
    p = payload(n={"current": {"value": 372}, "baseline": {"value": 1}})
    f = build_findings(p)
    assert f[0]["kind"] == "baseline_small" and "only 1 records" in f[0]["text"] and f[0]["severity"] == "high"
    kinds = {x["kind"] for x in f}
    assert not kinds & {"kpi_change", "category_shift", "combo_shift"}
    assert {"spike", "data_quality"} <= kinds            # findings that need no baseline survive


def test_empty_baseline_has_its_own_wording_and_both_small_gives_one_warning():
    f = build_findings(payload(n={"current": {"value": 372}, "baseline": {"value": 0}}))
    assert "has no records" in f[0]["text"]
    both = build_findings(payload(n={"current": {"value": 12}, "baseline": {"value": 5}}))
    assert [x["kind"] for x in both if x["kind"] in ("small_sample", "baseline_small")] == ["small_sample"]


def test_combo_shift():
    p = payload()
    p["spec"]["widgets"].append({"id": "c", "type": "combo", "label": "x", "line_label": "Avg VAS"})
    p["data"]["widgets"]["c"] = {"current": {"labels": ["Knee", "Hip"], "bar": [20, 3], "line": [1.0, 9.0]},
                                 "baseline": {"labels": ["Knee", "Hip"], "bar": [18, 3], "line": [1.5, 1.0]}}
    c = by_kind(build_findings(p), "combo_shift")
    assert len(c) == 1 and "Avg VAS for 'Knee' fell 33% (1.5 to 1)" in c[0]["text"]          # 'Hip' ignored: only 3 records


def test_findings_sorted_and_capped():
    f = build_findings(payload())
    ranks = [SEVERITY[x["severity"]] for x in f]
    assert ranks == sorted(ranks, reverse=True)
    assert len(top_findings(f, k=3)) == 3


def test_template_summary_uses_finding_text():
    f = build_findings(payload())
    s = template_summary(f, k=2)
    assert all(x["text"] in s for x in top_findings(f, 2))
    assert template_summary([]) == "No notable changes between the two periods."


# ---- end to end: the findings must rediscover what was planted in the fake data
GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "planted_facts.json").exists(), reason="run scripts/generate_fake_data.py first")


def run_findings(csv, spec_file, cur, base=None):
    df, rep = clean_data(load_file(GEN / csv))
    spec = json.loads((ROOT / "specs" / spec_file).read_text())
    return build_findings(build_payload(df, spec, rep["date_column"], cur, base or previous_period(*cur)))


@needs_data
def test_opd_planted_changes_are_found():
    f = run_findings("opd_visits.csv", "opd_example.json", ("2026-10-01", "2026-10-31"))
    texts = [(x["kind"], x["text"]) for x in f]
    assert any(k == "spike" and "Oct 14" in t for k, t in texts)
    assert any(k == "category_shift" and "'Lower back pain' rose" in t for k, t in texts)
    assert any(k == "kpi_change" and t.startswith("Average follow-up VAS fell") for k, t in texts)


@needs_data
def test_icu_planted_changes_are_found_with_same_code():
    f = run_findings("icu_admissions.csv", "icu_example.json", ("2026-10-01", "2026-10-31"))
    texts = [(x["kind"], x["text"], x["severity"]) for x in f]
    assert any(k == "kpi_change" and t.startswith("Ventilated patients rose") for k, t, _ in texts)
    assert any(k == "kpi_change" and t.startswith("Average stay (days) rose") and s in ("medium", "high") for k, t, s in texts)
    surge = run_findings("icu_admissions.csv", "icu_example.json", ("2026-10-10", "2026-10-20"))
    assert any(x["kind"] == "category_shift" and "'Sepsis' rose" in x["text"] for x in surge)