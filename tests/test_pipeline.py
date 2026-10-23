import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cleaning import DataValidationError
from src.pipeline import run
from src.renderer import PAYLOAD_RE, render_dashboard

GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "planted_facts.json").exists(),
                                reason="run scripts/generate_fake_data.py first")


def payload_of(html_path):
    return json.loads(PAYLOAD_RE.search(Path(html_path).read_text(encoding="utf-8")).group(2))


def test_renderer_escapes_script_close(tmp_path):
    p = {"spec": {"title": "A & B", "widgets": []}, "data": {"widgets": {"x": {"current": {"labels": ["</script><b>"]}}}}}
    out = render_dashboard(p, tmp_path / "d.html")
    html = out.read_text(encoding="utf-8")
    assert html.count("</script>") == 3          # Chart.js, payload and app script only; nothing injected
    assert payload_of(out)["data"]["widgets"]["x"]["current"]["labels"] == ["</script><b>"]
    assert "<title>A &amp; B</title>" in html


@needs_data
def test_opd_with_spec(tmp_path):
    spec = json.loads((ROOT / "specs/opd_example.json").read_text())
    r = run(GEN / "opd_visits.csv", tmp_path / "opd.html", ("2026-10-01", "2026-10-31"), ("2026-09-01", "2026-09-30"), spec)
    p = payload_of(r["path"])
    assert r["warnings"] == [] and p["spec"]["title"] == "CLARITY"
    assert p["data"]["widgets"]["visits"]["current"]["value"] > 0


@needs_data
def test_any_dataset_without_a_spec(tmp_path):
    for name in ("opd_visits.csv", "icu_admissions.csv"):
        r = run(GEN / name, tmp_path / f"{name}.html")
        p = payload_of(r["path"])
        assert r["warnings"] == [] and len(p["spec"]["widgets"]) >= 6
        assert set(p["data"]["widgets"]) == {w["id"] for w in p["spec"]["widgets"]}


@needs_data
def test_default_periods_are_last_30_days_and_the_30_before(tmp_path):
    r = run(GEN / "icu_admissions.csv", tmp_path / "x.html")
    assert r["current"] == ("2026-10-02", "2026-10-31") and r["baseline"] == ("2026-09-02", "2026-10-01")


@needs_data
def test_dirty_file_is_cleaned_then_rendered(tmp_path):
    r = run(GEN / "opd_visits_dirty.csv", tmp_path / "d.html")
    assert r["clean_report"]["duplicates_removed"] >= 15 and r["path"].exists()


def test_bad_file_gives_clear_error(tmp_path):
    import pandas as pd
    with pytest.raises(DataValidationError, match="empty"):
        run(pd.DataFrame(), tmp_path / "x.html")


# ---- insights + filters wired through the pipeline
from src.pipeline import make_dashboard, parse_filters, prepare

OPD_SPEC = lambda: json.loads((ROOT / "specs/opd_example.json").read_text())
OCT, SEP = ("2026-10-01", "2026-10-31"), ("2026-09-01", "2026-09-30")


def echo_first_finding(prompt):
    """A fake LLM that answers with the first finding it was given: always passes the number check."""
    first = prompt.split("FINDINGS\n")[1].split("\n")[0].split("] ", 1)[1]
    return json.dumps({"summary": first})


@needs_data
def test_insights_are_in_payload_and_html(tmp_path):
    r = run(GEN / "opd_visits.csv", tmp_path / "o.html", OCT, SEP, OPD_SPEC())
    ins = payload_of(r["path"])["insights"]
    assert ins["source"] == "template" and ins["scope"] == "All records" and ins["summary"]
    assert ins["findings"] and {"severity", "text"} <= set(ins["findings"][0])


@needs_data
def test_filters_change_numbers_and_scope():
    prep = prepare(GEN / "opd_visits.csv")
    whole = make_dashboard(prep, current=OCT, baseline=SEP, spec=OPD_SPEC())
    women = make_dashboard(prep, current=OCT, baseline=SEP, spec=OPD_SPEC(), filters={"gender": ["Female"]})
    n = lambda r: r["payload"]["data"]["widgets"]["visits"]["current"]["value"]
    assert 0 < n(women) < n(whole)
    assert women["insights"]["scope"] == "Gender: Female"
    assert women["insights"]["summary"].startswith("For Gender: Female")


@needs_data
def test_narrate_uses_llm_only_when_asked():
    prep = prepare(GEN / "opd_visits.csv")
    calls = []
    def llm(prompt):
        calls.append(1)
        return echo_first_finding(prompt)
    quiet = make_dashboard(prep, current=OCT, baseline=SEP, spec=OPD_SPEC(), llm=llm)
    assert quiet["insights"]["source"] == "template" and calls == []          # filter changes stay free
    loud = make_dashboard(prep, current=OCT, baseline=SEP, spec=OPD_SPEC(), llm=llm, narrate=True)
    assert loud["insights"]["source"] == "llm" and len(calls) == 1


@needs_data
def test_summary_cache_across_calls():
    prep, cache, calls = prepare(GEN / "opd_visits.csv"), {}, []
    def llm(prompt):
        calls.append(1)
        return echo_first_finding(prompt)
    for _ in range(3):
        make_dashboard(prep, current=OCT, baseline=SEP, spec=OPD_SPEC(), llm=llm, narrate=True, cache=cache)
    assert len(calls) == 1


@needs_data
def test_empty_view_gives_clear_message_not_a_crash():
    prep = prepare(GEN / "opd_visits.csv")
    r = make_dashboard(prep, current=OCT, baseline=SEP, spec=OPD_SPEC(), filters={"gender": ["Nobody"]}, narrate=True,
                       llm=lambda p: (_ for _ in ()).throw(AssertionError("no LLM call for an empty view")))
    assert "No records match" in r["insights"]["summary"]


def test_parse_filters():
    assert parse_filters(["gender=Female", "age_group=40-59,60+"]) == {"gender": ["Female"], "age_group": ["40-59", "60+"]}
    with pytest.raises(SystemExit):
        parse_filters(["oops"])


def test_insights_text_cannot_break_out_of_the_page(tmp_path):
    p = {"spec": {"widgets": []}, "data": {"dates": {}, "widgets": {}},
         "insights": {"summary": "</script><img src=x onerror=alert(1)>", "source": "llm", "scope": "", "findings": [], "notes": []}}
    out = render_dashboard(p, tmp_path / "x.html")
    assert out.read_text(encoding="utf-8").count("</script>") == 3
    assert payload_of(out)["insights"]["summary"].startswith("</script>")


def test_template_never_writes_spec_labels_with_innerhtml():
    """Widget labels can come from an LLM. They must reach the page via textContent, never innerHTML."""
    html = (ROOT / "templates/dashboard.html").read_text(encoding="utf-8")
    assert "${w.label}" not in html.replace("textContent = w.label", "")
    assert "textContent = w.label" in html