"""Headless tests for the Streamlit app (no browser, no network)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

GEN = ROOT / "data/generated"
pytestmark = pytest.mark.skipif(not (GEN / "opd_visits.csv").exists(), reason="run scripts/generate_fake_data.py first")


@pytest.fixture
def app(monkeypatch):
    for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)               # no key: every LLM call must fall back cleanly
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60)
    return at.run()


def page(at):
    """The dashboard HTML handed to the iframe."""
    return at.get("iframe")[0].proto.srcdoc


def insights_of(at):
    import json, re
    return json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', page(at), re.S).group(1).replace("<\\/", "</"))["insights"]


def test_loads_sample_and_renders_dashboard(app):
    assert not app.exception
    assert "Design: rules" in app.sidebar.caption[0].value or "Design: fallback" in app.sidebar.caption[0].value
    assert "Total records" in page(app) and "Auto summary" in page(app)


def test_filter_changes_the_page_and_scope(app):
    gender = next(m for m in app.sidebar.multiselect if m.label == "Gender")
    gender.set_value(["Female"]).run()
    assert not app.exception
    assert "Gender: Female" in page(app)


def test_summarise_without_key_falls_back_and_says_so(app):
    app.sidebar.button(key="summarise").click().run()
    assert not app.exception
    assert any("template summary" in w.value for w in app.sidebar.warning)
    assert "Auto summary" in page(app)


def test_example_spec_selectable(app):
    next(s for s in app.sidebar.selectbox if s.label == "Dashboard design").select("Spec: opd_example").run()
    assert not app.exception
    assert "CLARITY" in page(app)


def test_ai_design_without_key_falls_back(app):
    next(s for s in app.sidebar.selectbox if s.label == "Dashboard design").select("AI-designed (LLM)").run()
    assert not app.exception and "Design: fallback" in app.sidebar.caption[0].value


def test_switching_to_icu_data(app):
    next(s for s in app.sidebar.selectbox if s.label == "...or use a sample file").select("icu_admissions.csv").run()
    assert not app.exception
    assert "Diagnosis" in [m.label for m in app.sidebar.multiselect]


def test_dirty_file_is_cleaned_and_reported(app):
    next(s for s in app.sidebar.selectbox if s.label == "...or use a sample file").select("opd_visits_dirty.csv").run()
    assert not app.exception and "Rows:" in " ".join(m.value for m in app.sidebar.markdown)


def fake_llm(monkeypatch, summary):
    import json
    from src import llm
    monkeypatch.setattr(llm, "call_llm", lambda prompt: json.dumps({"summary": summary}))


def test_ai_summary_is_shown_when_it_passes_the_number_check(app, monkeypatch):
    fake_llm(monkeypatch, "Visits and unique patients both increased, and pain scores improved.")
    app.sidebar.button(key="summarise").click().run()
    assert not app.exception and not app.sidebar.warning
    assert "AI summary" in page(app) and "pain scores improved" in page(app)


def test_ai_summary_with_an_invented_number_is_rejected(app, monkeypatch):
    fake_llm(monkeypatch, "Visits rose by 9999% this period.")
    app.sidebar.button(key="summarise").click().run()
    assert any("not in the findings" in w.value for w in app.sidebar.warning)
    ins = insights_of(app)
    assert ins["source"] == "template" and "9999" not in ins["summary"]


def test_changing_a_filter_drops_the_stale_ai_summary(app, monkeypatch):
    fake_llm(monkeypatch, "Visits and unique patients both increased.")
    app.sidebar.button(key="summarise").click().run()
    assert "AI summary" in page(app)
    next(m for m in app.sidebar.multiselect if m.label == "Gender").set_value(["Male"]).run()
    assert "Auto summary" in page(app) and "Gender: Male" in page(app)       # a summary of a different view must not linger


def test_period_comparison_mode(app):
    app.sidebar.radio[0].set_value("Two periods (current vs the one before)").run()
    assert not app.exception and "Total records" in page(app)


def test_tiny_baseline_warns_next_to_the_controls(app):
    from datetime import date
    slider = next(x for x in app.sidebar.slider if x.label == "Baseline date and current date")
    slider.set_value((date(2026, 8, 1), date(2026, 10, 31))).run()      # baseline = the first day only
    assert not app.exception
    assert any("baseline" in w.value.lower() for w in app.sidebar.warning)
    assert "baseline_small" in page(app)