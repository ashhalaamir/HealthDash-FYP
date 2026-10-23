import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import llm
from src.cleaning import clean_data, load_file
from src.pipeline import run
from src.profiler import profile_data
from src.spec_generator import build_prompt, generate_spec, parse_json

GEN = ROOT / "data/generated"
needs_data = pytest.mark.skipif(not (GEN / "planted_facts.json").exists(),
                                reason="run scripts/generate_fake_data.py first")


@pytest.fixture(scope="module")
def icu():
    df, rep = clean_data(load_file(GEN / "icu_admissions.csv"))
    return df, profile_data(df, rep["date_column"])


GOOD = {"title": "ICU", "widgets": [
    {"id": "n", "type": "kpi", "label": "Admissions", "agg": "count"},
    {"id": "los", "type": "kpi", "label": "Average stay", "agg": "mean", "column": "length_of_stay_days", "higher_is_better": False},
    {"id": "t", "type": "trend", "label": "Admissions over time", "agg": "count"},
    {"id": "d", "type": "breakdown", "label": "By diagnosis", "by": "diagnosis", "chart": "bar"},
    {"id": "g", "type": "breakdown", "label": "By gender", "by": "gender", "chart": "donut"},
    {"id": "o", "type": "breakdown", "label": "Outcomes", "by": "outcome", "chart": "donut"},
]}


def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}
    with pytest.raises(Exception):
        parse_json("no json here")


@needs_data
def test_prompt_has_columns_but_no_patient_rows(icu):
    df, profile = icu
    prompt = build_prompt(profile, focus="ICU capacity")
    assert "length_of_stay_days" in prompt and "ICU capacity" in prompt
    assert not any(pid in prompt for pid in df["patient_id"].head(20))      # no identifiers leak


@needs_data
def test_good_llm_answer_is_used(icu):
    df, profile = icu
    spec, info = generate_spec(profile, df, llm=lambda p: json.dumps(GOOD))
    assert info == {"source": "llm", "warnings": []} and len(spec["widgets"]) == 6


@needs_data
def test_hallucinated_column_drops_one_widget_not_the_dashboard(icu):
    df, profile = icu
    bad = {**GOOD, "widgets": GOOD["widgets"] + [{"id": "x", "type": "breakdown", "label": "x", "by": "blood_type"}]}
    spec, info = generate_spec(profile, df, llm=lambda p: json.dumps(bad))
    assert info["source"] == "llm" and len(spec["widgets"]) == 6 and "blood_type" in info["warnings"][0]


@needs_data
@pytest.mark.parametrize("reply", ["total nonsense", '{"widgets": "no"}', json.dumps({"widgets": GOOD["widgets"][:2]}),
                                   json.dumps({"widgets": [{"id": "a", "type": "kpi", "label": "a", "agg": "sum", "column": "nope"}] * 6})])
def test_unusable_answers_fall_back(icu, reply):
    df, profile = icu
    spec, info = generate_spec(profile, df, llm=lambda p: reply)
    assert info["source"] == "fallback" and len(spec["widgets"]) >= 5


@needs_data
def test_llm_exception_falls_back(icu):
    df, profile = icu
    def boom(_): raise llm.LLMError("quota exceeded")
    spec, info = generate_spec(profile, df, llm=boom)
    assert info["source"] == "fallback" and "quota exceeded" in info["warnings"][0]


@needs_data
def test_pipeline_uses_llm_spec_and_reports_source(tmp_path):
    r = run(GEN / "icu_admissions.csv", tmp_path / "x.html", llm=lambda p: json.dumps(GOOD))
    assert r["spec_source"] == "llm" and r["spec"]["title"] == "ICU"
    r = run(GEN / "icu_admissions.csv", tmp_path / "y.html", llm=lambda p: "garbage")
    assert r["spec_source"] == "fallback" and r["path"].exists()


# ---- provider plumbing, with the network mocked
class FakeResp:
    def __init__(self, data): self.data = data
    def raise_for_status(self): pass
    def json(self): return self.data


def test_gemini_request_and_parse(monkeypatch):
    seen = {}
    def fake_post(url, **kw):
        seen.update(url=url, **kw)
        return FakeResp({"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]})
    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "k"); monkeypatch.setenv("GEMINI_MODEL", "m1")
    assert llm.call_llm("hi", "gemini") == '{"ok": 1}'
    assert seen["url"].endswith("/models/m1:generateContent") and seen["headers"]["x-goog-api-key"] == "k"
    assert seen["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_openai_and_ollama_parse(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: FakeResp({"choices": [{"message": {"content": "A"}}]}))
    assert llm.call_llm("hi", "openai") == "A"
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: FakeResp({"response": "B"}))
    assert llm.call_llm("hi", "ollama") == "B"


def test_missing_key_and_unknown_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.call_llm("hi", "gemini")
    with pytest.raises(llm.LLMError, match="Unknown"):
        llm.call_llm("hi", "claude-ish")