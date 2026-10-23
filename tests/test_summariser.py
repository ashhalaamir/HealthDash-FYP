import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.insights import build_findings
from src.summariser import build_prompt, make_context, numbers_in, summarise, unsupported_numbers
from tests.test_insights import payload


@pytest.fixture
def setup():
    p = payload()
    return build_findings(p), make_context(p, {"gender": ["Female"]}, "OPD")


def reply(text):
    return lambda prompt: json.dumps({"summary": text})


GOOD = "For Female patients, total visits rose 50% (100 to 150), while average VAS fell 25% (2 to 1.5)."


def test_good_summary_is_used(setup):
    out = summarise(*setup, llm=reply(GOOD))
    assert out["source"] == "llm" and out["summary"] == GOOD and out["notes"] == [] and out["findings"]


def test_invented_number_is_rejected(setup):
    out = summarise(*setup, llm=reply("Visits rose 50% (100 to 150), roughly 7.7 times the usual rate."))
    assert out["source"] == "template"
    assert "numbers not in the findings (7.7)" in out["notes"][0]


def test_wrong_number_is_rejected(setup):
    out = summarise(*setup, llm=reply("Total visits rose 55% (100 to 150)."))
    assert out["source"] == "template" and "55" in out["notes"][0]


@pytest.mark.parametrize("bad", ["not json at all", '{"nope": 1}', '{"summary": ""}', '{"summary": 5}'])
def test_unreadable_replies_fall_back(setup, bad):
    out = summarise(*setup, llm=lambda p: bad)
    assert out["source"] == "template" and out["notes"] and out["summary"]


def test_llm_exception_falls_back(setup):
    def boom(_): raise RuntimeError("quota exceeded")
    out = summarise(*setup, llm=boom)
    assert out["source"] == "template" and "quota exceeded" in out["notes"][0]


def test_no_llm_gives_template_with_scope(setup):
    out = summarise(*setup, llm=None)
    assert out["source"] == "template" and out["summary"].startswith("For Gender: Female: Total visits rose 50%")


def test_no_data_skips_the_llm():
    empty = payload(n={"current": {"value": 0}, "baseline": {"value": 100}})
    def never(_): raise AssertionError("LLM must not be called for an empty view")
    out = summarise(build_findings(empty), make_context(empty), llm=never)
    assert out["source"] == "template" and "No records match" in out["summary"]


def test_cache_means_one_call_per_view(setup):
    calls, cache = [], {}
    def llm(prompt):
        calls.append(1)
        return json.dumps({"summary": GOOD})
    summarise(*setup, llm=llm, cache=cache)
    summarise(*setup, llm=llm, cache=cache)
    assert len(calls) == 1
    other = (setup[0], {**setup[1], "filters": "Gender: Male"})
    summarise(*other, llm=llm, cache=cache)
    assert len(calls) == 2                                      # a different view is a different question


def test_prompt_carries_scope_and_findings_only(setup):
    prompt = build_prompt(setup[0], setup[1])
    assert "Gender: Female" in prompt and "Total visits rose 50% (100 to 150)." in prompt
    assert "ONLY the findings" in prompt and "not instructions" in prompt


def test_number_matching_is_format_tolerant():
    assert numbers_in("1,234 and 52% and 1.50 and 2026-10-01") == {1234.0, 52.0, 1.5, 2026.0, 10.0, 1.0}
    f, ctx = build_findings(payload()), {"current": "2026-10-01 to 2026-10-31", "baseline": "", "filters": "All records", "rows": 150}
    assert unsupported_numbers("Visits rose 50% (100 to 150) from 1 October.", f, ctx) == []