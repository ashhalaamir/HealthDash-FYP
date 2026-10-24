"""Step 5 of the pipeline: ask an LLM which widgets to show, then check its answer.

    spec, info = generate_spec(profile, df, title="ICU", llm=call_llm)

The LLM only ever sees the column profile (names, roles, top category counts, numeric ranges),
never patient rows. Whatever it returns goes through validate_spec, so a hallucinated column
drops one widget instead of breaking the dashboard. Any failure at all falls back to the
rule-based spec, so the pipeline always produces a dashboard.
"""
import json
import re

from .aggregator import validate_spec
from .fallback_spec import make_fallback_spec

MIN_WIDGETS = 5   # fewer valid widgets than this and the LLM answer is not worth using

PROMPT = """You design analytics dashboards. Given a profile of a dataset, return a dashboard spec as JSON.

WIDGET TYPES (every widget needs "id" (unique, short), "type" and "label" (plain English)):
- kpi: one number. Fields: agg, column (the measure; omit for count), where (optional, {{"col": "value"}} equality filter), higher_is_better (false when a rise is bad, e.g. waiting time, mortality, cost)
- trend: a metric over time. Fields: agg, column?, span (grid width 6 or 12)
- breakdown: a metric per category. Fields: by (category column), agg, column?, chart ("bar" or "donut"), top (max categories), order (optional explicit list of labels), span
- combo: bars plus a line on a second axis per category. Fields: by, bar {{agg, column?}}, line {{agg, column?}}, bar_label, line_label, span
agg is one of: count, sum, mean, median, min, max, nunique. "count" needs no column; sum/mean/median need a numeric column.

RULES
- Use ONLY column names that appear in the profile. Category columns for "by" and "where"; numeric columns as measures.
- Never use columns with role "id" or "text" as "by". An id column is fine for nunique (e.g. unique patients).
- Use "donut" only for 2-4 categories; use "bar" otherwise. Order age groups from youngest to oldest with "order".
- Include 3-4 kpi widgets, exactly one trend, 3-5 breakdown widgets and at most one combo. Pick what matters for this domain.
- Add "filters": 2-4 category columns (2-15 distinct values) that a manager would slice by, such as gender, age group or ward.
- Labels are for a non-technical reader. The "title" names the dataset in a few words.

{focus}DATASET PROFILE
{profile}

Return ONLY a JSON object: {{"title": "...", "widgets": [ ... ], "filters": ["col", ...]}}. No explanation, no markdown."""


def build_prompt(profile, focus=None):
    focus = f"FOCUS: {focus}\n\n" if focus else ""
    return PROMPT.format(focus=focus, profile=json.dumps(profile, indent=1))


def parse_json(text):
    """Pull a JSON object out of an LLM reply, tolerating ```json fences and chatter around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def generate_spec(profile, df, title="Dashboard", llm=None, focus=None):
    """Return (spec, info). info = {"source": "llm"|"fallback", "warnings": [...]}."""
    fallback = lambda why: (make_fallback_spec(profile, title), {"source": "fallback", "warnings": [why]})
    if llm is None:
        return make_fallback_spec(profile, title), {"source": "fallback", "warnings": []}   # chosen, not a failure
    try:
        spec = parse_json(llm(build_prompt(profile, focus)))
        if not isinstance(spec, dict) or not isinstance(spec.get("widgets"), list):
            return fallback("LLM reply had no widget list; used rule-based spec.")
    except Exception as e:          # network, quota, bad JSON: all handled the same way
        return fallback(f"LLM unavailable or unreadable ({type(e).__name__}: {e}); used rule-based spec.")

    spec["title"] = spec.get("title") or title
    try:
        clean, problems = validate_spec(spec, df)
    except Exception as e:                                    # never let a strange reply crash the app
        return fallback(f"LLM spec could not be validated ({type(e).__name__}: {e}); used rule-based spec.")
    if len(clean["widgets"]) < MIN_WIDGETS or not any(w["type"] == "kpi" for w in clean["widgets"]):
        return fallback(f"Only {len(clean['widgets'])} valid widgets from LLM; used rule-based spec. {problems}")
    return clean, {"source": "llm", "warnings": problems}