"""Layer 1 analytics: turn a payload (spec + current/baseline numbers) into ranked findings.

    findings = build_findings(payload)         # every finding, most important first
    top = top_findings(findings, k=8)          # what the UI shows
    text = template_summary(findings)          # no-LLM summary built from the same findings

Works on the payload, not the raw rows, so it is automatically filter-aware: whatever filters and
dates produced the payload are what get analysed. Each finding is a dict:
    {"kind", "severity" (high|medium|low), "text", "subject", "score"}
The text is already a readable sentence, which makes it the template fallback for the LLM summary.
"""
import statistics
from datetime import datetime

SEVERITY = {"high": 3, "medium": 2, "low": 1}
SMALL_N = 30            # fewer records than this and changes may be noise
MIN_COUNT = 5           # ignore category moves among tiny counts
BLANK = "Not recorded"
KIND_CAP = {"category_shift": 3, "combo_shift": 2, "top_share": 2, "data_quality": 2}


def _fmt(x):
    if x is None:
        return "n/a"
    x = float(x)
    return f"{int(x):,}" if x.is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _pct(cur, base):
    return None if not base else (cur - base) / abs(base) * 100


def _sev(pct, cuts=(25, 50)):
    a = abs(pct)
    return "high" if a >= cuts[1] else "medium" if a >= cuts[0] else "low"


def _bump(sev):
    return {"low": "medium", "medium": "high", "high": "high"}[sev]


def _label(w):
    """Widget label, tolerating payloads built by hand without one."""
    return str(w.get("label") or "").strip() or str(w.get("id", "this measure"))


def _finding(kind, severity, text, subject, score):
    return {"kind": kind, "severity": severity, "text": text, "subject": subject, "score": round(float(score), 2)}


def _rows(payload):
    """(current, baseline) record counts from the first plain 'count everything' KPI, or (None, None)."""
    r = payload["data"].get("rows")                  # recorded by the aggregator, independent of the spec
    if r:
        return r["current"], r.get("baseline")
    for w in payload["spec"]["widgets"]:
        if w["type"] == "kpi" and w.get("agg", "count") == "count" and not w.get("where") and not w.get("column"):
            d = payload["data"]["widgets"][w["id"]]
            return d["current"]["value"], d.get("baseline", {}).get("value")
    return None, None


# ------------------------------------------------------------------ finders
def _sample_size(payload):
    n, base_n = _rows(payload)
    if n == 0:
        return [_finding("no_data", "high", "No records match the current filters and dates.", "sample", 1e9)], True, False
    out = []
    if n is not None and n < SMALL_N:
        extra = f" (baseline has {_fmt(base_n)})" if base_n is not None else ""
        out.append(_finding("small_sample", "high",
                            f"Only {_fmt(n)} records in this view{extra}; treat changes as indicative, not conclusive.",
                            "sample", 1e8))
    tiny = base_n is not None and base_n < SMALL_N      # percentages against a tiny baseline mean nothing
    if tiny and (n is None or n >= SMALL_N):
        what = "has no records" if base_n == 0 else f"has only {_fmt(base_n)} records"
        out.append(_finding("baseline_small", "high",
                            f"The baseline {what}, so changes measured against it are not reliable. "
                            "Choose a later baseline date or a longer baseline period.", "sample", 1e8))
    return out, False, tiny


def _kpis(payload):
    out = []
    for w in payload["spec"]["widgets"]:
        if w["type"] != "kpi":
            continue
        d = payload["data"]["widgets"][w["id"]]
        if "baseline" not in d:
            continue
        cur, base = d["current"]["value"], d["baseline"]["value"]
        if cur == base:
            continue
        if base == 0:
            out.append(_finding("kpi_change", "medium", f"{_label(w)} is {_fmt(cur)} now, up from none before.", w["id"], 100))
            continue
        pct = _pct(cur, base)
        if abs(pct) < 5:
            continue
        sev = _sev(pct)
        hib = w.get("higher_is_better")
        if hib is not None and (pct > 0) != hib:        # explicitly the unfavourable direction
            sev = _bump(sev)
        if w.get("agg", "count") == "count" and (base < MIN_COUNT or abs(cur - base) < 3):
            sev = "low"
        verb = "rose" if pct > 0 else "fell"
        out.append(_finding("kpi_change", sev, f"{_label(w)} {verb} {abs(pct):.0f}% ({_fmt(base)} to {_fmt(cur)}).", w["id"], abs(pct)))
    return out


def _breakdowns(payload, compare=True):
    out = []
    for w in payload["spec"]["widgets"]:
        if w["type"] != "breakdown":
            continue
        d = payload["data"]["widgets"][w["id"]]
        cur = dict(zip(d["current"]["labels"], d["current"]["values"]))
        lab = _label(w)[0].lower() + _label(w)[1:]
        total = (d.get("totals") or {}).get("current") or sum(cur.values())   # whole period, not just the bars shown

        if w.get("agg", "count") == "count" and total:
            top, v = max(cur.items(), key=lambda kv: kv[1])
            if top != BLANK and v / total >= 0.5:
                out.append(_finding("top_share", "low", f"'{top}' makes up {v / total * 100:.0f}% of {lab}.", w["id"], v / total * 100))
            if cur.get(BLANK, 0) / total >= 0.10:
                share = cur[BLANK] / total * 100
                out.append(_finding("data_quality", "high" if share >= 25 else "medium",
                                    f"{share:.0f}% of records have no value for {lab} ('{BLANK}').", w["id"], share))

        if not compare or "baseline" not in d or w.get("agg", "count") != "count":
            continue
        base = dict(zip(d["baseline"]["labels"], d["baseline"]["values"]))
        best = None
        for cat in cur:
            if cat == BLANK:
                continue
            c, b = cur[cat], base.get(cat, 0)
            if max(c, b) < MIN_COUNT or c == b:
                continue
            pct = _pct(c, b) if b else 100.0
            if abs(pct) >= 20 and (best is None or abs(c - b) > abs(best[1] - best[2])):
                best = (cat, c, b, pct)
        if best:
            cat, c, b, pct = best
            verb = "rose" if c > b else "fell"
            sev = "high" if abs(pct) >= 100 else "medium" if abs(pct) >= 40 else "low"
            out.append(_finding("category_shift", sev, f"In {lab}, '{cat}' {verb} from {_fmt(b)} to {_fmt(c)} ({pct:+.0f}%).", w["id"], abs(c - b)))
    return out


def _spikes(payload):
    out = []
    for w in payload["spec"]["widgets"]:
        if w["type"] != "trend":
            continue
        cur = payload["data"]["widgets"][w["id"]]["current"]
        pts = [(l, v) for l, v in zip(cur["labels"], cur["values"]) if v is not None]
        if len(pts) < 5:
            continue
        vals = [v for _, v in pts]
        mean, sd, med = statistics.fmean(vals), statistics.pstdev(vals), statistics.median(vals)
        label, peak = max(pts, key=lambda p: p[1])
        z, ratio = (peak - mean) / sd if sd else 0, peak / med if med else float("inf")
        if z < 2 or ratio < 2.5 or peak - med < 5:      # plain count noise (9 vs a typical 4) is not a spike
            continue
        weekly = _is_weekly(cur["labels"])
        when = f"the week of {label}" if weekly else label
        typical = f"{_fmt(med)} per {'week' if weekly else 'day'}"
        sev = "high" if ratio >= 3 else "medium" if z >= 3 else "low"
        out.append(_finding("spike", sev, f"Activity peaked at {_fmt(peak)} on {when}, well above the typical {typical}.", w["id"], z * 10))
    return out


def _is_weekly(labels):
    try:
        a, b = (datetime.strptime(l + " 2000", "%b %d %Y") for l in labels[:2])
        return (b - a).days == 7
    except ValueError:
        return False


def _combos(payload):
    out = []
    for w in payload["spec"]["widgets"]:
        d = payload["data"]["widgets"].get(w["id"], {})
        if w["type"] != "combo" or "baseline" not in d:
            continue
        c, b = d["current"], d["baseline"]
        name = w.get("line_label") or "Average value"
        best = None
        for i, cat in enumerate(c["labels"]):
            if cat == BLANK or c["bar"][i] < MIN_COUNT or not b["line"][i]:
                continue
            pct = _pct(c["line"][i], b["line"][i])
            if abs(pct) >= 15 and (best is None or abs(pct) > abs(best[3])):
                best = (cat, b["line"][i], c["line"][i], pct)
        if best:
            cat, bv, cv, pct = best
            out.append(_finding("combo_shift", _sev(pct, (30, 60)),
                                f"{name} for '{cat}' {'rose' if pct > 0 else 'fell'} {abs(pct):.0f}% ({_fmt(bv)} to {_fmt(cv)}).",
                                w["id"], abs(pct)))
    return out


# ------------------------------------------------------------------ public
def build_findings(payload):
    """All findings, most important first. An empty view returns one 'no data' finding."""
    head, empty, tiny = _sample_size(payload)
    if empty:
        return head
    # With a tiny baseline every comparison is noise: keep the warning plus findings that need no baseline.
    found = head + ([] if tiny else _kpis(payload)) + _breakdowns(payload, compare=not tiny) + _spikes(payload) + ([] if tiny else _combos(payload))
    return sorted(found, key=lambda f: (-SEVERITY[f["severity"]], -f["score"]))


def top_findings(findings, k=8):
    """First k findings, with a cap per kind so one widget type cannot crowd out the rest."""
    seen, out = {}, []
    for f in findings:
        n = seen.get(f["kind"], 0)
        if n >= KIND_CAP.get(f["kind"], k):
            continue
        seen[f["kind"]] = n + 1
        out.append(f)
        if len(out) == k:
            break
    return out


def template_summary(findings, k=4):
    """Plain-language summary from the top findings, no LLM involved."""
    top = top_findings(findings, k)
    if not top:
        return "No notable changes between the two periods."
    return " ".join(f["text"] for f in top)