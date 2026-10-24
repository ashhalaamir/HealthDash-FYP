"""Step 3 of the pipeline: turn clean data + a dashboard spec + two periods into the payload
that templates/dashboard.html renders. All numbers are computed here, never by the LLM.

    payload = build_payload(df, spec, "visit_date",
                            current=("2026-08-01", "2026-10-30"),
                            baseline=("2026-08-01", "2026-10-01"))

Spec fields per widget (id and label are always required):
    kpi        agg, column?, where?        one number            {"value": n}
    trend      agg, column?, freq?         metric over time      {"labels", "values"}
    breakdown  by, agg, column?, top?, order?, sort?   metric per category   {"labels", "values"}
    combo      by, bar{agg,column?}, line{agg,column?}, top?   bars + line   {"labels", "bar", "line"}
agg is one of count, sum, mean, median, min, max, nunique. "column" is the measure, "by" the category.
"""
import pandas as pd

from .compare import compare_rows

AGGS = {"count", "sum", "mean", "median", "min", "max", "nunique"}
NEEDS_NUMERIC = {"sum", "mean", "median"}
TYPES = {"kpi", "trend", "breakdown", "combo"}
WEEKLY_AFTER_DAYS = 62


# ------------------------------------------------------------------ windows
def window(df, date_col, start, end):
    """Rows with start <= date <= end (end day included)."""
    d = df[date_col]
    return df[(d >= pd.Timestamp(start)) & (d < pd.Timestamp(end) + pd.Timedelta(days=1))]


def cumulative_windows(df, date_col, baseline_date, current_date):
    """Sir's style: everything up to each date. Returns (current, baseline) windows."""
    first = df[date_col].min().strftime("%Y-%m-%d")
    return (first, current_date), (first, baseline_date)


def previous_period(start, end):
    """Same-length window immediately before (start, end)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    days = (e - s).days + 1
    return (s - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), (s - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# --------------------------------------------------------------- validation
def validate_spec(spec, df):
    """Return (clean_spec, problems). Invalid widgets are dropped, so one bad LLM
    suggestion never breaks the dashboard."""
    problems, good, seen = [], [], set()

    def col_ok(c, need_numeric=False):
        if c not in df.columns:
            return f"unknown column '{c}'"
        if need_numeric and not pd.api.types.is_numeric_dtype(df[c]):
            return f"column '{c}' is not numeric"

    def agg_problem(a, col):
        if a not in AGGS:
            return f"unknown agg '{a}'"
        if a != "count" and not col:
            return f"agg '{a}' needs a column"
        return col and col_ok(col, a in NEEDS_NUMERIC)

    for w in spec.get("widgets", []):
        wid, t = w.get("id"), w.get("type")
        err = None
        if not wid or wid in seen:
            err = "missing or duplicate id"
        elif t not in TYPES:
            err = f"unknown type '{t}'"
        elif t in ("kpi", "trend"):
            err = agg_problem(w.get("agg", "count"), w.get("column"))
            for c in (w.get("where") or {}):
                err = err or col_ok(c)
        elif t == "breakdown":
            err = col_ok(w.get("by")) or agg_problem(w.get("agg", "count"), w.get("column"))
        elif t == "combo":
            err = col_ok(w.get("by"))
            for k in ("bar", "line"):
                part = w.get(k) or {}
                err = err or agg_problem(part.get("agg", "count"), part.get("column"))
        if err:
            problems.append(f"widget '{wid}' dropped: {err}")
        else:
            seen.add(wid)
            good.append(w)
    out = {**spec, "widgets": good}
    if "filters" in spec:
        out["filters"] = [c for c in spec["filters"] if c in df.columns]
        problems += [f"filter '{c}' dropped: unknown column '{c}'" for c in spec["filters"] if c not in df.columns]
    return out, problems


# --------------------------------------------------------------- primitives
def _num(x):
    if x is None or pd.isna(x):
        return 0
    x = float(x)
    return int(x) if x.is_integer() else round(x, 2)


def _where(df, cond):
    for col, val in (cond or {}).items():
        df = df[df[col] == val]
    return df


def _agg(df, agg, column):
    if agg == "count":
        return len(df)
    return 0 if df.empty else getattr(df[column], agg)()


def _grouped(df, by, agg, column):
    return df.groupby(by).size() if agg == "count" else df.groupby(by)[column].agg(agg)


def _kpi(df, w):
    return {"value": _num(_agg(_where(df, w.get("where")), w.get("agg", "count"), w.get("column")))}


def _trend(df, w, date_col, start, end):
    df = _where(df, w.get("where"))
    agg, col = w.get("agg", "count"), w.get("column")
    freq = w.get("freq") or ("W" if (pd.Timestamp(end) - pd.Timestamp(start)).days > WEEKLY_AFTER_DAYS else "D")
    if freq == "D":
        key, idx = df[date_col].dt.normalize(), pd.date_range(start, end, freq="D")
    else:
        key, idx = df[date_col].dt.to_period("W").dt.start_time, pd.period_range(start, end, freq="W").start_time
    s = df.groupby(key).size() if agg == "count" else df.groupby(key)[col].agg(agg)
    s = s.reindex(idx)
    if agg == "count":
        s = s.fillna(0)
    return {"labels": list(idx.strftime("%b %d")),
            "values": [None if pd.isna(v) else _num(v) for v in s]}


def _labels(cur, base, w, by_values):
    """Category order: explicit 'order', else top N by current value (or alphabetical if sort=label)."""
    if w.get("order"):
        return list(w["order"])
    s = by_values(cur) if len(cur) else by_values(base)
    labels = list(s.sort_values(ascending=False).head(w.get("top", 10)).index)
    return sorted(labels) if w.get("sort") == "label" else labels


def _series(df, labels, by, agg, column):
    if df is None:
        return None
    s = _grouped(df, by, agg, column) if len(df) else pd.Series(dtype=float)
    return [_num(v) for v in s.reindex(labels).fillna(0)]


def _breakdown(cur, base, w):
    by, agg, col = w["by"], w.get("agg", "count"), w.get("column")
    labels = _labels(cur, base, w, lambda d: _grouped(d, by, agg, col))
    out = {"current": {"labels": labels, "values": _series(cur, labels, by, agg, col)}}
    if base is not None:
        out["baseline"] = {"labels": labels, "values": _series(base, labels, by, agg, col)}
        out["compare"] = compare_rows(dict(zip(labels, out["current"]["values"])),
                                      dict(zip(labels, out["baseline"]["values"])), counts=(agg == "count"))
    return out


def _combo(cur, base, w):
    by, bar, line = w["by"], w.get("bar") or {}, w.get("line") or {}
    ba, bc = bar.get("agg", "count"), bar.get("column")
    la, lc = line.get("agg", "count"), line.get("column")
    labels = _labels(cur, base, w, lambda d: _grouped(d, by, ba, bc))
    mk = lambda d: {"labels": labels, "bar": _series(d, labels, by, ba, bc), "line": _series(d, labels, by, la, lc)}
    out = {"current": mk(cur)}
    if base is not None:
        out["baseline"] = mk(base)
        out["compare"] = compare_rows(dict(zip(labels, out["current"]["bar"])),
                                      dict(zip(labels, out["baseline"]["bar"])), counts=(ba == "count"))
    return out


# ------------------------------------------------------------------ payload
def build_payload(df, spec, date_col, current, baseline=None, labels=None):
    """current / baseline are (start, end) date pairs. Returns {"spec", "data", "warnings"}."""
    spec, problems = validate_spec(spec, df)
    cur = window(df, date_col, *current)
    base = window(df, date_col, *baseline) if baseline else None
    name = lambda p: f"{p[0]} to {p[1]}"
    dates = labels or {"current": name(current), "baseline": name(baseline) if baseline else ""}

    widgets = {}
    for w in spec["widgets"]:
        t = w["type"]
        if t == "kpi":
            r = {"current": _kpi(cur, w)}
            if base is not None:
                r["baseline"] = _kpi(base, w)
        elif t == "trend":
            r = {"current": _trend(cur, w, date_col, *current)}
            if base is not None:
                r["baseline"] = _trend(base, w, date_col, *baseline)
        elif t == "breakdown":
            r = _breakdown(cur, base, w)
        else:
            r = _combo(cur, base, w)
        widgets[w["id"]] = r
    return {"spec": spec, "data": {"dates": dates, "widgets": widgets}, "warnings": problems}