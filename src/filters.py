"""Filters: slice the clean data before aggregation, so every number, finding and summary on screen
describes exactly the view the user selected.

    sel = {"gender": ["Female"], "age_group": ["40-59", "60+"]}     # empty list or missing key = no filter
    view = apply_filters(df, sel)
    build_payload(view, spec, ...)                                    # both periods use the same filters
"""
import pandas as pd

MAX_FILTER_VALUES = 15


def filterable_columns(profile, spec=None, max_cols=4):
    """Columns worth offering as filters: the spec's own list if it has one, else the first few
    low-cardinality categories from the profile."""
    cats = {c["name"]: c for c in profile["columns"] if c["role"] == "category"}
    if spec and spec.get("filters"):
        return [c for c in spec["filters"] if c in cats][:max_cols]
    ok = [n for n, c in cats.items() if 2 <= c["distinct"] <= MAX_FILTER_VALUES]
    return ok[:max_cols]


def filter_options(df, columns):
    """{column: sorted values} for building the filter widgets. Uses every value in the data,
    not just the top ones from the profile."""
    return {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in columns}


def apply_filters(df, selected):
    """Rows matching ALL selected columns (any of the chosen values within a column)."""
    out = df
    for col, values in (selected or {}).items():
        if not values:
            continue
        if col not in df.columns:
            raise ValueError(f"Cannot filter on unknown column '{col}'.")
        out = out[out[col].astype(str).isin([str(v) for v in values])]
    return out


def active_filters(selected):
    return {c: v for c, v in (selected or {}).items() if v}


def describe_filters(selected):
    """'Gender: Female; Age group: 40-59, 60+' or 'All records'."""
    act = active_filters(selected)
    if not act:
        return "All records"
    nice = lambda c: c.replace("_", " ").strip().capitalize()
    return "; ".join(f"{nice(c)}: {', '.join(str(x) for x in v)}" for c, v in act.items())