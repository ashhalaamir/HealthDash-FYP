"""Step 2 of the pipeline: describe a cleaned DataFrame so a spec can be generated for it.

    profile = profile_data(clean_df, date_col)

The result is plain JSON-safe data. It is what gets sent to the LLM (never the raw rows)
and what the rule-based fallback spec reads.

Column roles: date | id | category | numeric | text
"""
import re

import pandas as pd

MAX_CATEGORIES = 25        # more distinct values than this is not a useful chart axis
MAX_INT_CATEGORIES = 8     # small integer columns (ward 1-5, rating 1-5) behave like categories
TOP_VALUES = 10
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|^ref|_ref$|mrn)", re.I)


def classify(series, name, date_col):
    """Return the role of one column."""
    if name == date_col:
        return "date"
    if ID_NAME.search(name):
        return "id"
    n = series.nunique(dropna=True)
    if pd.api.types.is_bool_dtype(series):
        return "category"
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series) and n <= MAX_INT_CATEGORIES:
            return "category"
        return "numeric"
    return "category" if n <= MAX_CATEGORIES else "text"


def describe(series, role):
    info = {"nulls": int(series.isna().sum()), "distinct": int(series.nunique(dropna=True))}
    if role == "category":
        counts = series.value_counts().head(TOP_VALUES)
        info["top_values"] = {str(k): int(v) for k, v in counts.items()}
        info["is_binary"] = info["distinct"] == 2
    elif role == "numeric":
        info.update(min=round(float(series.min()), 2), max=round(float(series.max()), 2),
                    mean=round(float(series.mean()), 2))
    return info


def profile_data(df, date_col):
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data.")
    dates = pd.to_datetime(df[date_col])
    columns = []
    for name in df.columns:
        role = classify(df[name], name, date_col)
        columns.append({"name": name, "role": role, **describe(df[name], role)})
    return {
        "rows": int(len(df)),
        "date_column": date_col,
        "date_range": {"start": dates.min().strftime("%Y-%m-%d"), "end": dates.max().strftime("%Y-%m-%d")},
        "columns": columns,
        "by_role": {r: [c["name"] for c in columns if c["role"] == r]
                    for r in ("category", "numeric", "id", "text")},
    }