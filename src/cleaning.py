"""Step 1 of the pipeline: load, validate and clean an uploaded file.

    df, report = clean_data(raw_df, required_columns=["visit_date", "diagnosis"])

Bad files raise DataValidationError with a message you can show directly in the UI.
"""
from pathlib import Path

import pandas as pd

BLANK_LABEL = "Not recorded"
MAX_BAD_DATE_SHARE = 0.5   # reject the file if more than half the dates are unreadable


class DataValidationError(ValueError):
    """The uploaded file cannot be used. The message is safe to show to the user."""


def load_file(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # "None" and "NA" can be real category values (e.g. investigation = None), so only
        # genuinely empty cells and explicit null markers count as missing.
        return pd.read_csv(path, keep_default_na=False, na_values=["", "NaN", "nan", "null", "NULL", "#N/A"])
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise DataValidationError(f"Unsupported file type '{path.suffix}'. Upload a .csv or .xlsx file.")


def parse_dates(series):
    """ISO dates (2026-10-05) are read as-is; anything else is read day-first (05/10/2026).
    Parsing everything day-first would silently swap month and day on ISO dates like 2026-10-05."""
    iso = pd.to_datetime(series, errors="coerce", format="ISO8601")
    rest = iso.isna() & series.notna()
    if rest.any():
        iso[rest] = pd.to_datetime(series[rest], errors="coerce", format="mixed", dayfirst=True)
    return iso


def find_date_column(df):
    """Return the most likely date column: a name containing 'date', else the first parseable column."""
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    for col in df.select_dtypes(exclude=["number", "datetime", "bool"]).columns:
        parsed = parse_dates(df[col])
        if parsed.notna().mean() > 0.8:
            return col
    return None


def clean_data(df, date_col=None, required_columns=None):
    """Return (clean_df, report). Raises DataValidationError for unusable files."""
    if df is None or df.empty:
        raise DataValidationError("The file is empty. Upload a file with at least one row of data.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report = {"rows_in": len(df)}

    missing = [c for c in (required_columns or []) if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. Found: {', '.join(df.columns)}.")

    date_col = date_col or find_date_column(df)
    if date_col is None or date_col not in df.columns:
        raise DataValidationError("No date column found. Add a column such as 'visit_date' or name one explicitly.")
    report["date_column"] = date_col

    # Dates: accept ISO and day-first strings in the same column
    parsed = parse_dates(df[date_col])
    bad = int(parsed.isna().sum())
    if bad / len(df) > MAX_BAD_DATE_SHARE:
        raise DataValidationError(
            f"{bad} of {len(df)} values in '{date_col}' are not valid dates. Check the column and re-upload.")
    df[date_col] = parsed
    df = df[parsed.notna()]
    report["bad_dates_dropped"] = bad

    # Text columns: trim whitespace, turn blanks into one consistent label
    text_cols = [c for c in df.select_dtypes(exclude=["number", "datetime", "bool"]).columns if c != date_col]
    filled = {}
    for c in text_cols:
        s = df[c].astype("string").str.strip()
        blank = s.isna() | (s == "")
        if blank.any():
            filled[c] = int(blank.sum())
        df[c] = s.mask(blank, BLANK_LABEL).astype(object)
    report["blanks_filled"] = filled

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    df = df.sort_values(date_col).reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report