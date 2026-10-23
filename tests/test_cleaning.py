import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cleaning import BLANK_LABEL, DataValidationError, clean_data, load_file


def frame(**over):
    base = {"visit_date": ["2026-10-01", "2026-10-02", "2026-10-02"],
            "diagnosis": ["Sprain", "Sprain", "Sprain"], "vas": [1.0, 2.0, 2.0]}
    base.update(over)
    return pd.DataFrame(base)


def test_removes_duplicates():
    df, rep = clean_data(frame())
    assert rep["duplicates_removed"] == 1 and len(df) == 2


def test_fills_blank_categories():
    df, rep = clean_data(frame(diagnosis=["Sprain", "", None], vas=[1, 2, 3]))
    assert (df["diagnosis"] == BLANK_LABEL).sum() == 2
    assert rep["blanks_filled"]["diagnosis"] == 2


def test_mixed_date_formats():
    df, _ = clean_data(frame(visit_date=["2026-10-14", "14/10/2026", "15/10/2026"], vas=[1, 2, 3]))
    assert df["visit_date"].dt.day.tolist() == [14, 14, 15]


def test_iso_dates_keep_their_month_when_day_is_12_or_less():
    df, _ = clean_data(frame(visit_date=["2026-10-05", "2026-09-03", "05/10/2026"], vas=[1, 2, 3]))
    assert [d.strftime("%Y-%m-%d") for d in df["visit_date"]] == ["2026-09-03", "2026-10-05", "2026-10-05"]


def test_drops_a_few_bad_dates():
    df, rep = clean_data(frame(visit_date=["2026-10-01", None, "2026-10-03"], vas=[1, 2, 3]))
    assert rep["bad_dates_dropped"] == 1 and len(df) == 2


def test_rejects_empty_file():
    with pytest.raises(DataValidationError, match="empty"):
        clean_data(pd.DataFrame())


def test_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="Missing required column"):
        clean_data(frame(), required_columns=["ward"])


def test_rejects_unreadable_dates():
    with pytest.raises(DataValidationError, match="not valid dates"):
        clean_data(frame(visit_date=["a", "b", "c"]))


def test_rejects_no_date_column():
    with pytest.raises(DataValidationError, match="No date column"):
        clean_data(pd.DataFrame({"x": ["a", "b"], "y": [1, 2]}))


def test_none_is_a_real_category_not_a_blank(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("visit_date,investigation,vas\n2026-10-01,None,1.5\n2026-10-02,,2.5\n2026-10-03,X-Ray,\n")
    df, rep = clean_data(load_file(f))
    assert df["investigation"].tolist() == ["None", BLANK_LABEL, "X-Ray"]
    assert rep["blanks_filled"] == {"investigation": 1}
    assert df["vas"].isna().sum() == 1            # numeric blanks stay numeric blanks


def test_rejects_unsupported_file_type():
    with pytest.raises(DataValidationError, match="Unsupported"):
        load_file("notes.txt")


@pytest.mark.skipif(not (ROOT / "data/generated/opd_visits_dirty.csv").exists(),
                    reason="run scripts/generate_fake_data.py first")
def test_dirty_opd_file_end_to_end():
    raw = load_file(ROOT / "data/generated/opd_visits_dirty.csv")
    df, rep = clean_data(raw, required_columns=["visit_date", "diagnosis"])
    assert rep["duplicates_removed"] >= 15
    assert rep["bad_dates_dropped"] == 10
    assert df["visit_date"].notna().all()
    assert not (df["diagnosis"] == "").any()