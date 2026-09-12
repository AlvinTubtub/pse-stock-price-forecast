"""Unit tests for strict raw-data loading and validation."""

from datetime import date
from pathlib import Path

import pytest

from config.companies import COMPANIES, get_company
from config.settings import SETTINGS
from src.data.loader import load_company_history
from src.data.validator import (
    OhlcvValidationError,
    validate_and_sort_records,
    validate_required_columns,
)


def row(
    trading_date: str,
    *,
    open_price: object = "10",
    high: object = "12",
    low: object = "9",
    close: object = "11",
    volume: object = "1000",
) -> dict[str, object]:
    return {
        "Date": trading_date,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }


def test_records_are_sorted_chronologically() -> None:
    records = validate_and_sort_records(
        [row("2024-01-03"), row("2024-01-02"), row("2024-01-04")]
    )

    assert [record.trading_date for record in records] == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]


def test_duplicate_trading_dates_are_rejected() -> None:
    with pytest.raises(OhlcvValidationError, match="Duplicate trading dates"):
        validate_and_sort_records([row("2024-01-02"), row("2024-01-02")])


def test_missing_required_column_is_rejected() -> None:
    with pytest.raises(OhlcvValidationError, match="Volume"):
        validate_required_columns(["Date", "Open", "High", "Low", "Close"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"close": ""}, "Close is blank"),
        ({"close": "not-a-number"}, "Close is not numeric"),
        ({"close": "NaN"}, "Close must be finite"),
        ({"volume": "Infinity"}, "Volume must be finite"),
        ({"volume": "-1"}, "Volume cannot be negative"),
        ({"high": "8"}, "High cannot be below"),
        ({"low": "10.5"}, "Low cannot be above"),
    ],
)
def test_invalid_ohlcv_is_rejected_without_imputation(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(OhlcvValidationError, match=message):
        validate_and_sort_records([row("2024-01-02", **overrides)])


def test_loader_uses_only_configured_raw_files() -> None:
    configured_files = {company.raw_filename for company in COMPANIES}
    raw_files = {path.name for path in SETTINGS.raw_data_dir.glob("*.csv")}

    assert SETTINGS.raw_data_dir == Path(__file__).resolve().parents[1] / "data" / "raw"
    assert raw_files == configured_files
    assert load_company_history("ali")[0].trading_date == date(2020, 1, 2)
    assert get_company("ALI").symbol == "ALI"


def test_unknown_company_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown configured company"):
        load_company_history("UNKNOWN")
