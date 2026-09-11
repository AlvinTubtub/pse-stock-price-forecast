"""Strict validation for immutable raw OHLCV observations."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math
from typing import Final


REQUIRED_OHLCV_COLUMNS: Final[tuple[str, ...]] = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)


class OhlcvValidationError(ValueError):
    """Raised when raw OHLCV input is incomplete, malformed, or ambiguous."""


@dataclass(frozen=True, slots=True)
class OhlcvRecord:
    """One validated market session."""

    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def validate_required_columns(fieldnames: Sequence[str] | None) -> None:
    """Require each canonical OHLCV CSV column exactly once."""

    if fieldnames is None:
        raise OhlcvValidationError("CSV header is missing")
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        raise OhlcvValidationError(f"Duplicate CSV columns: {', '.join(duplicates)}")
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in fieldnames]
    if missing:
        raise OhlcvValidationError(f"Missing required OHLCV columns: {', '.join(missing)}")


def _parse_date(value: object, *, row_number: int) -> date:
    raw = str(value).strip() if value is not None else ""
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise OhlcvValidationError(
            f"Row {row_number}: Date must be a valid YYYY-MM-DD value, got {raw!r}"
        ) from exc
    if parsed.isoformat() != raw:
        raise OhlcvValidationError(
            f"Row {row_number}: Date must use canonical YYYY-MM-DD format, got {raw!r}"
        )
    return parsed


def _parse_number(value: object, *, column: str, row_number: int) -> float:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        raise OhlcvValidationError(f"Row {row_number}: {column} is blank")
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise OhlcvValidationError(
            f"Row {row_number}: {column} is not numeric: {raw!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise OhlcvValidationError(
            f"Row {row_number}: {column} must be finite, got {raw!r}"
        )
    return parsed


def parse_ohlcv_record(row: Mapping[str, object], *, row_number: int) -> OhlcvRecord:
    """Parse and validate one raw mapping without imputing any value."""

    values = {
        column: _parse_number(row.get(column), column=column, row_number=row_number)
        for column in REQUIRED_OHLCV_COLUMNS
        if column != "Date"
    }
    record = OhlcvRecord(
        trading_date=_parse_date(row.get("Date"), row_number=row_number),
        open=values["Open"],
        high=values["High"],
        low=values["Low"],
        close=values["Close"],
        volume=values["Volume"],
    )
    _validate_market_values(record, row_number=row_number)
    return record


def _validate_market_values(record: OhlcvRecord, *, row_number: int) -> None:
    prices = (record.open, record.high, record.low, record.close)
    if any(value <= 0 for value in prices):
        raise OhlcvValidationError(f"Row {row_number}: OHLC prices must be positive")
    if record.volume < 0:
        raise OhlcvValidationError(f"Row {row_number}: Volume cannot be negative")
    if record.high < max(record.open, record.low, record.close):
        raise OhlcvValidationError(
            f"Row {row_number}: High cannot be below Open, Low, or Close"
        )
    if record.low > min(record.open, record.high, record.close):
        raise OhlcvValidationError(
            f"Row {row_number}: Low cannot be above Open, High, or Close"
        )


def validate_and_sort_records(rows: Iterable[Mapping[str, object]]) -> tuple[OhlcvRecord, ...]:
    """Parse rows, reject duplicate dates, and return chronological records."""

    records = tuple(
        parse_ohlcv_record(row, row_number=row_number)
        for row_number, row in enumerate(rows, start=2)
    )
    if not records:
        raise OhlcvValidationError("OHLCV input contains no data rows")

    dates = [record.trading_date for record in records]
    duplicates = sorted(value for value, count in Counter(dates).items() if count > 1)
    if duplicates:
        formatted = ", ".join(value.isoformat() for value in duplicates)
        raise OhlcvValidationError(f"Duplicate trading dates: {formatted}")

    return tuple(sorted(records, key=lambda record: record.trading_date))


def require_chronological_records(records: Sequence[OhlcvRecord]) -> None:
    """Reject empty, unsorted, or duplicate validated-record sequences."""

    if not records:
        raise OhlcvValidationError("At least one OHLCV record is required")
    dates = [record.trading_date for record in records]
    if any(current >= following for current, following in zip(dates, dates[1:])):
        raise OhlcvValidationError(
            "OHLCV records must have unique dates in strictly increasing order"
        )
