"""Deterministic cleanup for values extracted from PSE quotation tables."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import math
import re


_SPACE_PATTERN = re.compile(r"\s+")
_MISSING_NUMBERS = frozenset({"", "-", "--"})


class NumericCleaningError(ValueError):
    """Raised when a quotation value cannot be converted deterministically."""


@dataclass(frozen=True, slots=True)
class QuotationRecord:
    report_date: date
    issue_name: str
    symbol: str
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal | None
    volume: Decimal | None
    value: Decimal | None

    def ohlcv_mapping(self) -> dict[str, str]:
        return {
            "Date": self.report_date.isoformat(),
            "Open": decimal_text(self.open),
            "High": decimal_text(self.high),
            "Low": decimal_text(self.low),
            "Close": decimal_text(self.close),
            "Volume": decimal_text(self.volume),
        }


def clean_text(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", value.replace("\u00a0", " ").strip())


def clean_number(value: object) -> Decimal | None:
    """Parse commas and accounting negatives without guessing malformed input."""

    if value is None:
        return None
    raw = clean_text(str(value))
    if raw in _MISSING_NUMBERS:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1]
    normalized = raw.replace(",", "")
    try:
        parsed = Decimal(normalized)
    except InvalidOperation as exc:
        raise NumericCleaningError(f"Malformed numeric quotation value: {value!r}") from exc
    if not math.isfinite(float(parsed)):
        raise NumericCleaningError(f"Quotation value must be finite: {value!r}")
    return -parsed if negative else parsed


def decimal_text(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value, "f")


def clean_quotation(
    *,
    report_date: date,
    issue_name: str,
    symbol: str,
    open_value: object,
    high_value: object,
    low_value: object,
    close_value: object,
    volume_value: object,
    traded_value: object,
) -> QuotationRecord:
    return QuotationRecord(
        report_date=report_date,
        issue_name=clean_text(issue_name),
        symbol=clean_text(symbol).upper(),
        open=clean_number(open_value),
        high=clean_number(high_value),
        low=clean_number(low_value),
        close=clean_number(close_value),
        volume=clean_number(volume_value),
        value=clean_number(traded_value),
    )
