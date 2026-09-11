"""Unit tests for PSE session and Manila-time helpers."""

from datetime import date, datetime, timezone

import pytest

from src.data.calendar import (
    DISPLAY_TIMEZONE_NAME,
    PSETradingCalendar,
    next_pse_trading_day,
    timestamp_for_display,
)


def test_next_trading_day_skips_weekend() -> None:
    assert next_pse_trading_day(date(2024, 1, 5)) == date(2024, 1, 8)


def test_next_trading_day_skips_explicit_pse_closure() -> None:
    calendar = PSETradingCalendar.with_holidays([date(2024, 1, 8)])

    assert calendar.next_trading_day(date(2024, 1, 5)) == date(2024, 1, 9)


def test_display_timestamp_uses_asia_manila() -> None:
    displayed = timestamp_for_display(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))

    assert DISPLAY_TIMEZONE_NAME == "Asia/Manila"
    assert displayed.isoformat() == "2024-01-01T08:00:00+08:00"


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        timestamp_for_display(datetime(2024, 1, 1, 0, 0))
