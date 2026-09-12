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


@pytest.mark.parametrize(("origin", "expected"), [
    (date(2026, 9, 10), date(2026, 9, 11)),
    (date(2026, 9, 11), date(2026, 9, 14)),
    (date(2026, 11, 27), date(2026, 12, 1)),
    (date(2026, 12, 23), date(2026, 12, 28)),
    (date(2026, 4, 1), date(2026, 4, 6)),
])
def test_configured_closures_are_default(origin: date, expected: date) -> None:
    assert PSETradingCalendar().next_trading_day(origin) == expected
    assert next_pse_trading_day(origin) == expected
    assert PSETradingCalendar.with_holidays([]).next_trading_day(origin) == expected


def test_cli_emergency_closure_preserves_configured_closures() -> None:
    import argparse
    from scripts._common import add_runtime_options

    parser = argparse.ArgumentParser()
    add_runtime_options(parser)
    arguments = parser.parse_args(["--holiday", "2026-12-28"])
    calendar = PSETradingCalendar.with_holidays(arguments.holiday)
    assert calendar.next_trading_day(date(2026, 12, 23)) == date(2026, 12, 29)


def test_special_working_holiday_is_not_a_closure() -> None:
    assert PSETradingCalendar().is_trading_day(date(2026, 2, 25))
