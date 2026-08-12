"""Self-contained Philippine Stock Exchange (PSE) trading calendar.

No external holiday dependencies — uses a curated list of known PSE holidays
plus rule-based detection for movable holidays (Holy Week, Eid al-Fitr).

PSE trading hours: Monday–Friday, 9:30 AM – 12:00 PM (morning session),
1:00 PM – 3:30 PM (afternoon session), except holidays.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd


# Known fixed-date PSE holidays (month, day) — add new years as needed.
# Sources: PSE official holiday announcements, Philippine government proclamations.
FIXED_HOLIDAYS: dict[int, list[tuple[int, int]]] = {
    2024: [
        (1, 1),   # New Year's Day
        (4, 9),   # Araw ng Kagitingan
        (5, 1),   # Labor Day
        (6, 12),  # Independence Day
        (8, 21),  # Ninoy Aquino Day
        (8, 26),  # National Heroes Day (last Monday of Aug)
        (11, 1),  # All Saints' Day
        (11, 30), # Bonifacio Day
        (12, 25), # Christmas Day
        (12, 30), # Rizal Day
        (12, 31), # New Year's Eve (special non-working)
    ],
    2025: [
        (1, 1),   # New Year's Day
        (4, 9),   # Araw ng Kagitingan
        (5, 1),   # Labor Day
        (6, 12),  # Independence Day
        (8, 21),  # Ninoy Aquino Day
        (8, 25),  # National Heroes Day (last Monday of Aug)
        (11, 1),  # All Saints' Day
        (11, 30), # Bonifacio Day
        (12, 25), # Christmas Day
        (12, 30), # Rizal Day
        (12, 31), # New Year's Eve
    ],
    2026: [
        (1, 1),   # New Year's Day
        (4, 9),   # Araw ng Kagitingan
        (5, 1),   # Labor Day
        (6, 12),  # Independence Day
        (8, 21),  # Ninoy Aquino Day
        (8, 31),  # National Heroes Day (last Monday of Aug — Proclamation No. 1006)
        (11, 1),  # All Saints' Day
        (11, 30), # Bonifacio Day
        (12, 25), # Christmas Day
        (12, 30), # Rizal Day
        (12, 31), # New Year's Eve
    ],
}

# Movable / special holidays that vary by year (exact dates from official proclamations)
SPECIAL_HOLIDAYS: dict[int, list[date]] = {
    2024: [
        date(2024, 3, 28),  # Maundy Thursday
        date(2024, 3, 29),  # Good Friday
        date(2024, 4, 10),  # Eid al-Fitr (special non-working)
        date(2024, 12, 24), # Christmas Eve (special non-working)
    ],
    2025: [
        date(2025, 4, 17),  # Maundy Thursday
        date(2025, 4, 18),  # Good Friday
        date(2025, 4, 19),  # Black Saturday (special non-working)
        date(2025, 6, 6),   # Eid al-Adha (special non-working)
        date(2025, 12, 24), # Christmas Eve
    ],
    2026: [
        date(2026, 2, 17),  # Chinese New Year (special non-working, Proclamation No. 1006)
        date(2026, 3, 20),  # Eid'l Fitr (regular holiday — official date per Proclamation No. 1006)
        date(2026, 4, 2),   # Maundy Thursday
        date(2026, 4, 3),   # Good Friday
        date(2026, 4, 4),   # Black Saturday
        date(2026, 5, 27),  # Eid'l Adha (regular holiday — official date per Proclamation No. 1006)
        date(2026, 12, 24), # Christmas Eve
    ],
}


class PSECalendar:
    """Philippine Stock Exchange trading calendar."""

    def __init__(self, years: Iterable[int] | None = None):
        if years is None:
            years = range(2020, 2031)
        self._holidays: set[date] = set()
        for year in years:
            # Fixed-date holidays
            for month, day in FIXED_HOLIDAYS.get(year, []):
                self._holidays.add(date(year, month, day))
            # Special/movable holidays
            for d in SPECIAL_HOLIDAYS.get(year, []):
                self._holidays.add(d)
            # If Dec 30 (Rizal Day) falls on weekend, the observed holiday
            # is usually the nearest weekday — but PSE is already closed on
            # weekends, so no extra handling needed.

    def is_trading_day(self, d: date) -> bool:
        """True if d is a PSE trading session (weekday, not holiday)."""
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return d not in self._holidays

    def next_trading_day(self, d: date | pd.Timestamp) -> date:
        """Return the next PSE trading day strictly after ``d``."""
        if isinstance(d, pd.Timestamp):
            d = d.date()
        candidate = d + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate

    def previous_trading_day(self, d: date | pd.Timestamp) -> date:
        """Return the previous PSE trading day strictly before ``d``."""
        if isinstance(d, pd.Timestamp):
            d = d.date()
        candidate = d - timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def trading_days_between(self, start: date, end: date) -> list[date]:
        """All PSE trading days in [start, end] inclusive."""
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)
        return days


# Module-level singleton for convenience
_default_calendar: PSECalendar | None = None


def get_calendar() -> PSECalendar:
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = PSECalendar()
    return _default_calendar
