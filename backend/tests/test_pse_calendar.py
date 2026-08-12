"""Regression tests for the 2026 PSE holiday calendar corrections.

These lock in the officially confirmed 2026 non-trading dates (per
Malacañang Proclamation No. 1006) so a future edit can't silently
reintroduce the wrong/missing holidays that were fixed:

    - 2026-02-17  Chinese New Year (was missing)
    - 2026-03-20  Eid'l Fitr        (was the wrong estimated 05-25 date)
    - 2026-05-27  Eid'l Adha        (was missing)
    - 2026-08-31  National Heroes Day, last Monday of August (was 08-24)
    - 2026-05-25  must NOT be a holiday (the incorrect entry that was removed)
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.pse_calendar import PSECalendar


class TestPSECalendar2026(unittest.TestCase):
    def setUp(self):
        self.calendar = PSECalendar()

    def test_chinese_new_year_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 2, 17)))

    def test_eidl_fitr_correct_date_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 3, 20)))

    def test_eidl_adha_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 5, 27)))

    def test_incorrect_may_25_is_not_a_holiday(self):
        """The old (wrong) Eid'l Fitr placeholder date must be a normal
        trading day now that it has been removed. 2026-05-25 is a Monday."""
        self.assertEqual(date(2026, 5, 25).weekday(), 0)  # sanity: it's a weekday
        self.assertTrue(self.calendar.is_trading_day(date(2026, 5, 25)))

    def test_national_heroes_day_last_monday_of_august(self):
        """National Heroes Day 2026 is Aug 31 (last Monday of August),
        not Aug 24."""
        self.assertFalse(self.calendar.is_trading_day(date(2026, 8, 31)))
        self.assertTrue(self.calendar.is_trading_day(date(2026, 8, 24)))

    def test_next_trading_day_around_eidl_fitr(self):
        """Thursday Mar 19 -> Eid'l Fitr Fri Mar 20 is closed -> next session Mon Mar 23."""
        next_day = self.calendar.next_trading_day(date(2026, 3, 19))
        self.assertEqual(next_day, date(2026, 3, 23))

    def test_next_trading_day_around_chinese_new_year(self):
        """Chinese New Year 2026 falls on a Tuesday; Monday Feb 16 -> Wed Feb 18."""
        next_day = self.calendar.next_trading_day(date(2026, 2, 16))
        self.assertEqual(next_day, date(2026, 2, 18))

    def test_next_trading_day_around_eidl_adha(self):
        """Eid'l Adha 2026 falls on a Wednesday; Tuesday May 26 -> Thu May 28."""
        next_day = self.calendar.next_trading_day(date(2026, 5, 26))
        self.assertEqual(next_day, date(2026, 5, 28))


if __name__ == "__main__":
    unittest.main()
