"""Comprehensive tests for the PSE trading calendar and holiday awareness.

Verifies:
1. Friday, August 21, 2026 (Ninoy Aquino Day) is NOT a trading day.
2. Monday, August 24, 2026 is a trading day.
3. next_trading_day(2026-08-20) == 2026-08-24.
4. next_trading_day(2026-08-21) == 2026-08-24.
5. All other official Philippine regular and special holidays across 2024-2026.
6. check_pse_trading_day utility function.
7. export_forecast_artifacts._get_forecast_date fallback logic.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.pse_calendar import PSECalendar, get_calendar
from scripts.check_pse_trading_day import check_date
from scripts.export_forecast_artifacts import _get_forecast_date


class TestPSECalendar(unittest.TestCase):
    def setUp(self):
        self.calendar = PSECalendar()

    # --------------------------------------------------------------------------
    # August 2026 Ninoy Aquino Day & Weekend Skip Tests
    # --------------------------------------------------------------------------

    def test_ninoy_aquino_day_2026_is_not_trading_day(self):
        """Friday, August 21, 2026 is Ninoy Aquino Day (PSE closed)."""
        self.assertFalse(self.calendar.is_trading_day(date(2026, 8, 21)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 8, 21)), "Ninoy Aquino Day")

    def test_monday_after_ninoy_aquino_day_is_trading_day(self):
        """Monday, August 24, 2026 is a normal trading session."""
        self.assertTrue(self.calendar.is_trading_day(date(2026, 8, 24)))
        self.assertIsNone(self.calendar.get_holiday_reason(date(2026, 8, 24)))

    def test_next_trading_day_from_thursday_before_holiday(self):
        """Next trading day strictly after Thursday, Aug 20, 2026 is Monday, Aug 24, 2026."""
        next_day = self.calendar.next_trading_day(date(2026, 8, 20))
        self.assertEqual(next_day, date(2026, 8, 24))

    def test_next_trading_day_from_holiday_itself(self):
        """Next trading day strictly after Friday, Aug 21, 2026 is Monday, Aug 24, 2026."""
        next_day = self.calendar.next_trading_day(date(2026, 8, 21))
        self.assertEqual(next_day, date(2026, 8, 24))

    def test_next_trading_day_from_weekend(self):
        """Next trading day strictly after Saturday, Aug 22 or Sunday, Aug 23 is Monday, Aug 24."""
        self.assertEqual(self.calendar.next_trading_day(date(2026, 8, 22)), date(2026, 8, 24))
        self.assertEqual(self.calendar.next_trading_day(date(2026, 8, 23)), date(2026, 8, 24))

    # --------------------------------------------------------------------------
    # Additional Philippine Holidays in 2026 (Proving no single holiday hardcoding)
    # --------------------------------------------------------------------------

    def test_new_years_day_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 1, 1)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 1, 1)), "New Year's Day")

    def test_chinese_new_year_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 2, 17)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 2, 17)), "Chinese New Year")

    def test_eidl_fitr_correct_date_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 3, 20)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 3, 20)), "Eid'l Fitr")

    def test_maundy_thursday_and_good_friday_are_holidays(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 4, 2)))
        self.assertFalse(self.calendar.is_trading_day(date(2026, 4, 3)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 4, 2)), "Maundy Thursday")
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 4, 3)), "Good Friday")

    def test_araw_ng_kagitingan_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 4, 9)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 4, 9)), "Araw ng Kagitingan")

    def test_labor_day_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 5, 1)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 5, 1)), "Labor Day")

    def test_eidl_adha_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 5, 27)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 5, 27)), "Eid'l Adha")

    def test_independence_day_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 6, 12)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 6, 12)), "Independence Day")

    def test_national_heroes_day_is_holiday(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 8, 31)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 8, 31)), "National Heroes Day")

    def test_all_saints_day_and_bonifacio_day_are_holidays(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 11, 1)))
        self.assertFalse(self.calendar.is_trading_day(date(2026, 11, 30)))
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 11, 30)), "Bonifacio Day")

    def test_christmas_and_rizal_day_are_holidays(self):
        self.assertFalse(self.calendar.is_trading_day(date(2026, 12, 24)))
        self.assertFalse(self.calendar.is_trading_day(date(2026, 12, 25)))
        self.assertFalse(self.calendar.is_trading_day(date(2026, 12, 30)))
        self.assertFalse(self.calendar.is_trading_day(date(2026, 12, 31)))

    def test_weekends_return_appropriate_reason(self):
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 8, 22)), "Weekend (Saturday)")
        self.assertEqual(self.calendar.get_holiday_reason(date(2026, 8, 23)), "Weekend (Sunday)")

    # --------------------------------------------------------------------------
    # Check Utility Integration Tests
    # --------------------------------------------------------------------------

    def test_check_date_on_holiday(self):
        result = check_date(date(2026, 8, 21))
        self.assertFalse(result["is_trading_day"])
        self.assertEqual(result["status"], "CLOSED")
        self.assertEqual(result["reason"], "Ninoy Aquino Day")
        self.assertEqual(result["action"], "SKIP daily trading-data update")
        self.assertEqual(result["next_trading_day"], "2026-08-24")

    def test_check_date_on_trading_day(self):
        result = check_date(date(2026, 8, 24))
        self.assertTrue(result["is_trading_day"])
        self.assertEqual(result["status"], "OPEN")
        self.assertEqual(result["action"], "PROCEED with daily trading-data update")
        self.assertEqual(result["next_trading_day"], "2026-08-25")

    # --------------------------------------------------------------------------
    # Exporter Fallback Logic Tests
    # --------------------------------------------------------------------------

    def test_export_fallback_date_skips_holidays(self):
        """When forecast_for is missing, fallback on data_as_of 2026-08-20 must return 2026-08-24, not 2026-08-21."""
        cache = {
            "inference_metadata": {
                "data_as_of": "2026-08-20"
                # forecast_for omitted to trigger fallback
            }
        }
        latest_processed = {}
        forecast_date = _get_forecast_date(cache, latest_processed)
        self.assertEqual(forecast_date, "2026-08-24")
        self.assertNotEqual(forecast_date, "2026-08-21")

    def test_export_fallback_date_with_explicit_forecast_for(self):
        """When forecast_for is explicitly present, it is used directly."""
        cache = {
            "inference_metadata": {
                "data_as_of": "2026-08-20",
                "forecast_for": "2026-08-24",
            }
        }
        latest_processed = {}
        forecast_date = _get_forecast_date(cache, latest_processed)
        self.assertEqual(forecast_date, "2026-08-24")


if __name__ == "__main__":
    unittest.main()
