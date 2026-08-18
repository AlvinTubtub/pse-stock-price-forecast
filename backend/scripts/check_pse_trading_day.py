#!/usr/bin/env python3
"""CLI utility to check whether a given date is a PSE trading day.

Uses the repository's single source of truth (services.pse_calendar.PSECalendar).
Always uses Philippine Time (UTC+8) by default.

Usage:
    python scripts/check_pse_trading_day.py
    python scripts/check_pse_trading_day.py --date 2026-08-21
    python scripts/check_pse_trading_day.py --date 2026-08-24

Exit code is always 0 (a holiday is an expected non-error condition).
If running inside GitHub Actions (GITHUB_OUTPUT is set), outputs:
    is_trading_day=true|false
    status=OPEN|CLOSED
    reason=<holiday reason if closed>
    next_trading_day=YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from services.pse_calendar import get_calendar

PHT = timezone(timedelta(hours=8))  # Philippine Time (UTC+8)


def check_date(target_date: date | None = None) -> dict[str, str | bool]:
    """Check trading status for a date and return metadata dict."""
    if target_date is None:
        target_date = datetime.now(PHT).date()

    calendar = get_calendar()
    is_open = calendar.is_trading_day(target_date)
    reason = calendar.get_holiday_reason(target_date)
    next_session = calendar.next_trading_day(target_date)

    return {
        "date": target_date.isoformat(),
        "is_trading_day": is_open,
        "status": "OPEN" if is_open else "CLOSED",
        "reason": reason or ("Trading Session" if is_open else "Non-Trading Day"),
        "action": "PROCEED with daily trading-data update" if is_open else "SKIP daily trading-data update",
        "next_trading_day": next_session.isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check if a date is a PSE trading day.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to check in YYYY-MM-DD format (defaults to current date in Philippine Time).",
    )
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None
    result = check_date(target_date)

    # Print human-readable summary
    print("=" * 60)
    print("PSE Trading Calendar Check")
    print(f"Date: {result['date']}")
    print(f"PSE Status: {result['status']}")
    if not result["is_trading_day"]:
        print(f"Reason: {result['reason']}")
    print(f"Action: {result['action']}")
    print(f"Next PSE Trading Session: {result['next_trading_day']}")
    print("=" * 60)

    # Write to GITHUB_OUTPUT if available
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"is_trading_day={'true' if result['is_trading_day'] else 'false'}\n")
            f.write(f"status={result['status']}\n")
            f.write(f"reason={result['reason']}\n")
            f.write(f"next_trading_day={result['next_trading_day']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
