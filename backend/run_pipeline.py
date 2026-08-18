#!/usr/bin/env python3
"""Headless runner for the PSE data-ingestion pipeline (PDF -> data/raw/ CSVs).

In production this is invoked exclusively by the Fast Pipeline
(.github/workflows/update_pipeline.yml, Monday-Friday, always with
--no-train), which is itself triggered externally by Cron-job.org
(4:00 PM Philippine Time) via a repository_dispatch call.

Model retraining is a separate concern, handled weekly by
.github/workflows/train_models.yml calling
services.model_selector.train_and_select_all() directly.

Usage:
    python run_pipeline.py                  # download + process + train
    python run_pipeline.py --no-download     # only process staged PDFs
    python run_pipeline.py --no-train        # skip retraining
    python run_pipeline.py --no-inference    # skip daily inference
    python run_pipeline.py --start-date 2026-07-01 --end-date 2026-07-27

Exit codes:
    0  success
    1  failure — nothing usable extracted, or daily inference failed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from services.pdf_pipeline import run_pipeline
from services.pdf_pipeline.config import RAW_DIR, TARGET_COMPANIES
from services.pse_calendar import get_calendar

PHT = timezone(timedelta(hours=8))  # Philippine Time (UTC+8, no DST)

BASE_DIR = Path(__file__).resolve().parent
PREDICTION_CACHE_DIR = BASE_DIR / "prediction_cache"

EXPECTED_TICKERS = sorted(TARGET_COMPANIES.keys())

# Statuses for which we still check/attempt daily inference — raw CSVs on
# disk are meaningful (and may be stale relative to the cache) even on a
# run that didn't ingest anything new this time. "error" is excluded:
# something went genuinely wrong upstream and we don't want to mask that
# by quietly refreshing forecasts.
INFERENCE_ELIGIBLE_STATUSES = {"ok", "merged_with_warnings", "no_files", "no_rows"}

# Statuses that still count as a successful CI run — "no_files" means the
# pipeline correctly found nothing new to do (e.g. a market holiday, or the
# data is already current), which is a normal outcome, not a failure.
SUCCESS_STATUSES = {"ok", "merged_with_warnings", "no_files"}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _latest_raw_date(symbol: str, raw_dir: Path = RAW_DIR) -> date | None:
    """Latest Date present in data/raw/<symbol>.csv, or None if the file
    is missing/empty."""
    csv_path = raw_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["Date"], parse_dates=["Date"])
    except Exception:
        return None
    if df.empty:
        return None
    return df["Date"].max().date()


def _cached_data_as_of(symbol: str, cache_dir: Path = PREDICTION_CACHE_DIR) -> date | None:
    """The data_as_of date the cached forecast was last computed against,
    or None if there is no cache yet or it has never had inference run
    (no inference_metadata block)."""
    cache_path = cache_dir / f"{symbol}.json"
    if not cache_path.exists():
        return None
    try:
        cache = json.loads(cache_path.read_text())
    except Exception:
        return None
    meta = cache.get("inference_metadata") or {}
    data_as_of = meta.get("data_as_of")
    if not data_as_of:
        return None
    try:
        return date.fromisoformat(data_as_of)
    except ValueError:
        return None


def tickers_needing_inference(
    tickers: list[str] = EXPECTED_TICKERS,
    raw_dir: Path = RAW_DIR,
    cache_dir: Path = PREDICTION_CACHE_DIR,
) -> list[str]:
    """Compare, per expected ticker, the latest raw OHLCV date against the
    latest date the cached forecast was actually computed against.

    A ticker needs inference when:
      - it has no cache yet, or the cache has never had inference run
        (no inference_metadata.data_as_of), or
      - raw data is strictly newer than what the cache was computed on.

    A ticker is skipped (considered current) when raw data exists and its
    date is <= the cache's data_as_of — this covers both "the cache is
    already current" and "the market has been closed since the last run"
    without needing a separate calendar check: if no new trading session
    has posted data, the raw date simply hasn't advanced.

    Tickers with no raw CSV at all are left for the 15-ticker enforcement
    in daily_inference to report explicitly, rather than silently skipped
    here.
    """
    stale: list[str] = []
    for symbol in tickers:
        latest_raw = _latest_raw_date(symbol, raw_dir)
        if latest_raw is None:
            # No raw data yet for this ticker — nothing to run inference
            # against; the 15-ticker enforcement step will flag this.
            continue
        cached_as_of = _cached_data_as_of(symbol, cache_dir)
        if cached_as_of is None or cached_as_of < latest_raw:
            stale.append(symbol)
    return stale


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PSE PDF ingestion pipeline headlessly.")
    parser.add_argument(
        "--no-download", dest="download", action="store_false",
        help="Skip fetching new reports from PSE EDGE; only process PDFs already in data/pdf_reports/.",
    )
    parser.add_argument(
        "--no-train", dest="train_models", action="store_false",
        help="Skip model retraining after ingestion (only update data/raw/ CSVs). "
             "Useful for lightweight CI runs that don't need scikit-learn/statsmodels/torch installed.",
    )
    parser.add_argument(
        "--inference", dest="run_inference", action="store_true",
        help="Run daily inference using persisted weekly models after data merge. "
             "This is the default for the Fast Pipeline; use --no-inference to skip.",
    )
    parser.add_argument(
        "--no-inference", dest="run_inference", action="store_false",
        help="Skip the daily inference step (keeps stale predictions in prediction_cache/).",
    )
    parser.add_argument("--start-date", type=_parse_date, default=None, help="YYYY-MM-DD, defaults to the day after the newest data on file.")
    parser.add_argument("--end-date", type=_parse_date, default=None, help="YYYY-MM-DD, defaults to today.")
    parser.add_argument(
        "--ignore-calendar", dest="ignore_calendar", action="store_true", default=False,
        help="Bypass the PSE trading calendar check (force-run on weekends or holidays).",
    )
    parser.set_defaults(download=True, train_models=True, run_inference=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.monotonic()

    today_pht = datetime.now(PHT).date()
    calendar = get_calendar()

    # Guard: on non-trading days (weekends & PSE holidays), skip daily update if no explicit date range was requested
    if not args.start_date and not args.end_date and not args.ignore_calendar:
        if not calendar.is_trading_day(today_pht):
            reason = calendar.get_holiday_reason(today_pht) or "Non-Trading Day"
            next_session = calendar.next_trading_day(today_pht)
            print("=" * 60)
            print("PSE Trading Calendar Check")
            print(f"Date: {today_pht}")
            print("PSE Status: CLOSED")
            print(f"Reason: {reason}")
            print("Action: SKIP daily trading-data update")
            print(f"Next PSE Trading Session: {next_session}")
            print("=" * 60)
            print("Finished successfully (holiday/non-trading day skip).")
            return 0

    print("=" * 60)
    print("Starting pipeline...")
    print("=" * 60)

    if args.download:
        print("Downloading reports...")
    print("Extracting...")
    print("Cleaning...")
    print("Validating...")
    print("Merging...")
    if args.train_models:
        print("Training models...")
    if args.run_inference:
        print("Daily inference (persisted weekly models)...")

    result = run_pipeline(
        download=args.download,
        start_date=args.start_date,
        end_date=args.end_date,
        train_models=args.train_models,
    )

    # ------------------------------------------------------------------
    # DAILY INFERENCE: run after data merge, before frontend export
    # ------------------------------------------------------------------
    inference_result = None
    if args.run_inference and result["status"] in INFERENCE_ELIGIBLE_STATUSES:
        # Don't rely on merge_summaries alone to decide whether inference
        # runs — a day with no new PDF (status "no_files") can still have
        # raw data that's newer than the cache (e.g. a previous inference
        # run failed, or data was added out-of-band). Compare every
        # expected ticker's latest raw date against its cached
        # data_as_of and only skip when everything is already current.
        stale_tickers = tickers_needing_inference()
        if stale_tickers:
            print("-" * 60)
            print(
                f"{len(stale_tickers)}/{len(EXPECTED_TICKERS)} ticker(s) have stale or "
                f"missing cached forecasts vs. raw data: {', '.join(stale_tickers)}"
            )
            print("Running daily inference...")
            try:
                from scripts.daily_inference import run_daily_inference
                inference_result = run_daily_inference()
                print(f"Daily inference: {inference_result['status']} "
                      f"({len(inference_result['symbols_processed'])} OK, "
                      f"{len(inference_result['symbols_failed'])} failed)")
                if inference_result["symbols_failed"]:
                    print("Failures:")
                    for sym, err in inference_result["symbols_failed"].items():
                        print(f"  {sym}: {err}")
                    # Fail the workflow if daily inference fails for any symbol
                    result["status"] = "error"
                    result["error"] = (
                        f"Daily inference failed for {len(inference_result['symbols_failed'])} symbol(s): "
                        + ", ".join(inference_result["symbols_failed"].keys())
                    )
                missing_tickers = sorted(set(EXPECTED_TICKERS) - set(inference_result["symbols_processed"]) - set(inference_result["symbols_failed"]))
                if missing_tickers:
                    # Enforce the full 15-ticker universe here too, in case
                    # a ticker's raw CSV doesn't exist at all yet.
                    result["status"] = "error"
                    result["error"] = (
                        (result["error"] + "; " if result.get("error") else "")
                        + f"Missing expected ticker(s) entirely (no raw data): {', '.join(missing_tickers)}"
                    )
            except Exception as exc:
                print(f"Daily inference step failed unexpectedly: {exc}")
                result["status"] = "error"
                result["error"] = f"Daily inference exception: {exc}"
        else:
            print(
                f"All {len(EXPECTED_TICKERS)} cached forecasts are already current "
                "with the latest raw data — skipping daily inference."
            )

    elapsed = round(time.monotonic() - started, 2)
    status = result["status"]

    print("-" * 60)
    if result["download"] is not None:
        dl = result["download"]
        print(f"Download: {len(dl.downloaded)} new, {len(dl.skipped)} already had, "
              f"{len(dl.not_found)} not published yet, {len(dl.errors)} failed")
    print(f"PDFs processed : {result['pdf_count']} ({result['parsed_count']} parsed OK)")
    print(f"Records extracted: {result['record_count']}")
    if result["parse_warnings"]:
        print(f"Warnings        : {len(result['parse_warnings'])}")
    if result["merge_summaries"]:
        print(f"Symbols updated : {len(result['merge_summaries'])}")
    if result["post_validation_errors"]:
        print(f"Post-validation failures: {len(result['post_validation_errors'])}")
    if result["training"]:
        print(f"Models trained  : {len(result['training']['best_models'])} ticker(s)")
        print("Statistical tests: statistical_tests.json")
    if inference_result:
        print(f"Daily inference : {inference_result['status']} ({len(inference_result['symbols_processed'])} symbols)")
    print(f"Status          : {status}")
    print(f"Elapsed         : {elapsed}s")
    print("-" * 60)

    if status in SUCCESS_STATUSES:
        print("Finished successfully.")
        return 0

    print(f"Finished with a failure status: {status}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print("See data/pdf_pipeline/pipeline.log for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
