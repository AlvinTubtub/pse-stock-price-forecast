"""Tests for the run_pipeline.py retry logic and daily_inference.py
15-ticker universe enforcement.

Covers:
  - tickers_needing_inference() correctly detects a stale/missing cache
    vs. raw data, independent of merge_summaries.
  - run_daily_inference() fails (rather than silently skipping) any
    expected ticker with no raw CSV at all, when called in its default
    (production) "infer everything" mode.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline
from scripts import daily_inference


def _make_ohlcv_df(end_date: str, n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range(end=end_date, periods=n_days, freq="B")
    np.random.seed(7)
    base = 100.0 + np.cumsum(np.random.randn(n_days) * 0.5)
    df = pd.DataFrame({
        "Date": dates,
        "Open": base,
        "High": base + 1,
        "Low": base - 1,
        "Close": base,
        "Volume": 1_000_000,
    })
    return df


class TestTickersNeedingInference(unittest.TestCase):
    """run_pipeline.tickers_needing_inference() — the retry-logic fix."""

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp(prefix="pse_retry_test_")
        self.temp_path = Path(self._temp_dir)
        self.raw_dir = self.temp_path / "data" / "raw"
        self.cache_dir = self.temp_path / "prediction_cache"
        self.raw_dir.mkdir(parents=True)
        self.cache_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _write_raw(self, symbol: str, end_date: str):
        df = _make_ohlcv_df(end_date)
        df.to_csv(self.raw_dir / f"{symbol}.csv", index=False)

    def _write_cache(self, symbol: str, data_as_of: str | None):
        payload = {"next_close": {"lag": 1, "arima": 1, "lstm": 1}}
        if data_as_of:
            payload["inference_metadata"] = {
                "data_as_of": data_as_of,
                "forecast_for": "2026-08-10",
                "inference_at": "2026-08-07T16:00:00+08:00",
                "models_retrained": False,
                "model_source": "weekly_persisted_artifacts",
            }
        (self.cache_dir / f"{symbol}.json").write_text(json.dumps(payload))

    def test_stale_cache_is_flagged(self):
        """Raw data newer than the cached data_as_of -> needs inference."""
        self._write_raw("ALI", "2026-08-07")
        self._write_cache("ALI", "2026-08-06")  # cache one trading day behind
        stale = run_pipeline.tickers_needing_inference(
            tickers=["ALI"], raw_dir=self.raw_dir, cache_dir=self.cache_dir
        )
        self.assertEqual(stale, ["ALI"])

    def test_current_cache_is_not_flagged(self):
        """Cache already matches the latest raw date -> skip."""
        self._write_raw("ALI", "2026-08-07")
        self._write_cache("ALI", "2026-08-07")
        stale = run_pipeline.tickers_needing_inference(
            tickers=["ALI"], raw_dir=self.raw_dir, cache_dir=self.cache_dir
        )
        self.assertEqual(stale, [])

    def test_missing_inference_metadata_is_flagged(self):
        """A cache with no inference_metadata at all (e.g. training-only
        cache, never had inference run) must be treated as stale."""
        self._write_raw("ALI", "2026-08-07")
        self._write_cache("ALI", data_as_of=None)
        stale = run_pipeline.tickers_needing_inference(
            tickers=["ALI"], raw_dir=self.raw_dir, cache_dir=self.cache_dir
        )
        self.assertEqual(stale, ["ALI"])

    def test_no_cache_file_at_all_is_flagged(self):
        self._write_raw("ALI", "2026-08-07")
        stale = run_pipeline.tickers_needing_inference(
            tickers=["ALI"], raw_dir=self.raw_dir, cache_dir=self.cache_dir
        )
        self.assertEqual(stale, ["ALI"])

    def test_no_new_pdf_but_stale_cache_still_flagged(self):
        """The core bug being fixed: even with nothing new to merge this
        run (simulating a 'no_files' status day), a raw/cache mismatch
        from a previous failed run must still surface as needing inference —
        it cannot depend on merge_summaries."""
        self._write_raw("ALI", "2026-08-07")
        self._write_raw("BPI", "2026-08-07")
        self._write_cache("ALI", "2026-08-07")  # current
        self._write_cache("BPI", "2026-08-05")  # stale (e.g. prior inference failure)
        stale = run_pipeline.tickers_needing_inference(
            tickers=["ALI", "BPI"], raw_dir=self.raw_dir, cache_dir=self.cache_dir
        )
        self.assertEqual(stale, ["BPI"])

    def test_missing_raw_csv_is_not_flagged_here(self):
        """No raw data at all for a ticker isn't this function's concern —
        that's the 15-ticker enforcement's job (see TestUniverseEnforcement
        below), so it must not appear in the staleness list."""
        stale = run_pipeline.tickers_needing_inference(
            tickers=["GHOST"], raw_dir=self.raw_dir, cache_dir=self.cache_dir
        )
        self.assertEqual(stale, [])


class TestUniverseEnforcement(unittest.TestCase):
    """run_daily_inference() must fail if any expected ticker is missing
    entirely, when run in its default (production) mode."""

    def setUp(self):
        self._temp_dir = tempfile.mkdtemp(prefix="pse_universe_test_")
        self.temp_path = Path(self._temp_dir)

        self._orig_base_dir = daily_inference.BASE_DIR
        daily_inference.BASE_DIR = self.temp_path
        daily_inference.RAW_DIR = self.temp_path / "data" / "raw"
        daily_inference.MODELS_DIR = self.temp_path / "models"
        daily_inference.LAG_MODELS_DIR = daily_inference.MODELS_DIR / "lag_regression"
        daily_inference.ARIMA_MODELS_DIR = daily_inference.MODELS_DIR / "arima"
        daily_inference.LSTM_MODELS_DIR = daily_inference.MODELS_DIR / "lstm"
        daily_inference.PREDICTION_CACHE_DIR = self.temp_path / "prediction_cache"
        self._orig_production_history = daily_inference.PRODUCTION_HISTORY_DIR
        daily_inference.PRODUCTION_HISTORY_DIR = self.temp_path / "production_history"

        for d in (
            daily_inference.RAW_DIR,
            daily_inference.LAG_MODELS_DIR,
            daily_inference.ARIMA_MODELS_DIR,
            daily_inference.LSTM_MODELS_DIR,
            daily_inference.PREDICTION_CACHE_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # A deliberately incomplete universe: only 2 of the 15 expected
        # tickers actually have raw data.
        self._orig_expected = daily_inference.EXPECTED_TICKERS
        daily_inference.EXPECTED_TICKERS = ["ALI", "BPI", "GHOST_TICKER"]

        for sym in ("ALI", "BPI"):
            df = _make_ohlcv_df("2026-08-07")
            df.to_csv(daily_inference.RAW_DIR / f"{sym}.csv", index=False)
            for d in (daily_inference.LAG_MODELS_DIR, daily_inference.ARIMA_MODELS_DIR):
                (d / f"{sym}.pkl").write_bytes(b"dummy")
            (daily_inference.LSTM_MODELS_DIR / f"{sym}.pth").write_bytes(b"dummy")
            (daily_inference.PREDICTION_CACHE_DIR / f"{sym}.json").write_text(
                json.dumps({"next_close": {}})
            )

    def tearDown(self):
        daily_inference.BASE_DIR = self._orig_base_dir
        daily_inference.EXPECTED_TICKERS = self._orig_expected
        daily_inference.PRODUCTION_HISTORY_DIR = self._orig_production_history
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def test_missing_ticker_fails_the_batch(self):
        def mock_arima_load(path):
            sym = Path(path).stem
            df_sym = pd.read_csv(daily_inference.RAW_DIR / f"{sym}.csv")
            from unittest.mock import MagicMock
            m = MagicMock()
            m.nobs = len(df_sym)
            m.model.endog = df_sym["Close"].values.reshape(-1, 1)
            return m

        with patch("scripts.daily_inference.lag_regression.load"),              patch("scripts.daily_inference.lag_regression.predict_next", return_value=101.0),              patch("scripts.daily_inference.arima_model.load", side_effect=mock_arima_load),              patch("scripts.daily_inference.arima_model.predict_next", return_value=102.0),              patch("scripts.daily_inference.lstm_model.load"),              patch("scripts.daily_inference.lstm_model.predict_next", return_value=103.0):
            # symbols=None -> production mode -> enforces the full universe
            result = daily_inference.run_daily_inference(raw_dir=daily_inference.RAW_DIR, symbols=None)

        self.assertIn("GHOST_TICKER", result["symbols_failed"])
        self.assertIn("missing entirely", result["symbols_failed"]["GHOST_TICKER"])
        self.assertEqual(set(result["symbols_processed"]), {"ALI", "BPI"})
        self.assertIn(result["status"], ("partial_failure", "failure"))

    def test_explicit_symbols_list_does_not_enforce_universe(self):
        """Passing an explicit symbols= list (e.g. from tests, or a
        targeted re-run) must NOT trigger universe enforcement — only the
        production default (symbols=None) does."""
        result = daily_inference.run_daily_inference(
            raw_dir=daily_inference.RAW_DIR, symbols=["NOT_IN_UNIVERSE"]
        )
        self.assertIn("NOT_IN_UNIVERSE", result["symbols_failed"])
        self.assertNotIn("missing entirely", result["symbols_failed"]["NOT_IN_UNIVERSE"])


if __name__ == "__main__":
    unittest.main()
