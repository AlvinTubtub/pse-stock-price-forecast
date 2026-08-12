"""Tests for daily_inference.py.

All model loading and inference is mocked — no real LASSO/ARIMA/LSTM
models are trained during unit tests, so the suite runs in seconds.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import daily_inference
from services.pse_calendar import PSECalendar


class TestDailyInference(unittest.TestCase):
    """Unit tests for daily_inference.py using mocked models."""

    def setUp(self):
        # Create a fresh temp directory for each test
        self._temp_dir = tempfile.mkdtemp(prefix="pse_test_")
        self.temp_path = Path(self._temp_dir)

        # Point daily_inference at our temp directory structure
        self._orig_base_dir = daily_inference.BASE_DIR
        daily_inference.BASE_DIR = self.temp_path
        daily_inference.RAW_DIR = self.temp_path / "data" / "raw"
        daily_inference.MODELS_DIR = self.temp_path / "models"
        daily_inference.LAG_MODELS_DIR = daily_inference.MODELS_DIR / "lag_regression"
        daily_inference.ARIMA_MODELS_DIR = daily_inference.MODELS_DIR / "arima"
        daily_inference.LSTM_MODELS_DIR = daily_inference.MODELS_DIR / "lstm"
        daily_inference.PREDICTION_CACHE_DIR = self.temp_path / "prediction_cache"

        for d in (
            daily_inference.RAW_DIR,
            daily_inference.LAG_MODELS_DIR,
            daily_inference.ARIMA_MODELS_DIR,
            daily_inference.LSTM_MODELS_DIR,
            daily_inference.PREDICTION_CACHE_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Create test data with valid OHLCV consistency
        self.symbol = "TEST"
        self.n_days = 100
        dates = pd.date_range(end="2026-08-07", periods=self.n_days, freq="B")
        np.random.seed(42)
        base = 100.0 + np.cumsum(np.random.randn(self.n_days) * 0.5)
        self.df = pd.DataFrame({
            "Date": dates,
            "Open": base + np.random.randn(self.n_days) * 0.3,
            "High": base + abs(np.random.randn(self.n_days)) * 0.8 + 0.2,
            "Low": base - abs(np.random.randn(self.n_days)) * 0.8 - 0.2,
            "Close": base + np.random.randn(self.n_days) * 0.3,
            "Volume": np.random.randint(1_000_000, 10_000_000, self.n_days),
        })
        # Ensure OHLC consistency: Low <= min(Open,Close) <= max(Open,Close) <= High
        self.df["Low"] = self.df[["Open", "High", "Low", "Close"]].min(axis=1) - 0.01
        self.df["High"] = self.df[["Open", "High", "Low", "Close"]].max(axis=1) + 0.01
        self.csv_path = daily_inference.RAW_DIR / f"{self.symbol}.csv"
        self.df.to_csv(self.csv_path, index=False)

        # Seed a prediction cache with dummy training-time metadata
        self.cache = {
            "metrics": {
                "lag_reg": {"rmse": "1.0", "mae": "0.8", "mase": "0.9", "r2": "0.5"},
                "arima": {"rmse": "1.1", "mae": "0.9", "mase": "1.0", "r2": "0.4"},
                "lstm": {"rmse": "1.2", "mae": "1.0", "mase": "1.1", "r2": "0.3"},
                "naive": {"rmse": "2.0", "mae": "1.5", "mase": "1.5", "r2": "0.0"},
            },
            "next_close": {"lag": 100.0, "arima": 100.0, "lstm": 100.0},
            "backtest30": list(self.df["Close"].iloc[-30:]),
            "backtest_by_model": {
                "Lag-Informed Regression": list(self.df["Close"]),
                "ARIMA": list(self.df["Close"]),
                "LSTM": list(self.df["Close"]),
            },
        }
        (daily_inference.PREDICTION_CACHE_DIR / f"{self.symbol}.json").write_text(
            json.dumps(self.cache)
        )

    def tearDown(self):
        daily_inference.BASE_DIR = self._orig_base_dir
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # LASSO inference (mocked)
    # ------------------------------------------------------------------
    def test_lasso_inference_no_training(self):
        """LASSO inference loads artifact and predicts without fitting."""
        dummy_path = daily_inference.LAG_MODELS_DIR / f"{self.symbol}.pkl"
        dummy_path.write_bytes(b"dummy")

        mock_artifact = MagicMock()
        with patch("scripts.daily_inference.lag_regression.load", return_value=mock_artifact) as mock_load,              patch("scripts.daily_inference.lag_regression.predict_next", return_value=101.5) as mock_pred:
            pred = daily_inference._infer_lasso(self.symbol, self.df)
            mock_load.assert_called_once_with(dummy_path)
            mock_pred.assert_called_once_with(mock_artifact, self.df)
            self.assertEqual(pred, 101.5)

    def test_lasso_missing_artifact_fails(self):
        """Missing LASSO artifact must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            daily_inference._infer_lasso(self.symbol, self.df)

    # ------------------------------------------------------------------
    # ARIMA inference (mocked) — lineage validation + append
    # ------------------------------------------------------------------
    def test_arima_inference_no_refit(self):
        """ARIMA loads model, validates lineage against endog, appends with refit=False."""
        dummy_path = daily_inference.ARIMA_MODELS_DIR / f"{self.symbol}.pkl"
        dummy_path.write_bytes(b"dummy")

        n_endog = 90
        mock_model = MagicMock()
        mock_model.nobs = n_endog
        mock_model.model.endog = self.df["Close"].iloc[:n_endog].values.reshape(-1, 1)

        mock_appended = MagicMock()
        mock_model.append.return_value = mock_appended

        with patch("scripts.daily_inference.arima_model.load", return_value=mock_model) as mock_load,              patch("scripts.daily_inference.arima_model.predict_next", return_value=102.0) as mock_pred:
            pred = daily_inference._infer_arima(self.symbol, self.df)

            mock_load.assert_called_once_with(dummy_path)
            mock_model.append.assert_called_once()
            args, kwargs = mock_model.append.call_args
            appended_values = args[0]
            self.assertEqual(len(appended_values), 10)
            self.assertFalse(kwargs.get("refit", True))
            mock_pred.assert_called_once_with(mock_appended)
            self.assertEqual(pred, 102.0)

    def test_arima_missing_artifact_fails(self):
        """Missing ARIMA artifact must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            daily_inference._infer_arima(self.symbol, self.df)

    def test_arima_lineage_mismatch_fails(self):
        """ARIMA must fail if model.model.endog does not match historical data."""
        dummy_path = daily_inference.ARIMA_MODELS_DIR / f"{self.symbol}.pkl"
        dummy_path.write_bytes(b"dummy")

        mock_model = MagicMock()
        mock_model.nobs = 90
        mock_model.model.endog = (self.df["Close"].iloc[:90].values + 100.0).reshape(-1, 1)

        with patch("scripts.daily_inference.arima_model.load", return_value=mock_model):
            with self.assertRaises(ValueError) as ctx:
                daily_inference._infer_arima(self.symbol, self.df)
            self.assertIn("lineage mismatch", str(ctx.exception).lower())

    def test_arima_no_new_observations(self):
        """ARIMA with no new data should still forecast without calling append."""
        dummy_path = daily_inference.ARIMA_MODELS_DIR / f"{self.symbol}.pkl"
        dummy_path.write_bytes(b"dummy")

        mock_model = MagicMock()
        mock_model.nobs = len(self.df)
        mock_model.model.endog = self.df["Close"].values.reshape(-1, 1)

        with patch("scripts.daily_inference.arima_model.load", return_value=mock_model) as mock_load,              patch("scripts.daily_inference.arima_model.predict_next", return_value=103.0) as mock_pred:
            pred = daily_inference._infer_arima(self.symbol, self.df)
            mock_model.append.assert_not_called()
            mock_pred.assert_called_once_with(mock_model)
            self.assertEqual(pred, 103.0)

    # ------------------------------------------------------------------
    # LSTM inference (mocked)
    # ------------------------------------------------------------------
    def test_lstm_inference_no_training(self):
        """LSTM inference loads artifact and predicts without fitting."""
        dummy_path = daily_inference.LSTM_MODELS_DIR / f"{self.symbol}.pth"
        dummy_path.write_bytes(b"dummy")

        mock_artifact = {"state_dict": {}, "seq_len": 10, "hidden_size": 50}
        with patch("scripts.daily_inference.lstm_model.load", return_value=mock_artifact) as mock_load,              patch("scripts.daily_inference.lstm_model.predict_next", return_value=104.0) as mock_pred:
            pred = daily_inference._infer_lstm(self.symbol, self.df)
            mock_load.assert_called_once_with(dummy_path)
            mock_pred.assert_called_once_with(mock_artifact, self.df)
            self.assertEqual(pred, 104.0)

    def test_lstm_missing_artifact_fails(self):
        """Missing LSTM artifact must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            daily_inference._infer_lstm(self.symbol, self.df)

    # ------------------------------------------------------------------
    # End-to-end: infer_symbol
    # ------------------------------------------------------------------
    def test_infer_symbol_updates_cache(self):
        """infer_symbol must update next_close and add inference_metadata."""
        for d in (daily_inference.LAG_MODELS_DIR, daily_inference.ARIMA_MODELS_DIR):
            (d / f"{self.symbol}.pkl").write_bytes(b"dummy")
        (daily_inference.LSTM_MODELS_DIR / f"{self.symbol}.pth").write_bytes(b"dummy")

        with patch("scripts.daily_inference.lag_regression.load") as mock_lag_load,              patch("scripts.daily_inference.lag_regression.predict_next", return_value=101.0),              patch("scripts.daily_inference.arima_model.load") as mock_arima_load,              patch("scripts.daily_inference.arima_model.predict_next", return_value=102.0),              patch("scripts.daily_inference.lstm_model.load") as mock_lstm_load,              patch("scripts.daily_inference.lstm_model.predict_next", return_value=103.0):

            mock_arima_model = MagicMock()
            mock_arima_model.nobs = len(self.df)
            mock_arima_model.model.endog = self.df["Close"].values.reshape(-1, 1)
            mock_arima_load.return_value = mock_arima_model

            updated = daily_inference.infer_symbol(self.symbol, self.df)

        self.assertEqual(updated["next_close"]["lag"], 101.0)
        self.assertEqual(updated["next_close"]["arima"], 102.0)
        self.assertEqual(updated["next_close"]["lstm"], 103.0)

        self.assertIn("inference_metadata", updated)
        meta = updated["inference_metadata"]
        self.assertEqual(meta["models_retrained"], False)
        self.assertEqual(meta["model_source"], "weekly_persisted_artifacts")
        self.assertEqual(meta["data_as_of"], "2026-08-07")
        self.assertIn("forecast_for", meta)
        self.assertIn("inference_at", meta)

        self.assertIn("metrics", updated)
        self.assertIn("backtest30", updated)
        self.assertIn("backtest_by_model", updated)

    def test_infer_symbol_fails_on_missing_artifact(self):
        """If any model artifact is missing, infer_symbol must raise."""
        (daily_inference.LAG_MODELS_DIR / f"{self.symbol}.pkl").write_bytes(b"dummy")

        with patch("scripts.daily_inference.lag_regression.load") as mock_lag_load,              patch("scripts.daily_inference.lag_regression.predict_next", return_value=101.0):
            mock_lag_load.return_value = MagicMock()
            with self.assertRaises(FileNotFoundError) as ctx:
                daily_inference.infer_symbol(self.symbol, self.df)
            self.assertIn("ARIMA artifact missing", str(ctx.exception))

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------
    def test_run_daily_inference_batch(self):
        """run_daily_inference processes specified symbols and reports status."""
        for sym in ["A", "B"]:
            df = self.df.copy()
            df["Close"] = df["Close"] + np.random.randn() * 5
            # Recompute High/Low to maintain OHLC consistency
            df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1) - 0.01
            df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1) + 0.01
            csv = daily_inference.RAW_DIR / f"{sym}.csv"
            df.to_csv(csv, index=False)

            for d in (daily_inference.LAG_MODELS_DIR, daily_inference.ARIMA_MODELS_DIR):
                (d / f"{sym}.pkl").write_bytes(b"dummy")
            (daily_inference.LSTM_MODELS_DIR / f"{sym}.pth").write_bytes(b"dummy")

            cache = self.cache.copy()
            cache["next_close"] = {"lag": 100.0, "arima": 100.0, "lstm": 100.0}
            (daily_inference.PREDICTION_CACHE_DIR / f"{sym}.json").write_text(json.dumps(cache))

        def mock_arima_load(path):
            sym = Path(path).stem
            df_sym = pd.read_csv(daily_inference.RAW_DIR / f"{sym}.csv")
            m = MagicMock()
            m.nobs = len(df_sym)
            m.model.endog = df_sym["Close"].values.reshape(-1, 1)
            return m

        with patch("scripts.daily_inference.lag_regression.load"),              patch("scripts.daily_inference.lag_regression.predict_next", return_value=101.0),              patch("scripts.daily_inference.arima_model.load", side_effect=mock_arima_load),              patch("scripts.daily_inference.arima_model.predict_next", return_value=102.0),              patch("scripts.daily_inference.lstm_model.load"),              patch("scripts.daily_inference.lstm_model.predict_next", return_value=103.0):

            result = daily_inference.run_daily_inference(raw_dir=daily_inference.RAW_DIR, symbols=["A", "B"])

        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["symbols_processed"]), {"A", "B"})
        self.assertEqual(result["symbols_failed"], {})

        for sym in ["A", "B"]:
            cache = json.loads((daily_inference.PREDICTION_CACHE_DIR / f"{sym}.json").read_text())
            self.assertIn("inference_metadata", cache)
            self.assertEqual(cache["next_close"]["lag"], 101.0)

    def test_run_daily_inference_missing_csv(self):
        """Missing CSV should be reported as failure, not crash."""
        (daily_inference.PREDICTION_CACHE_DIR / "NOCSV.json").write_text(json.dumps(self.cache))
        result = daily_inference.run_daily_inference(symbols=["NOCSV"])
        self.assertIn("NOCSV", result["symbols_failed"])
        self.assertEqual(result["status"], "failure")

    # ------------------------------------------------------------------
    # Cache replacement / stale prevention
    # ------------------------------------------------------------------
    def test_fresh_cache_replaces_stale(self):
        """Daily inference must overwrite stale predictions, never keep old ones."""
        for d in (daily_inference.LAG_MODELS_DIR, daily_inference.ARIMA_MODELS_DIR):
            (d / f"{self.symbol}.pkl").write_bytes(b"dummy")
        (daily_inference.LSTM_MODELS_DIR / f"{self.symbol}.pth").write_bytes(b"dummy")

        stale_cache = self.cache.copy()
        stale_cache["next_close"] = {"lag": 50.0, "arima": 50.0, "lstm": 50.0}
        stale_cache["inference_metadata"] = {
            "data_as_of": "2026-08-01",
            "forecast_for": "2026-08-02",
            "inference_at": "2026-08-01T16:00:00+08:00",
            "models_retrained": False,
            "model_source": "weekly_persisted_artifacts",
        }
        (daily_inference.PREDICTION_CACHE_DIR / f"{self.symbol}.json").write_text(json.dumps(stale_cache))

        with patch("scripts.daily_inference.lag_regression.load"),              patch("scripts.daily_inference.lag_regression.predict_next", return_value=110.0),              patch("scripts.daily_inference.arima_model.load") as mock_arima_load,              patch("scripts.daily_inference.arima_model.predict_next", return_value=111.0),              patch("scripts.daily_inference.lstm_model.load"),              patch("scripts.daily_inference.lstm_model.predict_next", return_value=112.0):
            mock_arima_model = MagicMock()
            mock_arima_model.nobs = len(self.df)
            mock_arima_model.model.endog = self.df["Close"].values.reshape(-1, 1)
            mock_arima_load.return_value = mock_arima_model

            updated = daily_inference.infer_symbol(self.symbol, self.df)

        self.assertEqual(updated["next_close"]["lag"], 110.0)
        self.assertEqual(updated["next_close"]["arima"], 111.0)
        self.assertEqual(updated["next_close"]["lstm"], 112.0)
        self.assertEqual(updated["inference_metadata"]["data_as_of"], "2026-08-07")
        self.assertEqual(updated["inference_metadata"]["forecast_for"], "2026-08-10")

    # ------------------------------------------------------------------
    # PSE Calendar: Friday -> Monday / holiday handling
    # ------------------------------------------------------------------
    def test_next_trading_day_friday_to_monday(self):
        """Friday data should forecast for Monday, not Saturday."""
        calendar = PSECalendar()
        friday = date(2026, 8, 7)
        next_day = calendar.next_trading_day(friday)
        self.assertEqual(next_day, date(2026, 8, 10))

    def test_next_trading_day_thursday_to_friday(self):
        """Thursday data should forecast for Friday."""
        calendar = PSECalendar()
        thursday = date(2026, 8, 6)
        next_day = calendar.next_trading_day(thursday)
        self.assertEqual(next_day, date(2026, 8, 7))

    def test_next_trading_day_good_friday_holiday(self):
        """Friday before Good Friday -> next session after Easter."""
        calendar = PSECalendar()
        # 2025: Maundy Thursday = Apr 17, Good Friday = Apr 18
        # Wednesday Apr 16 -> Monday Apr 21
        wed_before = date(2025, 4, 16)
        next_day = calendar.next_trading_day(wed_before)
        self.assertEqual(next_day, date(2025, 4, 21))

    def test_next_trading_day_new_year(self):
        """New Year's Eve period -> next trading day after holidays."""
        calendar = PSECalendar()
        dec_29 = date(2025, 12, 29)
        next_day = calendar.next_trading_day(dec_29)
        # Dec 30 = Rizal Day (holiday), Dec 31 = special non-working, Jan 1 = holiday
        self.assertEqual(next_day, date(2026, 1, 2))

    # ------------------------------------------------------------------
    # Fast Pipeline ordering
    # ------------------------------------------------------------------
    def test_fast_pipeline_ordering(self):
        """Verify the Fast Pipeline workflow file has correct step ordering."""
        workflow_path = Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "update_pipeline.yml"
        self.assertTrue(workflow_path.exists(), "Workflow file must exist")
        content = workflow_path.read_text()

        # Verify no model training command
        self.assertNotIn("python -m services.model_selector", content)

        # Verify correct ordering: ingestion -> inference -> export
        ingest_pos = content.find("Run pipeline (ingestion")
        inference_pos = content.find("Run daily inference")
        export_pos = content.find("Export frontend JSON artifacts")

        self.assertGreater(ingest_pos, 0, "Ingestion step must exist")
        self.assertGreater(inference_pos, 0, "Inference step must exist")
        self.assertGreater(export_pos, 0, "Export step must exist")
        self.assertLess(ingest_pos, inference_pos, "Ingestion must come before inference")
        self.assertLess(inference_pos, export_pos, "Inference must come before export")

    # ------------------------------------------------------------------
    # No training confirmation
    # ------------------------------------------------------------------
    def test_no_training_in_daily_inference(self):
        """daily_inference.py must not import or call any training functions."""
        import ast
        source = Path(daily_inference.__file__).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr in ("fit", "train", "fit_transform"):
                        self.fail(f"daily_inference.py contains training call: {func.attr}")
                elif isinstance(func, ast.Name):
                    if func.id in ("fit", "train"):
                        self.fail(f"daily_inference.py contains training call: {func.id}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "model_selector" in alias.name or "train" in alias.name:
                        self.fail(f"daily_inference.py imports training module: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and ("model_selector" in node.module or "train" in node.module):
                    self.fail(f"daily_inference.py imports from training module: {node.module}")


if __name__ == "__main__":
    unittest.main()
