"""End-to-end integration tests for the final-test evaluation suite.

These exercise the real per-model runners (services/forecasting/*.py,
unmodified) against small synthetic tickers, so they double as the
"feature alignment" and "identical final-test dates" checks the task
requires, plus a leakage smoke test and a reproducibility check.

ARIMA/LSTM automatically use their existing statsmodels/torch-unavailable
fallback paths when those optional dependencies aren't installed (see
services/forecasting/arima_model.py and lstm_model.py) — the suite's
runners mirror that same fallback deterministically (see runners.py), so
these tests pass in both environments without being skipped.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.predictive_accuracy.leakage_checks import LeakageError, assert_identical_test_dates
from tests.predictive_accuracy.runners import align_runs_to_common_dates, run_arima, run_lag_regression, run_lstm, run_naive
from tests.predictive_accuracy.splits import chronological_split

from ._helpers import make_ohlcv


class TestFeatureAlignment(unittest.TestCase):
    """Every model's predictions must be indexed to the correct calendar
    date (the date being predicted, not the date its input features were
    computed from) and must never reference a date beyond the final-test
    window."""

    def setUp(self):
        self.df = make_ohlcv(n=220, seed=11)
        self.split = chronological_split(self.df, "TEST", final_test_fraction=0.15, min_trainval_rows=100, min_final_test_rows=10)

    def test_lag_reg_dates_are_subset_of_final_test_dates(self):
        run = run_lag_regression(self.df, self.split)
        self.assertTrue(set(run.dates).issubset(set(self.split.final_test["Date"])))

    def test_lag_reg_base_close_precedes_predicted_date(self):
        run = run_lag_regression(self.df, self.split)
        close_by_date = self.df.set_index("Date")["Close"]
        for date, base in zip(run.dates, run.base_close):
            pos = self.df.index[self.df["Date"] == date][0]
            prior_actual_close = float(self.df["Close"].iloc[pos - 1])
            self.assertAlmostEqual(float(base), prior_actual_close, places=6)

    def test_naive_and_lag_reg_agree_on_base_close_for_shared_dates(self):
        naive_run = run_naive(self.df, self.split)
        lag_run = run_lag_regression(self.df, self.split)
        naive_by_date = dict(zip(naive_run.dates, naive_run.base_close))
        for date, base in zip(lag_run.dates, lag_run.base_close):
            if date in naive_by_date:
                self.assertAlmostEqual(float(base), float(naive_by_date[date]), places=6)


class TestIdenticalFinalTestDates(unittest.TestCase):
    def setUp(self):
        self.df = make_ohlcv(n=220, seed=12)
        self.split = chronological_split(self.df, "TEST", final_test_fraction=0.15, min_trainval_rows=100, min_final_test_rows=10)

    def test_align_runs_to_common_dates_produces_identical_dates(self):
        runs = {
            "naive": run_naive(self.df, self.split),
            "lag_reg": run_lag_regression(self.df, self.split),
            "arima": run_arima(self.df, self.split),
            "lstm": run_lstm(self.df, self.split),
        }
        aligned = align_runs_to_common_dates(runs)
        assert_identical_test_dates({k: r.dates for k, r in aligned.items()}, "TEST")  # must not raise

        lengths = {k: len(r.dates) for k, r in aligned.items()}
        self.assertGreater(min(lengths.values()), 0)
        self.assertEqual(len(set(lengths.values())), 1)  # all models cover exactly the same count

    def test_mismatched_dates_are_detected(self):
        run_a = run_naive(self.df, self.split)
        run_b = run_naive(self.df, self.split)
        run_b.dates = run_b.dates[:-1]  # artificially drop the last date
        with self.assertRaises(LeakageError):
            assert_identical_test_dates({"a": run_a.dates, "b": run_b.dates}, "TEST")


class TestNoLeakageEndToEnd(unittest.TestCase):
    """Structural leakage smoke test: none of the frozen artifacts' fitted
    parameters can have seen a final-test row, because they were produced
    by calling straight into services/forecasting/*.py with a dataframe
    that IS split.trainval — the final-test rows are simply not present in
    the object passed in."""

    def test_lag_regression_artifact_is_fit_only_on_trainval_rows(self):
        from services.forecasting import lag_regression

        df = make_ohlcv(n=200, seed=21)
        split = chronological_split(df, "TEST", final_test_fraction=0.15, min_trainval_rows=100, min_final_test_rows=10)

        artifact, _metrics, _next, _backtest, _test_actual, _test_pred = lag_regression.train(split.trainval)

        # The scaler's fitted mean_ vector has one entry per candidate
        # feature; its fit-time sample count (n_samples_seen_) must equal
        # the number of trainval rows with a complete feature row — never
        # more, which would mean a final-test row slipped in.
        n_seen = int(artifact.scaler.n_samples_seen_)
        from services.feature_engineering import build_full_features

        trainval_features = build_full_features(split.trainval).dropna()
        self.assertEqual(n_seen, len(trainval_features))
        self.assertLessEqual(n_seen, len(split.trainval))

    def test_run_arima_never_receives_final_test_rows(self):
        """A tripwire double of arima_model.train that raises if it is ever
        called with a dataframe containing a final-test date — proves the
        runner truly only ever passes TRAIN+VALIDATION through."""
        from services.forecasting import arima_model

        df = make_ohlcv(n=200, seed=22)
        split = chronological_split(df, "TEST", final_test_fraction=0.15, min_trainval_rows=100, min_final_test_rows=10)
        final_test_dates = set(split.final_test["Date"])

        original_train = arima_model.train

        def tripwire_train(passed_df):
            if set(passed_df["Date"]) & final_test_dates:
                raise AssertionError("arima_model.train received a final-test row!")
            return original_train(passed_df)

        arima_model.train = tripwire_train
        try:
            run_arima(df, split)  # must not raise
        finally:
            arima_model.train = original_train


class TestReproducibility(unittest.TestCase):
    def test_lag_regression_predictions_are_deterministic(self):
        df = make_ohlcv(n=200, seed=31)
        split = chronological_split(df, "TEST", final_test_fraction=0.15, min_trainval_rows=100, min_final_test_rows=10)

        run_a = run_lag_regression(df, split)
        run_b = run_lag_regression(df, split)

        np.testing.assert_array_almost_equal(run_a.y_pred, run_b.y_pred, decimal=8)
        self.assertEqual(list(run_a.dates), list(run_b.dates))

    def test_naive_baseline_is_deterministic(self):
        df = make_ohlcv(n=150, seed=32)
        split = chronological_split(df, "TEST", final_test_fraction=0.15, min_trainval_rows=80, min_final_test_rows=10)
        run_a = run_naive(df, split)
        run_b = run_naive(df, split)
        np.testing.assert_array_equal(run_a.y_pred, run_b.y_pred)

    def test_synthetic_data_generator_is_deterministic(self):
        df_a = make_ohlcv(n=100, seed=99)
        df_b = make_ohlcv(n=100, seed=99)
        pd.testing.assert_frame_equal(df_a, df_b)


if __name__ == "__main__":
    unittest.main()
