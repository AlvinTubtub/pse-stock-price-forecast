from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.predictive_accuracy.leakage_checks import (
    LeakageError,
    assert_fit_input_excludes_dates,
    assert_identical_test_dates,
    assert_model_selection_before_final_test,
    assert_naive_uses_prior_close_only,
    assert_no_date_overlap,
    assert_no_future_values_in_features,
    assert_scaler_fit_row_count,
)

from ._helpers import make_ohlcv


class TestLeakageChecks(unittest.TestCase):
    def setUp(self):
        self.df = make_ohlcv(n=100)

    def test_no_date_overlap_passes_for_clean_split(self):
        trainval, final_test = self.df.iloc[:80], self.df.iloc[80:]
        assert_no_date_overlap(trainval, final_test, "TEST")  # should not raise

    def test_no_date_overlap_detects_overlap(self):
        trainval, final_test = self.df.iloc[:85], self.df.iloc[80:]
        with self.assertRaises(LeakageError):
            assert_no_date_overlap(trainval, final_test, "TEST")

    def test_no_date_overlap_detects_out_of_order(self):
        trainval, final_test = self.df.iloc[80:], self.df.iloc[:80]
        with self.assertRaises(LeakageError):
            assert_no_date_overlap(trainval, final_test, "TEST")

    def test_fit_input_excludes_dates_passes(self):
        fit_dates = self.df["Date"].iloc[:80]
        forbidden = self.df["Date"].iloc[80:]
        assert_fit_input_excludes_dates(fit_dates, forbidden, "TEST", "scaler")

    def test_fit_input_excludes_dates_detects_leak(self):
        fit_dates = self.df["Date"].iloc[:85]  # includes 5 "forbidden" dates
        forbidden = self.df["Date"].iloc[80:]
        with self.assertRaises(LeakageError):
            assert_fit_input_excludes_dates(fit_dates, forbidden, "TEST", "scaler")

    def test_identical_test_dates_passes(self):
        dates = list(self.df["Date"].iloc[80:])
        assert_identical_test_dates({"a": dates, "b": list(dates)}, "TEST")

    def test_identical_test_dates_detects_mismatch(self):
        dates_a = list(self.df["Date"].iloc[80:])
        dates_b = list(self.df["Date"].iloc[81:])  # one date short
        with self.assertRaises(LeakageError):
            assert_identical_test_dates({"a": dates_a, "b": dates_b}, "TEST")

    def test_identical_test_dates_single_model_is_trivially_ok(self):
        assert_identical_test_dates({"a": [1, 2, 3]}, "TEST")

    def test_naive_uses_prior_close_only_passes(self):
        prior_close = np.array([10.0, 11.0, 12.0])
        assert_naive_uses_prior_close_only(prior_close, prior_close, "TEST")

    def test_naive_uses_prior_close_only_detects_deviation(self):
        prior_close = np.array([10.0, 11.0, 12.0])
        drifted = prior_close + 0.5
        with self.assertRaises(LeakageError):
            assert_naive_uses_prior_close_only(drifted, prior_close, "TEST")

    def test_naive_uses_prior_close_only_detects_shape_mismatch(self):
        with self.assertRaises(LeakageError):
            assert_naive_uses_prior_close_only(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), "TEST")

    def test_model_selection_before_final_test_passes(self):
        assert_model_selection_before_final_test("trainval", "TEST", "lag_reg")
        assert_model_selection_before_final_test("naive", "TEST", "naive")

    def test_model_selection_before_final_test_detects_bad_source(self):
        with self.assertRaises(LeakageError):
            assert_model_selection_before_final_test("final_test", "TEST", "lag_reg")

    def test_scaler_fit_row_count_passes(self):
        assert_scaler_fit_row_count(80, 80, "TEST", "StandardScaler")

    def test_scaler_fit_row_count_detects_mismatch(self):
        with self.assertRaises(LeakageError):
            assert_scaler_fit_row_count(85, 80, "TEST", "StandardScaler")

    def test_no_future_values_in_features_passes(self):
        window_dates = self.df["Date"].iloc[10:15]
        target_date = self.df["Date"].iloc[15]
        assert_no_future_values_in_features(window_dates, target_date, "TEST")

    def test_no_future_values_in_features_detects_future_leak(self):
        window_dates = self.df["Date"].iloc[10:16]  # includes the target date itself
        target_date = self.df["Date"].iloc[15]
        with self.assertRaises(LeakageError):
            assert_no_future_values_in_features(window_dates, target_date, "TEST")


if __name__ == "__main__":
    unittest.main()
