from __future__ import annotations

import unittest

import numpy as np

from tests.predictive_accuracy.runners import run_naive
from tests.predictive_accuracy.splits import chronological_split

from ._helpers import make_ohlcv


class TestNaiveBaseline(unittest.TestCase):
    def setUp(self):
        self.df = make_ohlcv(n=150)
        self.split = chronological_split(self.df, "TEST", final_test_fraction=0.15, min_trainval_rows=50, min_final_test_rows=5)

    def test_naive_prediction_equals_prior_close(self):
        run = run_naive(self.df, self.split)
        merged = self.df.set_index("Date")["Close"]
        for date, pred in zip(run.dates, run.y_pred):
            date_pos = self.df.index[self.df["Date"] == date][0]
            expected_prior_close = float(self.df["Close"].iloc[date_pos - 1])
            self.assertAlmostEqual(float(pred), expected_prior_close, places=8)

    def test_naive_actual_equals_same_day_close(self):
        run = run_naive(self.df, self.split)
        for date, actual in zip(run.dates, run.y_true):
            expected = float(self.df.loc[self.df["Date"] == date, "Close"].iloc[0])
            self.assertAlmostEqual(float(actual), expected, places=8)

    def test_naive_covers_every_final_test_date(self):
        run = run_naive(self.df, self.split)
        self.assertEqual(set(run.dates), set(self.split.final_test["Date"]))

    def test_naive_base_close_equals_prediction(self):
        run = run_naive(self.df, self.split)
        np.testing.assert_array_almost_equal(run.base_close, run.y_pred)

    def test_naive_selection_source_is_naive(self):
        run = run_naive(self.df, self.split)
        self.assertEqual(run.selection_source, "naive")
        self.assertFalse(run.fallback)


if __name__ == "__main__":
    unittest.main()
