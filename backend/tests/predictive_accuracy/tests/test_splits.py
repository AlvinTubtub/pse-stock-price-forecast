from __future__ import annotations

import unittest

import pandas as pd

from tests.predictive_accuracy.splits import SplitError, chronological_split

from ._helpers import make_ohlcv


class TestChronologicalSplit(unittest.TestCase):
    def setUp(self):
        self.df = make_ohlcv(n=200)

    def test_fraction_based_split_sizes(self):
        split = chronological_split(self.df, "TEST", final_test_fraction=0.2, min_trainval_rows=50, min_final_test_rows=5)
        self.assertEqual(len(split.trainval) + len(split.final_test), len(self.df))
        self.assertAlmostEqual(len(split.final_test) / len(self.df), 0.2, delta=0.02)

    def test_trainval_strictly_before_final_test(self):
        split = chronological_split(self.df, "TEST", final_test_fraction=0.15, min_trainval_rows=50, min_final_test_rows=5)
        self.assertLess(split.trainval["Date"].max(), split.final_test["Date"].min())

    def test_no_row_lost_or_duplicated(self):
        split = chronological_split(self.df, "TEST", final_test_fraction=0.15, min_trainval_rows=50, min_final_test_rows=5)
        all_dates = pd.concat([split.trainval["Date"], split.final_test["Date"]])
        self.assertEqual(sorted(all_dates), sorted(self.df["Date"]))
        self.assertEqual(all_dates.duplicated().sum(), 0)

    def test_explicit_date_window(self):
        start = self.df["Date"].iloc[180].strftime("%Y-%m-%d")
        split = chronological_split(self.df, "TEST", final_test_start=start, min_trainval_rows=50, min_final_test_rows=5)
        self.assertTrue((split.final_test["Date"] >= pd.Timestamp(start)).all())
        self.assertTrue((split.trainval["Date"] < pd.Timestamp(start)).all())

    def test_explicit_date_window_with_end(self):
        start = self.df["Date"].iloc[180].strftime("%Y-%m-%d")
        end = self.df["Date"].iloc[190].strftime("%Y-%m-%d")
        split = chronological_split(self.df, "TEST", final_test_start=start, final_test_end=end, min_trainval_rows=50, min_final_test_rows=5)
        self.assertTrue((split.final_test["Date"] <= pd.Timestamp(end)).all())
        self.assertEqual(len(split.final_test), 11)

    def test_raises_when_trainval_too_small(self):
        with self.assertRaises(SplitError):
            chronological_split(self.df, "TEST", final_test_fraction=0.9, min_trainval_rows=100, min_final_test_rows=5)

    def test_raises_when_final_test_too_small(self):
        # Explicit window covering only the last 3 rows, but 20 required.
        start = self.df["Date"].iloc[-3].strftime("%Y-%m-%d")
        with self.assertRaises(SplitError):
            chronological_split(self.df, "TEST", final_test_start=start, min_trainval_rows=50, min_final_test_rows=20)

    def test_raises_on_unsorted_input(self):
        shuffled = self.df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        with self.assertRaises(SplitError):
            chronological_split(shuffled, "TEST", final_test_fraction=0.15, min_trainval_rows=50, min_final_test_rows=5)

    def test_raises_on_duplicate_dates(self):
        dup = pd.concat([self.df, self.df.iloc[[-1]]]).reset_index(drop=True)
        with self.assertRaises(SplitError):
            chronological_split(dup, "TEST", final_test_fraction=0.15, min_trainval_rows=50, min_final_test_rows=5)

    def test_end_before_start_raises(self):
        start = self.df["Date"].iloc[180].strftime("%Y-%m-%d")
        end = self.df["Date"].iloc[170].strftime("%Y-%m-%d")
        with self.assertRaises(SplitError):
            chronological_split(self.df, "TEST", final_test_start=start, final_test_end=end)


if __name__ == "__main__":
    unittest.main()
