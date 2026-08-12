from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # backend/

import numpy as np

from tests.predictive_accuracy.metrics import (
    bootstrap_ci,
    directional_accuracy,
    error_stats,
    evaluate_run,
    hit_rate_vs_naive,
    mean_directional_error,
)


class TestDirectionalMetrics(unittest.TestCase):
    def test_perfect_directional_accuracy(self):
        base = np.array([10.0, 10.0, 10.0, 10.0])
        actual = np.array([11.0, 9.0, 10.0, 12.0])  # up, down, flat, up
        pred = actual.copy()
        self.assertEqual(directional_accuracy(actual, pred, base), 1.0)

    def test_zero_directional_accuracy_when_always_opposite(self):
        base = np.array([10.0, 10.0])
        actual = np.array([11.0, 9.0])  # up, down
        pred = np.array([9.0, 11.0])  # down, up
        self.assertEqual(directional_accuracy(actual, pred, base), 0.0)

    def test_mean_directional_error_zero_when_unbiased(self):
        base = np.array([10.0, 10.0])
        actual = np.array([11.0, 9.0])
        pred = actual.copy()
        self.assertEqual(mean_directional_error(actual, pred, base), 0.0)

    def test_mean_directional_error_positive_when_systematically_bullish(self):
        base = np.array([10.0, 10.0])
        actual = np.array([9.0, 9.0])  # both down
        pred = np.array([11.0, 11.0])  # both predicted up
        self.assertEqual(mean_directional_error(actual, pred, base), 2.0)

    def test_flat_direction_uses_epsilon(self):
        base = np.array([10.0])
        actual = np.array([10.0 + 1e-12])  # effectively flat
        pred = np.array([10.0])
        self.assertEqual(directional_accuracy(actual, pred, base), 1.0)


class TestHitRate(unittest.TestCase):
    def test_hit_rate_all_beat_naive(self):
        abs_errors = np.array([0.1, 0.2, 0.05])
        naive_errors = np.array([0.5, 0.5, 0.5])
        self.assertEqual(hit_rate_vs_naive(abs_errors, naive_errors), 1.0)

    def test_hit_rate_none_beat_naive(self):
        abs_errors = np.array([1.0, 1.0])
        naive_errors = np.array([0.1, 0.1])
        self.assertEqual(hit_rate_vs_naive(abs_errors, naive_errors), 0.0)

    def test_hit_rate_requires_equal_length(self):
        with self.assertRaises(ValueError):
            hit_rate_vs_naive(np.array([1.0, 2.0]), np.array([1.0]))

    def test_hit_rate_ties_count_as_hits(self):
        self.assertEqual(hit_rate_vs_naive(np.array([0.5]), np.array([0.5])), 1.0)


class TestErrorStats(unittest.TestCase):
    def test_bias_and_spread(self):
        y_true = np.array([10.0, 10.0, 10.0])
        y_pred = np.array([9.0, 10.0, 11.0])
        stats = error_stats(y_true, y_pred)
        self.assertAlmostEqual(stats["bias_mean_error"], 0.0)
        self.assertAlmostEqual(stats["min_error"], -1.0)
        self.assertAlmostEqual(stats["max_error"], 1.0)


class TestBootstrapCI(unittest.TestCase):
    def test_omitted_for_too_few_observations(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 2.1])
        self.assertIsNone(bootstrap_ci(y_true, y_pred))

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(100, 5, 50)
        y_pred = y_true + rng.normal(0, 1, 50)
        ci = bootstrap_ci(y_true, y_pred, n_iterations=500)
        self.assertIsNotNone(ci)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        self.assertLessEqual(ci["rmse_ci"][0], rmse + 1e-6)
        self.assertGreaterEqual(ci["rmse_ci"][1], rmse - 1e-6)

    def test_ci_is_deterministic_given_seed(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(50, 3, 30)
        y_pred = y_true + rng.normal(0, 1, 30)
        ci_a = bootstrap_ci(y_true, y_pred, n_iterations=300, seed=7)
        ci_b = bootstrap_ci(y_true, y_pred, n_iterations=300, seed=7)
        self.assertEqual(ci_a, ci_b)


class TestEvaluateRun(unittest.TestCase):
    def test_evaluate_run_shape(self):
        rng = np.random.default_rng(2)
        y_true = rng.normal(50, 2, 40)
        y_pred = y_true + rng.normal(0, 0.5, 40)
        base_close = y_true - rng.normal(0, 1, 40)
        y_train = rng.normal(50, 2, 200)
        naive_abs_errors = np.abs(rng.normal(0, 1, 40))

        result = evaluate_run(y_true, y_pred, base_close, y_train, naive_abs_errors=naive_abs_errors)
        for key in ("rmse", "mae", "mase", "r2", "directional_accuracy", "mean_directional_error", "error_stats", "hit_rate_vs_naive"):
            self.assertIn(key, result)
        self.assertEqual(result["n_observations"], 40)


if __name__ == "__main__":
    unittest.main()
