from __future__ import annotations

import unittest

import numpy as np

from tests.predictive_accuracy.runners import ModelRun
from tests.predictive_accuracy.statistical_tests import (
    aggregate_ranking,
    model_selection_frequency,
    naive_baseline_comparison,
    per_ticker_diebold_mariano,
    per_ticker_ranking,
    rmse_table,
    run_full_statistical_suite,
)


def _run(model_key, ticker, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return ModelRun(
        model_key=model_key, ticker=ticker, dates=list(range(len(y_true))),
        y_true=y_true, y_pred=y_pred, base_close=y_true, y_train_reference=y_true,
        selection_source="trainval" if model_key != "naive" else "naive",
    )


def _metrics_for(model_key, rmse):
    return {"rmse": f"{rmse:.4f}", "mae": f"{rmse:.4f}", "mase": "1.0", "r2": "0.5", "n_observations": 20}


class TestRankingHelpers(unittest.TestCase):
    def setUp(self):
        self.model_keys = ("naive", "lag_reg", "arima", "lstm")
        self.metrics_by_ticker = {
            "AAA": {"naive": _metrics_for("naive", 2.0), "lag_reg": _metrics_for("lag_reg", 1.5), "arima": _metrics_for("arima", 3.0), "lstm": _metrics_for("lstm", 1.8)},
            "BBB": {"naive": _metrics_for("naive", 2.2), "lag_reg": _metrics_for("lag_reg", 1.6), "arima": _metrics_for("arima", 2.9), "lstm": _metrics_for("lstm", 2.5)},
            "CCC": {"naive": _metrics_for("naive", 1.9), "lag_reg": _metrics_for("lag_reg", 2.1), "arima": _metrics_for("arima", 3.5), "lstm": _metrics_for("lstm", 1.7)},
        }

    def test_rmse_table_shape(self):
        table = rmse_table(self.metrics_by_ticker, self.model_keys)
        for key in self.model_keys:
            self.assertEqual(len(table[key]), 3)

    def test_per_ticker_ranking_orders_by_rmse(self):
        ranking = per_ticker_ranking(self.metrics_by_ticker, self.model_keys)
        self.assertEqual(ranking["AAA"][0], "lag_reg")  # lowest RMSE (1.5)
        self.assertEqual(ranking["AAA"][-1], "arima")  # highest RMSE (3.0)

    def test_aggregate_ranking_orders_by_mean_rmse(self):
        ranking = aggregate_ranking(self.metrics_by_ticker, self.model_keys)
        self.assertEqual(ranking[0]["model"], "lag_reg")
        self.assertEqual(ranking[0]["rank"], 1)
        self.assertEqual([e["rank"] for e in ranking], sorted(e["rank"] for e in ranking))

    def test_model_selection_frequency_sums_to_n_tickers(self):
        freq = model_selection_frequency(self.metrics_by_ticker, ("lag_reg", "arima", "lstm"))
        self.assertEqual(sum(freq["counts"].values()), 3)
        self.assertAlmostEqual(sum(freq["fraction"].values()), 1.0)

    def test_naive_baseline_comparison_flags_wins(self):
        cmp = naive_baseline_comparison(self.metrics_by_ticker, ("lag_reg", "arima", "lstm"))
        self.assertEqual(cmp["lag_reg"]["tickers_beating_naive"], 2)  # AAA, BBB (not CCC)
        self.assertEqual(cmp["arima"]["tickers_beating_naive"], 0)


class TestDieboldMariano(unittest.TestCase):
    def test_per_ticker_dm_pairs_and_holm_adjustment(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(50, 2, 60)
        runs_by_ticker = {
            "AAA": {
                "naive": _run("naive", "AAA", y_true, y_true + rng.normal(0, 2, 60)),
                "lag_reg": _run("lag_reg", "AAA", y_true, y_true + rng.normal(0, 0.5, 60)),
            }
        }
        result = per_ticker_diebold_mariano(runs_by_ticker)
        self.assertIn("AAA", result)
        self.assertIn("lag_reg vs naive", result["AAA"])
        entry = result["AAA"]["lag_reg vs naive"]
        self.assertIn("dm_statistic", entry)
        self.assertIn("p_value", entry)
        self.assertIn("holm_p_value", entry)
        self.assertGreaterEqual(entry["holm_p_value"], entry["p_value"] - 1e-9)


class TestFullSuiteInputValidation(unittest.TestCase):
    def test_full_suite_runs_on_minimal_valid_input(self):
        rng = np.random.default_rng(1)
        y_true = rng.normal(50, 2, 40)
        tickers = ["AAA", "BBB", "CCC"]
        model_keys = ("naive", "lag_reg", "arima")
        runs_by_ticker, metrics_by_ticker = {}, {}
        for t in tickers:
            runs_by_ticker[t] = {
                mk: _run(mk, t, y_true, y_true + rng.normal(0, 1 + i, 40)) for i, mk in enumerate(model_keys)
            }
            metrics_by_ticker[t] = {
                mk: _metrics_for(mk, float(np.sqrt(np.mean((runs_by_ticker[t][mk].y_true - runs_by_ticker[t][mk].y_pred) ** 2))))
                for mk in model_keys
            }
            for mk in model_keys:
                metrics_by_ticker[t][mk]["n_observations"] = 40

        stats = run_full_statistical_suite(runs_by_ticker, metrics_by_ticker, model_keys, min_consistency_tickers=2)
        for key in ("diebold_mariano_per_ticker", "per_ticker_ranking", "aggregate_ranking", "model_selection_frequency", "naive_baseline_comparison", "friedman", "wilcoxon_holm_posthoc", "best_model_consistency"):
            self.assertIn(key, stats)
        self.assertEqual(stats["n_tickers"], 3)


if __name__ == "__main__":
    unittest.main()
