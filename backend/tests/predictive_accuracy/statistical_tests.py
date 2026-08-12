"""Cross-model, cross-ticker statistical-significance suite for the final
unseen-test evaluation.

Reuses ``services.evaluation``'s Diebold-Mariano/HLN, Holm-Bonferroni,
Friedman, and Wilcoxon implementations unmodified (the same functions the
production pipeline already uses in ``services/model_selector.py``) rather
than re-implementing them, so there is exactly one statistical-testing
implementation in the repository.

Adds what the production pipeline's statistical suite doesn't need but this
evaluation does:

  - Per-ticker, per-model-pair Diebold-Mariano tests computed directly from
    each ``ModelRun``'s final-test errors (including each model vs the
    naive baseline, not just model-vs-model).
  - Aggregate + per-ticker model rankings (by RMSE).
  - Model-selection frequency (how often each model was the per-ticker
    RMSE winner).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from services.evaluation import (
    diebold_mariano_test,
    friedman_test,
    holm_correction,
    holm_wilcoxon_posthoc,
    best_model_consistency_check,
)

from .runners import ModelRun


def _errors(run: ModelRun) -> np.ndarray:
    return run.y_true - run.y_pred


def per_ticker_diebold_mariano(runs_by_ticker: dict[str, dict[str, ModelRun]]) -> dict:
    """Pairwise DM tests (including vs. naive) within each ticker, on that
    ticker's aligned final-test dates, Holm-corrected across its pairs."""
    result = {}
    for ticker, runs in runs_by_ticker.items():
        keys = sorted(runs.keys())
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]
        raw = []
        for a, b in pairs:
            dm_stat, p_value = diebold_mariano_test(_errors(runs[a]), _errors(runs[b]))
            raw.append((a, b, dm_stat, p_value))
        adjusted = holm_correction([p for *_, p in raw])
        result[ticker] = {
            f"{a} vs {b}": {"dm_statistic": stat, "p_value": p, "holm_p_value": holm_p}
            for (a, b, stat, p), holm_p in zip(raw, adjusted)
        }
    return result


def rmse_table(metrics_by_ticker: dict[str, dict[str, dict]], model_keys: tuple[str, ...]) -> dict[str, list[float]]:
    """{"lag_reg": [rmse_ticker1, rmse_ticker2, ...], ...}, paired by
    ticker (same order for every model)."""
    tickers = sorted(metrics_by_ticker.keys())
    return {
        model: [float(metrics_by_ticker[t][model]["rmse"]) for t in tickers]
        for model in model_keys
    }


def per_ticker_ranking(metrics_by_ticker: dict[str, dict[str, dict]], model_keys: tuple[str, ...]) -> dict[str, list[str]]:
    """{ticker: [best_model, ..., worst_model]}, ranked by RMSE ascending."""
    return {
        ticker: sorted(model_keys, key=lambda m: float(metrics_by_ticker[ticker][m]["rmse"]))
        for ticker in metrics_by_ticker
    }


def aggregate_ranking(metrics_by_ticker: dict[str, dict[str, dict]], model_keys: tuple[str, ...]) -> list[dict]:
    """Models ranked by mean RMSE across every ticker (ascending)."""
    means = {
        model: float(np.mean([float(metrics_by_ticker[t][model]["rmse"]) for t in metrics_by_ticker]))
        for model in model_keys
    }
    ranked = sorted(means.items(), key=lambda kv: kv[1])
    return [{"model": model, "mean_rmse": mean_rmse, "rank": i + 1} for i, (model, mean_rmse) in enumerate(ranked)]


def model_selection_frequency(metrics_by_ticker: dict[str, dict[str, dict]], model_keys: tuple[str, ...]) -> dict:
    """How often each model had the lowest RMSE across tickers (the
    'selected' model per ticker, if selection were done on the final test
    — reported for descriptive purposes only; real model selection in
    this suite happens on TRAIN+VALIDATION, never on the final test)."""
    winners = [
        min(model_keys, key=lambda m: float(metrics_by_ticker[t][m]["rmse"]))
        for t in metrics_by_ticker
    ]
    counts = Counter(winners)
    n = len(winners)
    return {
        "counts": {m: counts.get(m, 0) for m in model_keys},
        "fraction": {m: counts.get(m, 0) / n if n else 0.0 for m in model_keys},
        "n_tickers": n,
    }


def naive_baseline_comparison(metrics_by_ticker: dict[str, dict[str, dict]], model_keys: tuple[str, ...]) -> dict:
    """Per-model, per-ticker RMSE improvement (%) over the naive baseline,
    plus the fraction of tickers where each model beat naive."""
    result = {}
    for model in model_keys:
        improvements, wins = [], 0
        for ticker, metrics in metrics_by_ticker.items():
            model_rmse = float(metrics[model]["rmse"])
            naive_rmse = float(metrics["naive"]["rmse"])
            improvement_pct = 100.0 * (naive_rmse - model_rmse) / naive_rmse if naive_rmse else float("nan")
            improvements.append(improvement_pct)
            wins += int(model_rmse < naive_rmse)
        n = len(metrics_by_ticker)
        result[model] = {
            "mean_rmse_improvement_pct": float(np.mean(improvements)) if improvements else float("nan"),
            "tickers_beating_naive": wins,
            "n_tickers": n,
            "fraction_beating_naive": wins / n if n else 0.0,
        }
    return result


def run_full_statistical_suite(
    runs_by_ticker: dict[str, dict[str, ModelRun]],
    metrics_by_ticker: dict[str, dict[str, dict]],
    model_keys: tuple[str, ...],
    min_consistency_tickers: int,
) -> dict:
    """Everything the suite's README promises: per-ticker DM, aggregate +
    per-ticker rankings, model-selection frequency, naive comparison,
    Friedman test, Holm-adjusted pairwise Wilcoxon, and best-model
    consistency — all computed strictly from the frozen final-test
    predictions."""
    tuned_keys = tuple(k for k in model_keys if k != "naive")
    rmse_by_model = rmse_table(metrics_by_ticker, model_keys)
    rmse_by_model_tuned = rmse_table(metrics_by_ticker, tuned_keys)

    mase_by_model = {
        model: [
            float(metrics_by_ticker[t][model]["mase"])
            for t in sorted(metrics_by_ticker.keys())
        ]
        for model in model_keys
    }

    mase_by_model_tuned = {
    model: [
        float(metrics_by_ticker[t][model]["mase"])
        for t in sorted(metrics_by_ticker.keys())
    ]
    for model in tuned_keys
}

    return {
        "n_tickers": len(metrics_by_ticker),
        "tickers": sorted(metrics_by_ticker.keys()),
        "diebold_mariano_per_ticker": per_ticker_diebold_mariano(runs_by_ticker),
        "per_ticker_ranking": per_ticker_ranking(metrics_by_ticker, model_keys),
        "aggregate_ranking": aggregate_ranking(metrics_by_ticker, model_keys),
        "model_selection_frequency": model_selection_frequency(metrics_by_ticker, tuned_keys),
        "naive_baseline_comparison": naive_baseline_comparison(metrics_by_ticker, tuned_keys),
        "friedman": friedman_test(mase_by_model_tuned),
        "wilcoxon_holm_posthoc": (
            holm_wilcoxon_posthoc(mase_by_model_tuned)
            if metrics_by_ticker
            else {}
        ),
        "best_model_consistency": best_model_consistency_check(rmse_by_model_tuned, min_companies=min_consistency_tickers),
        "friedman_including_naive": friedman_test(mase_by_model),
    }
