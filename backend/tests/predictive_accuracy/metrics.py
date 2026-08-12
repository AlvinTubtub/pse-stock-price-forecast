"""Metrics for the final-test evaluation, built on top of
``services.evaluation.compute_metrics`` (RMSE/MAE/MASE/R²) rather than
re-implementing them, plus the extra metrics this suite requires.

Definitions used below (documented here so results are unambiguous):

``directional_accuracy``
    Fraction of final-test days where the predicted direction of change
    (up/flat/down, relative to the actual close on the immediately
    preceding trading day — the same reference the naive baseline uses)
    matches the actual direction of change.

``mean_directional_error`` (MDE)
    Mean of (predicted_direction - actual_direction), each encoded as
    {+1, 0, -1}. Ranges over [-2, 2]; 0 means no systematic directional
    bias, positive means the model is systematically more bullish than
    reality, negative more bearish.

``hit_rate``
    Fraction of final-test days where the model's absolute error is less
    than or equal to the naive baseline's absolute error on that same day
    — i.e. how often the model "hits" (matches or beats) the naive
    benchmark, date for date. Requires the naive baseline's per-date
    absolute errors on the *same* dates (see ``evaluate_run``).

``error_stats``
    Bias (mean signed error, actual - predicted), std of the error,
    min/max error, and median absolute error.

``confidence_interval_95``
    Percentile bootstrap (resampling the per-date errors with
    replacement) 95% CI for RMSE and MAE. Omitted (not fabricated) when
    there are too few final-test observations to bootstrap meaningfully.
"""
from __future__ import annotations

import numpy as np

from services.evaluation import compute_metrics

from . import config


def _direction(delta: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """+1 up, -1 down, 0 flat (within a tiny epsilon of zero, to avoid
    floating-point noise being called a 'direction')."""
    return np.where(delta > eps, 1, np.where(delta < -eps, -1, 0))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, base_close: np.ndarray) -> float:
    actual_dir = _direction(y_true - base_close)
    pred_dir = _direction(y_pred - base_close)
    return float(np.mean(actual_dir == pred_dir))


def mean_directional_error(y_true: np.ndarray, y_pred: np.ndarray, base_close: np.ndarray) -> float:
    actual_dir = _direction(y_true - base_close)
    pred_dir = _direction(y_pred - base_close)
    return float(np.mean(pred_dir - actual_dir))


def hit_rate_vs_naive(abs_errors: np.ndarray, naive_abs_errors: np.ndarray) -> float:
    if len(abs_errors) != len(naive_abs_errors):
        raise ValueError(
            f"hit_rate_vs_naive requires date-aligned arrays of equal length "
            f"(got {len(abs_errors)} vs {len(naive_abs_errors)})."
        )
    return float(np.mean(abs_errors <= naive_abs_errors))


def error_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_true - y_pred
    return {
        "bias_mean_error": float(np.mean(error)),
        "std_error": float(np.std(error, ddof=1)) if len(error) > 1 else 0.0,
        "min_error": float(np.min(error)),
        "max_error": float(np.max(error)),
        "median_abs_error": float(np.median(np.abs(error))),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_iterations: int = config.BOOTSTRAP_ITERATIONS,
    confidence: float = config.CONFIDENCE_LEVEL,
    seed: int = config.RANDOM_SEED,
) -> dict | None:
    """Percentile bootstrap CI for RMSE and MAE over the final-test errors.

    Returns ``None`` (rather than a misleadingly narrow interval) when
    there are fewer than ``config.MIN_ROWS_FOR_CI`` observations.
    """
    n = len(y_true)
    if n < config.MIN_ROWS_FOR_CI:
        return None

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmses = np.empty(n_iterations)
    maes = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        err = y_true[idx] - y_pred[idx]
        rmses[i] = np.sqrt(np.mean(err**2))
        maes[i] = np.mean(np.abs(err))

    alpha = 1.0 - confidence
    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "confidence_level": confidence,
        "n_iterations": n_iterations,
        "rmse_ci": [float(np.percentile(rmses, lo_pct)), float(np.percentile(rmses, hi_pct))],
        "mae_ci": [float(np.percentile(maes, lo_pct)), float(np.percentile(maes, hi_pct))],
    }


def evaluate_run(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base_close: np.ndarray,
    y_train_reference: np.ndarray,
    naive_abs_errors: np.ndarray | None = None,
) -> dict:
    """Full metrics dict for one model's predictions on the final-test
    window: RMSE/MAE/MASE/R² (reused from services.evaluation) plus every
    extra metric this suite requires."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    base_close = np.asarray(base_close, dtype=float)

    metrics = compute_metrics(y_true, y_pred, y_train=y_train_reference)
    abs_errors = np.abs(y_true - y_pred)

    metrics["n_observations"] = int(len(y_true))
    metrics["directional_accuracy"] = directional_accuracy(y_true, y_pred, base_close)
    metrics["mean_directional_error"] = mean_directional_error(y_true, y_pred, base_close)
    metrics["error_stats"] = error_stats(y_true, y_pred)
    metrics["confidence_interval_95"] = bootstrap_ci(y_true, y_pred)

    if naive_abs_errors is not None:
        metrics["hit_rate_vs_naive"] = hit_rate_vs_naive(abs_errors, np.asarray(naive_abs_errors, dtype=float))

    return metrics
