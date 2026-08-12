"""Run the final unseen-test predictive-accuracy evaluation for every
configured ticker and model, and write machine-readable results.

Usage (from backend/):

    python -m tests.predictive_accuracy.run_evaluation
    PA_TICKERS=BPI,ALI python -m tests.predictive_accuracy.run_evaluation      # fast subset
    PA_FINAL_TEST_FRACTION=0.10 python -m tests.predictive_accuracy.run_evaluation

See config.py for every environment-variable override, and README.md for
methodology, output-file documentation, and full command reference.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # backend/, for `services.*`

from services.data_validator import CSVValidationError, validate_ohlcv_csv

from tests.predictive_accuracy import config
from tests.predictive_accuracy.leakage_checks import (
    LeakageError,
    assert_fit_input_excludes_dates,
    assert_identical_test_dates,
    assert_model_selection_before_final_test,
    assert_no_date_overlap,
)
from tests.predictive_accuracy.metrics import evaluate_run
from tests.predictive_accuracy.runners import RUNNERS, align_runs_to_common_dates
from tests.predictive_accuracy.splits import SplitError, chronological_split
from tests.predictive_accuracy.statistical_tests import run_full_statistical_suite

log = logging.getLogger("predictive_accuracy")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _load_ticker(ticker: str) -> pd.DataFrame:
    csv_path = config.RAW_DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found for ticker {ticker!r} at {csv_path}.")
    return validate_ohlcv_csv(csv_path)


def evaluate_ticker(ticker: str) -> tuple[dict, dict]:
    """Runs every model + naive for one ticker. Returns
    (runs_by_model_key, metrics_by_model_key)."""
    df = _load_ticker(ticker)

    split = chronological_split(
        df,
        ticker,
        final_test_fraction=config.FINAL_TEST_FRACTION,
        final_test_start=config.FINAL_TEST_START_DATE,
        final_test_end=config.FINAL_TEST_END_DATE,
        min_trainval_rows=config.MIN_TRAINVAL_ROWS,
        min_final_test_rows=config.MIN_FINAL_TEST_ROWS,
    )
    assert_no_date_overlap(split.trainval, split.final_test, ticker)
    assert_fit_input_excludes_dates(split.trainval["Date"], split.final_test["Date"], ticker, "TRAIN+VALIDATION split")

    runs = {}
    for model_key in config.ALL_MODEL_KEYS:
        t0 = time.monotonic()
        run = RUNNERS[model_key](df, split)
        assert_model_selection_before_final_test(run.selection_source, ticker, model_key)
        runs[model_key] = run
        log.info(
            "[%s/%s] %d final-test predictions in %.1fs%s",
            ticker, model_key, len(run.dates), time.monotonic() - t0,
            " (fallback)" if run.fallback else "",
        )

    runs = align_runs_to_common_dates(runs)
    assert_identical_test_dates({k: r.dates for k, r in runs.items()}, ticker)

    naive_abs_errors = np.abs(runs["naive"].y_true - runs["naive"].y_pred)
    metrics = {}
    for model_key, run in runs.items():
        metrics[model_key] = evaluate_run(
            run.y_true, run.y_pred, run.base_close, run.y_train_reference,
            naive_abs_errors=naive_abs_errors if model_key != "naive" else None,
        )
        metrics[model_key]["fallback"] = run.fallback
        metrics[model_key]["extra"] = run.extra

    return runs, metrics


def _metrics_csv_rows(metrics_by_ticker: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for ticker, by_model in metrics_by_ticker.items():
        for model_key, m in by_model.items():
            rows.append({
                "ticker": ticker,
                "model": config.MODEL_LABELS.get(model_key, model_key),
                "model_key": model_key,
                "n_observations": m["n_observations"],
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "mase": float(m["mase"]),
                "r2": float(m["r2"]),
                "directional_accuracy": m["directional_accuracy"],
                "mean_directional_error": m["mean_directional_error"],
                "hit_rate_vs_naive": m.get("hit_rate_vs_naive"),
                "bias_mean_error": m["error_stats"]["bias_mean_error"],
                "std_error": m["error_stats"]["std_error"],
                "median_abs_error": m["error_stats"]["median_abs_error"],
                "rmse_ci_95_low": (m["confidence_interval_95"] or {}).get("rmse_ci", [None, None])[0],
                "rmse_ci_95_high": (m["confidence_interval_95"] or {}).get("rmse_ci", [None, None])[1],
                "fallback": m["fallback"],
            })
    return rows


def _model_comparison_csv_rows(stats: dict) -> list[dict]:
    rows = []
    for entry in stats["aggregate_ranking"]:
        model_key = entry["model"]
        selection = stats["model_selection_frequency"]
        naive_cmp = stats["naive_baseline_comparison"].get(model_key, {})
        rows.append({
            "model": config.MODEL_LABELS.get(model_key, model_key),
            "model_key": model_key,
            "rank": entry["rank"],
            "mean_rmse": entry["mean_rmse"],
            "selection_frequency": selection["fraction"].get(model_key),
            "selection_count": selection["counts"].get(model_key),
            "mean_rmse_improvement_pct_vs_naive": naive_cmp.get("mean_rmse_improvement_pct"),
            "fraction_beating_naive": naive_cmp.get("fraction_beating_naive"),
        })
    return rows


def _write_summary_md(path: Path, metrics_by_ticker: dict, stats: dict, tickers: list[str]) -> None:
    lines = [
        "# Final Unseen-Test Evaluation Summary",
        "",
        f"- Tickers evaluated: {len(tickers)} ({', '.join(tickers)})",
        f"- Final-test fraction: {config.FINAL_TEST_FRACTION}" if not config.FINAL_TEST_START_DATE else
        f"- Final-test window: {config.FINAL_TEST_START_DATE} to {config.FINAL_TEST_END_DATE or 'latest'}",
        f"- Random seed: {config.RANDOM_SEED}",
        "",
        "## Aggregate ranking (mean RMSE across tickers, ascending)",
        "",
        "| Rank | Model | Mean RMSE | Selection frequency | Mean RMSE improvement vs naive |",
        "|---|---|---|---|---|",
    ]
    selection = stats["model_selection_frequency"]
    naive_cmp = stats["naive_baseline_comparison"]
    for entry in stats["aggregate_ranking"]:
        m = entry["model"]
        freq = selection["fraction"].get(m)
        improvement = naive_cmp.get(m, {}).get("mean_rmse_improvement_pct")
        lines.append(
            f"| {entry['rank']} | {config.MODEL_LABELS.get(m, m)} | {entry['mean_rmse']:.4f} | "
            f"{freq:.0%} | {improvement:+.2f}%" if freq is not None and improvement is not None else
            f"| {entry['rank']} | {config.MODEL_LABELS.get(m, m)} | {entry['mean_rmse']:.4f} | - | - |"
        )

    friedman = stats["friedman"]
    lines += [
        "",
        "## Statistical significance",
        "",
        f"- Friedman test across tickers (tuned models only): "
        f"statistic={friedman['statistic']:.4f}, p={friedman['p_value']:.4g}, "
        f"n_tickers={friedman['n_companies']}",
        f"- Best-model consistency: {stats['best_model_consistency']['dominant_model']} "
        f"is lowest-RMSE on {stats['best_model_consistency']['dominant_count']}/"
        f"{stats['best_model_consistency']['total_companies']} tickers "
        f"(pass={stats['best_model_consistency']['pass']}, "
        f"threshold={stats['best_model_consistency']['min_required']})",
        "",
        "See `metrics.csv`/`metrics.json` for per-ticker detail and "
        "`statistical_tests.json` for the full pairwise Diebold-Mariano, "
        "Wilcoxon-Holm, and Friedman output.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _seed_everything(config.RANDOM_SEED)

    if not config.TICKERS:
        log.error("No tickers configured (no CSVs found in %s and PA_TICKERS unset).", config.RAW_DATA_DIR)
        return 1

    log.info("Evaluating %d ticker(s): %s", len(config.TICKERS), ", ".join(config.TICKERS))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs_by_ticker: dict[str, dict] = {}
    metrics_by_ticker: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for ticker in config.TICKERS:
        try:
            runs, metrics = evaluate_ticker(ticker)
        except (SplitError, LeakageError, CSVValidationError, FileNotFoundError, ValueError) as exc:
            log.error("Skipping %s: %s", ticker, exc)
            failures[ticker] = str(exc)
            continue
        runs_by_ticker[ticker] = runs
        metrics_by_ticker[ticker] = metrics

    if not metrics_by_ticker:
        log.error("No ticker completed evaluation successfully. Failures: %s", failures)
        return 1

    stats = run_full_statistical_suite(
        runs_by_ticker, metrics_by_ticker, config.ALL_MODEL_KEYS,
        min_consistency_tickers=min(config.MIN_CONSISTENCY_COMPANIES, len(metrics_by_ticker)),
    )
    if failures:
        stats["failed_tickers"] = failures

    # --- write outputs ---
    metrics_json_path = config.RESULTS_DIR / "metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_by_ticker, indent=2, sort_keys=True, default=str))

    metrics_csv_rows = _metrics_csv_rows(metrics_by_ticker)
    pd.DataFrame(metrics_csv_rows).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)

    model_comparison_rows = _model_comparison_csv_rows(stats)
    pd.DataFrame(model_comparison_rows).to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)

    (config.RESULTS_DIR / "statistical_tests.json").write_text(json.dumps(stats, indent=2, sort_keys=True, default=str))

    _write_summary_md(config.RESULTS_DIR / "evaluation_summary.md", metrics_by_ticker, stats, sorted(metrics_by_ticker.keys()))

    log.info("Wrote results to %s", config.RESULTS_DIR)
    log.info(
        "Aggregate ranking: %s",
        ", ".join(f"{e['rank']}. {e['model']} (RMSE={e['mean_rmse']:.4f})" for e in stats["aggregate_ranking"]),
    )
    if failures:
        log.warning("%d ticker(s) failed evaluation and were excluded: %s", len(failures), list(failures))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
