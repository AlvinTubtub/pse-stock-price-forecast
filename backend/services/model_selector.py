"""Training orchestration + best-model selection.

This is the one place that trains all three forecasting models for every
ticker, evaluates them, saves each trained model to disk, caches the full
results (metrics/predictions/backtests) for the dashboard to load without
retraining, writes ``best_models.json`` mapping each ticker to whichever
model had the lowest test-set RMSE, and — after every ticker has been
trained — runs the capstone paper's cross-model statistical-significance
suite (Diebold-Mariano/HLN within each company, Friedman + Holm-adjusted
Wilcoxon across companies, and the best-model consistency check) and
writes ``statistical_tests.json``.

Called exclusively by services/pdf_pipeline/pipeline.py after every
successful merge, and by the standalone ``python -m services.model_selector``
entrypoint for local/manual runs — never from inside the deployed
frontend, which only ever reads what this module wrote (via the exported
JSON, see scripts/export_forecast_artifacts.py).

Directory layout produced:

    models/
        lag_regression/<TICKER>.pkl
        arima/<TICKER>.pkl
        lstm/<TICKER>.pth
    prediction_cache/<TICKER>.json   # cached metrics + predictions for ui/data.py
    best_models.json                 # {"BDO": "LSTM", "MER": "ARIMA", ...}
    statistical_tests.json           # cross-model significance test results
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv
from services.pdf_pipeline.config import TARGET_COMPANIES
from services.evaluation import build_naive_formal_forecasts, evaluate_naive, run_formal_statistical_tests, select_best_model
from services.formal_evaluation import (
    FORMAL_FORECAST_COLUMNS,
    FORMAL_MODEL_KEYS,
    FormalHoldoutAlignmentError,
    validate_formal_holdout_alignment,
)
from services.forecasting import MODEL_LABELS, arima_model, lag_regression, lstm_model
from services.time_series_cv import FormalEvaluationPlan, create_formal_evaluation_plan

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
LAG_MODELS_DIR = MODELS_DIR / "lag_regression"
ARIMA_MODELS_DIR = MODELS_DIR / "arima"
LSTM_MODELS_DIR = MODELS_DIR / "lstm"
PREDICTION_CACHE_DIR = BASE_DIR / "prediction_cache"
BEST_MODELS_PATH = BASE_DIR / "best_models.json"
STATISTICAL_TESTS_PATH = BASE_DIR / "statistical_tests.json"

STAT_MODEL_KEYS = ("lag_reg", "arima", "lstm")
MIN_CONSISTENCY_COMPANIES = 8
EXPECTED_TICKERS = tuple(sorted(TARGET_COMPANIES.keys()))


def _ensure_dirs() -> None:
    for d in (LAG_MODELS_DIR, ARIMA_MODELS_DIR, LSTM_MODELS_DIR, PREDICTION_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)




def train_symbol(symbol: str, df: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray]]:
    """Trains + evaluates all three models for one ticker, saves each to
    disk, and returns (result, test_errors).

    ``result`` is cached for the dashboard (same shape as the legacy
    ``run_all_models`` return value, plus whatever additive diagnostic
    fields each model's metrics dict now includes, e.g. ARIMA's
    ``ljung_box_pvalue``). ``test_errors`` is {model_key: np.ndarray of
    (actual - predicted) reconstructed-price errors}, aligned to a common
    test window — used by train_and_select_all for the cross-model
    statistical tests, not persisted to the dashboard cache.
    """
    log.info("Training models for %s (%d rows)...", symbol, len(df))
    plan = create_formal_evaluation_plan(df, symbol)
    log.info(
        "%s formal evaluation plan: %d development and %d hold-out target dates (%s to %s).",
        symbol, plan.development_count, plan.holdout_count,
        plan.holdout_start_date.date(), plan.holdout_end_date.date(),
    )

    # Formal regression is development-only and supplies the genuine OOS
    # backtest/metrics.  The separately refit deployment artifact preserves
    # current daily-inference behavior and is the only artifact persisted.
    formal_lag = lag_regression.train_formal_lag_regression(df, plan)
    validate_formal_holdout_alignment({"lag_reg": formal_lag.forecasts}, plan, required_models=("lag_reg",))
    lag_artifact = lag_regression.train_deployment_lag_regression(df)
    lag_regression.save(lag_artifact, LAG_MODELS_DIR / f"{symbol}.pkl")
    lag_metrics = formal_lag.metrics
    lag_next = lag_regression.predict_next(lag_artifact, df)
    lag_backtest = formal_lag.backtest

    formal_arima = arima_model.train_formal_arima(df, plan)
    validate_formal_holdout_alignment({"arima": formal_arima.forecasts}, plan, required_models=("arima",))
    arima_fitted, deployment_order = arima_model.train_deployment_arima(df)
    arima_model.save(arima_fitted, ARIMA_MODELS_DIR / f"{symbol}.pkl")
    arima_metrics = formal_arima.metrics
    arima_next = arima_model.predict_next(arima_fitted) if arima_fitted is not None else float(df["Close"].iloc[-1])
    arima_backtest = formal_arima.backtest
    log.info("%s ARIMA formal order=%s deployment order=%s", symbol, formal_arima.order, deployment_order)

    formal_lstm = lstm_model.train_formal_lstm(df, plan)
    validate_formal_holdout_alignment({"lstm": formal_lstm.forecasts}, plan, required_models=("lstm",))
    lstm_artifact = lstm_model.train_deployment_lstm(df, formal_lstm.selected_config)
    lstm_model.save(lstm_artifact, LSTM_MODELS_DIR / f"{symbol}.pth")
    lstm_metrics = formal_lstm.metrics
    lstm_next = lstm_model.predict_next(lstm_artifact, df) if lstm_artifact is not None else float(df["Close"].iloc[-1])
    lstm_backtest = formal_lstm.backtest

    naive_metrics = evaluate_naive(df, plan=plan)

    result = {
        "metrics": {
            "lag_reg": lag_metrics,
            "arima": arima_metrics,
            "lstm": lstm_metrics,
            "naive": naive_metrics,
        },
        "next_close": {
            "lag": round(lag_next, 2),
            "arima": round(arima_next, 2),
            "lstm": round(lstm_next, 2),
        },
        "backtest30": lag_backtest[-30:] if len(lag_backtest) >= 30 else lag_backtest,
        "backtest_by_model": {
            "Lag-Informed Regression": lag_backtest,
            "ARIMA": arima_backtest,
            "LSTM": lstm_backtest,
        },
    }

    # All four methods now have exact date-indexed formal rows. Validate the
    # common plan before deriving aligned errors, but keep Phase 5 inference
    # disabled rather than reviving the old DM/Friedman workflow.
    naive_rows = build_naive_formal_forecasts(df, plan)
    validated = validate_formal_holdout_alignment(
        {"lag_reg": formal_lag.forecasts, "arima": formal_arima.forecasts, "lstm": formal_lstm.forecasts, "naive": naive_rows}, plan
    )
    return result, {"forecasts": validated, "development_close": df.loc[pd.to_datetime(df["Date"]) <= plan.development_end_date, "Close"].to_numpy(dtype=float)}


def train_and_select_all(raw_dir: Path = RAW_DIR, *, strict: bool = False) -> dict[str, str]:
    """Trains + saves models for every ticker CSV in ``raw_dir``, caches
    each ticker's results for the dashboard, writes best_models.json, and
    runs + saves the cross-model statistical-significance suite.

    Returns the {symbol: best_model_label} mapping. In the default mode,
    bad/unreadable CSVs are logged and skipped for backwards compatibility.
    In strict mode, the canonical 15-ticker universe must be present and
    every ticker must successfully train all three models; any failure raises
    RuntimeError and the caller must not commit partial training output.
    """
    _ensure_dirs()
    best_models: dict[str, str] = {}
    formal_by_symbol: dict[str, dict] = {}

    csv_paths = sorted(raw_dir.glob("*.csv"))
    csv_by_symbol = {p.stem.upper(): p for p in csv_paths}

    if strict:
        missing = sorted(set(EXPECTED_TICKERS) - set(csv_by_symbol))
        extra = sorted(set(csv_by_symbol) - set(EXPECTED_TICKERS))
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing expected ticker(s): {', '.join(missing)}")
            if extra:
                problems.append(f"unexpected ticker(s): {', '.join(extra)}")
            raise RuntimeError("Strict weekly training universe check failed — " + "; ".join(problems))
        csv_paths = [csv_by_symbol[symbol] for symbol in EXPECTED_TICKERS]
    elif not csv_paths:
        log.warning("No CSVs found in %s — nothing to train.", raw_dir)
        return best_models

    failures: dict[str, str] = {}

    for csv_path in csv_paths:
        symbol = csv_path.stem.upper()
        try:
            df = validate_ohlcv_csv(csv_path)
        except CSVValidationError as exc:
            message = f"failed OHLCV validation: {exc}"
            failures[symbol] = message
            log.warning("Skipping %s — %s", symbol, message)
            if strict:
                continue
            continue
        except Exception as exc:
            message = f"unexpected error while loading CSV: {exc}"
            failures[symbol] = message
            log.exception("Skipping %s — unexpected error while loading CSV.", symbol)
            if strict:
                continue
            continue

        try:
            result, formal_result = train_symbol(symbol, df)
        except Exception as exc:
            failures[symbol] = str(exc)
            log.exception("Training failed for %s — skipping.", symbol)
            continue

        best_key = select_best_model(result["metrics"], list(STAT_MODEL_KEYS))
        best_models[symbol] = MODEL_LABELS[best_key]

        (PREDICTION_CACHE_DIR / f"{symbol}.json").write_text(json.dumps(result, indent=2))
        log.info("%s best model: %s (RMSE %s)", symbol, MODEL_LABELS[best_key], result["metrics"][best_key]["rmse"])

        formal_by_symbol[symbol] = formal_result

    if strict and failures:
        detail = "; ".join(f"{symbol}: {message}" for symbol, message in sorted(failures.items()))
        raise RuntimeError(
            f"Strict weekly training failed for {len(failures)} ticker(s); "
            f"{len(best_models)}/{len(EXPECTED_TICKERS)} completed. {detail}"
        )

    if strict and set(best_models) != set(EXPECTED_TICKERS):
        missing = sorted(set(EXPECTED_TICKERS) - set(best_models))
        raise RuntimeError(
            f"Strict weekly training produced only {len(best_models)}/{len(EXPECTED_TICKERS)} "
            f"tickers; missing: {', '.join(missing)}"
        )

    BEST_MODELS_PATH.write_text(json.dumps(best_models, indent=2, sort_keys=True))
    log.info("Wrote %s (%d tickers)", BEST_MODELS_PATH, len(best_models))

    if formal_by_symbol:
        try:
            stats = run_formal_statistical_tests(formal_by_symbol)
            STATISTICAL_TESTS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True))
            log.info("Wrote Phase-5 formal statistics to %s", STATISTICAL_TESTS_PATH)
        except Exception:
            log.exception("Cross-model statistical tests failed — best_models.json is still valid, but statistical_tests.json was not updated.")

    return best_models


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Train and persist the weekly PSE forecasting models.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exactly the canonical 15-ticker universe and successful training for every ticker.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mapping = train_and_select_all(strict=args.strict)
    print(json.dumps(mapping, indent=2, sort_keys=True))
