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

    models/deployment/current/
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
from services.artifact_runs import (
    FormalRunWriter,
    create_run_id,
    deployment_current_dir,
    git_worktree_is_dirty,
    source_data_manifest,
    write_deployment_manifest,
)
from services.pdf_pipeline.config import TARGET_COMPANIES
from services.evaluation import build_naive_formal_forecasts, compute_canonical_formal_metrics, evaluate_naive, run_formal_residual_diagnostics, run_formal_statistical_tests, select_best_model
from services.formal_evaluation import validate_formal_holdout_alignment
from services.forecasting import MODEL_LABELS, arima_model, lag_regression, lstm_model
from services.time_series_cv import FormalEvaluationPlan, create_development_cv_date_plan, create_formal_evaluation_plan

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
DEPLOYMENT_CURRENT_DIR = deployment_current_dir(BASE_DIR)
LAG_MODELS_DIR = DEPLOYMENT_CURRENT_DIR / "lag_regression"
ARIMA_MODELS_DIR = DEPLOYMENT_CURRENT_DIR / "arima"
LSTM_MODELS_DIR = DEPLOYMENT_CURRENT_DIR / "lstm"
PREDICTION_CACHE_DIR = BASE_DIR / "prediction_cache"
BEST_MODELS_PATH = BASE_DIR / "best_models.json"
STATISTICAL_TESTS_PATH = BASE_DIR / "statistical_tests.json"

STAT_MODEL_KEYS = ("lag_reg", "arima", "lstm")
MIN_CONSISTENCY_COMPANIES = 8
EXPECTED_TICKERS = tuple(sorted(TARGET_COMPANIES.keys()))


def _ensure_dirs() -> None:
    for d in (LAG_MODELS_DIR, ARIMA_MODELS_DIR, LSTM_MODELS_DIR, PREDICTION_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _deployment_backtest_payload(forecasts: dict[str, pd.DataFrame]) -> dict:
    """Serialize the audited OOS holdout by target date for frontend export.

    This is deployment metadata, not a formal-run artifact.  The formal run
    remains immutable under ``results/formal``; the cache only carries the
    same validated, date-indexed OOS observations needed by the live charts.
    """
    labels = {
        "lag_reg": "Lag-Informed Regression",
        "arima": "ARIMA",
        "lstm": "LSTM",
        "naive": "Naive baseline",
    }
    reference = forecasts["lag_reg"].sort_values("target_date").reset_index(drop=True)
    target_dates = pd.to_datetime(reference["target_date"], errors="raise")
    return {
        "schema_version": 1,
        "source": "audited_oos_holdout",
        "alignment": "common_target_date",
        "target_dates": [date.strftime("%Y-%m-%d") for date in target_dates],
        "actual": [float(value) for value in reference["actual_close"]],
        "by_model": {
            labels[key]: [
                float(value)
                for value in forecasts[key].sort_values("target_date")["predicted_close"]
            ]
            for key in labels
        },
    }

def evaluate_formal_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """Evaluate a symbol without writing deployment, cache, or frontend artifacts."""
    log.info("Formal evaluation for %s (%d rows).", symbol, len(df))
    plan = create_formal_evaluation_plan(df, symbol)
    development_cv_plan = create_development_cv_date_plan(plan)
    lag = lag_regression.train_formal_lag_regression(df, plan)
    arima = arima_model.train_formal_arima(df, plan)
    lstm = lstm_model.train_formal_lstm(df, plan)
    forecasts = validate_formal_holdout_alignment({"lag_reg": lag.forecasts, "arima": arima.forecasts, "lstm": lstm.forecasts, "naive": build_naive_formal_forecasts(df, plan)}, plan)
    development_close = df.loc[
        pd.to_datetime(df["Date"]) <= plan.development_end_date, "Close"
    ].to_numpy(dtype=float)
    mase_denominator, metrics = compute_canonical_formal_metrics(
        forecasts,
        development_close,
        existing_metrics={
            "lag_reg": lag.metrics,
            "arima": arima.metrics,
            "lstm": lstm.metrics,
            "naive": evaluate_naive(df, plan=plan),
        },
    )
    log.info("Formal evaluation for %s uses canonical MASE denominator %.12g.", symbol, mase_denominator)
    lag_metadata = {
        "selected_alpha": lag.artifact.alpha,
        "selected_features": lag.artifact.selected_features,
        "coefficients": {name: float(value) for name, value in zip(lag.artifact.candidate_features, lag.artifact.model.coef_)},
        **run_formal_residual_diagnostics(forecasts["lag_reg"]["error"], include_ljung_box=True),
    }
    lstm_metadata = {"training_metadata": lstm.metadata, **run_formal_residual_diagnostics(forecasts["lstm"]["error"])}
    return {"plan": plan, "development_cv_plan": development_cv_plan, "forecasts": forecasts, "metrics": metrics, "diagnostics": {"lag_reg": lag_metadata, "arima": arima.diagnostics, "lstm": lstm_metadata}, "development_close": development_close, "mase_denominator": mase_denominator, "lstm_config": lstm.selected_config, "backtests": {"lag_reg": lag.backtest, "arima": arima.backtest, "lstm": lstm.backtest}}


def train_symbol(symbol: str, df: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray]]:
    """Refit deployment artifacts after deriving the established formal configuration."""
    formal = evaluate_formal_symbol(symbol, df)
    log.info("Deployment refit for %s writes only deployment-current artifacts.", symbol)
    lag_artifact = lag_regression.train_deployment_lag_regression(df); lag_regression.save(lag_artifact, LAG_MODELS_DIR / f"{symbol}.pkl")
    arima_fitted, deployment_order = arima_model.train_deployment_arima(df); arima_model.save(arima_fitted, ARIMA_MODELS_DIR / f"{symbol}.pkl")
    lstm_artifact = lstm_model.train_deployment_lstm(df, formal["lstm_config"]); lstm_model.save(lstm_artifact, LSTM_MODELS_DIR / f"{symbol}.pth")
    result = {
        "metrics": formal["metrics"],
        "next_close": {
            "lag": round(lag_regression.predict_next(lag_artifact, df), 2),
            "arima": round(arima_model.predict_next(arima_fitted), 2) if arima_fitted is not None else float(df["Close"].iloc[-1]),
            "lstm": round(lstm_model.predict_next(lstm_artifact, df), 2) if lstm_artifact is not None else float(df["Close"].iloc[-1]),
        },
        "deployment_backtest": _deployment_backtest_payload(formal["forecasts"]),
    }
    log.info("%s deployment ARIMA order=%s", symbol, deployment_order)
    return result, {"forecasts": formal["forecasts"], "development_close": formal["development_close"]}


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

    deployment_artifacts = {
        symbol: {
            "lag_regression": str((LAG_MODELS_DIR / f"{symbol}.pkl").relative_to(BASE_DIR)),
            "arima": str((ARIMA_MODELS_DIR / f"{symbol}.pkl").relative_to(BASE_DIR)),
            "lstm": str((LSTM_MODELS_DIR / f"{symbol}.pth").relative_to(BASE_DIR)),
        }
        for symbol in best_models
    }
    manifest = write_deployment_manifest(BASE_DIR, deployment_artifacts)
    log.info("Wrote deployment manifest to %s", manifest)

    return best_models


def run_formal_evaluation(
    raw_dir: Path = RAW_DIR,
    *,
    symbols: list[str],
    run_id: str | None = None,
) -> Path:
    """Manually create one immutable formal run; never write deployment state.

    A caller must explicitly name its company universe.  This prevents a
    scheduled deployment workflow from accidentally launching a full formal
    research experiment.
    """
    if not symbols:
        raise ValueError("Formal evaluation requires an explicit symbol list (for example: BPI).")
    requested = [symbol.upper() for symbol in symbols]
    if len(set(requested)) != len(requested):
        raise ValueError("Formal evaluation symbol list contains duplicates.")
    git_dirty = git_worktree_is_dirty()
    if git_dirty:
        raise RuntimeError("Formal evaluation requires a clean Git worktree; commit or stash changes first.")
    writer = FormalRunWriter(BASE_DIR, create_run_id(run_id))
    writer.create()
    formal_by_symbol: dict[str, dict] = {}
    plans: dict[str, FormalEvaluationPlan] = {}
    source_provenance: dict[str, dict[str, object]] = {}
    cutoff_dates: list[pd.Timestamp] = []
    try:
        for symbol in requested:
            csv_path = raw_dir / f"{symbol}.csv"
            df = validate_ohlcv_csv(csv_path)
            payload = evaluate_formal_symbol(symbol, df)
            plans[symbol] = payload["plan"]
            source_provenance[symbol] = source_data_manifest(csv_path, df)
            formal_by_symbol[symbol] = payload
            cutoff_dates.append(pd.to_datetime(df["Date"]).max())
        writer.write_split_manifest(plans)
        writer.write_data_manifest(source_provenance)
        writer.write_methodology_manifest(str(max(cutoff_dates).date()), requested, git_dirty=git_dirty)
        statistics = run_formal_statistical_tests(formal_by_symbol)
        for symbol in requested:
            stage1 = statistics["per_company"][symbol]["dm_squared_error"]["stage1_vs_naive"]
            flags = {item["model_a"]: {"beats_naive_rmse": item["beats_naive_rmse"], "significantly_beats_naive": item["significantly_beats_naive"]} for item in stage1}
            for model_key in ("lag_reg", "arima", "lstm"):
                formal_by_symbol[symbol]["diagnostics"][model_key].update(flags[model_key])
            writer.write_company(symbol, formal_by_symbol[symbol]["forecasts"], formal_by_symbol[symbol]["metrics"], formal_by_symbol[symbol]["diagnostics"])
        writer.write_statistics(statistics)
        finalized = writer.finalize()
        log.info("Finalized formal run %s", writer.path)
        return finalized
    except Exception:
        log.exception("Formal run %s failed before finalization; deployment artifacts were not modified.", writer.run_id)
        raise


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run deployment training or a manually requested formal evaluation.")
    parser.add_argument("--mode", choices=("deployment", "formal"), default="deployment")
    parser.add_argument("--symbols", nargs="+", help="Required for manual formal mode; use BPI for integration validation.")
    parser.add_argument("--run-id", help="Optional immutable formal-run identifier.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require exactly the canonical 15-ticker universe and successful training for every ticker.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.mode == "formal":
        if args.strict:
            parser.error("--strict is a deployment-only option.")
        finalized = run_formal_evaluation(symbols=args.symbols or [], run_id=args.run_id)
        print(finalized)
    else:
        if args.symbols or args.run_id:
            parser.error("--symbols and --run-id are formal-mode options.")
        mapping = train_and_select_all(strict=args.strict)
        print(json.dumps(mapping, indent=2, sort_keys=True))
