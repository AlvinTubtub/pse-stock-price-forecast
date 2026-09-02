"""Explicitly separated formal, deployment-refresh, and retuning workflows.

Scheduled deployment refresh reads approved configurations, refits on current
validated data, and preserves prior research metrics and model-family choices.
Formal evaluation and manual challenger retuning have separate entrypoints and
artifact destinations.

Directory layout produced:

    models/deployment/current/
        lag_regression/<TICKER>.pkl
        arima/<TICKER>.pkl
        lstm/<TICKER>.pth
    prediction_cache/<TICKER>.json   # cached metrics + predictions for ui/data.py
    best_models.json                 # read-only approved family mapping during refresh
    statistical_tests.json           # written only by formal evaluation
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv
from services.artifact_runs import (
    DeploymentConfigurationError,
    FormalRunWriter,
    create_run_id,
    deployment_current_dir,
    git_worktree_is_dirty,
    load_approved_deployment_configurations,
    source_data_manifest,
    write_deployment_manifest,
)
from services.pdf_pipeline.config import TARGET_COMPANIES
from services.evaluation import build_naive_formal_forecasts, compute_canonical_formal_metrics, evaluate_naive, run_formal_residual_diagnostics, run_formal_statistical_tests
from services.formal_evaluation import validate_formal_holdout_alignment
from services.forecasting import arima_model, lag_regression, lstm_model
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
    lstm = lstm_model.train_formal_lstm(df, plan, development_cv_plan)
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


def train_symbol(symbol: str, df: pd.DataFrame) -> tuple[dict, dict[str, object]]:
    """Backward-compatible single-symbol deployment refresh entrypoint."""
    approved = load_approved_deployment_configurations(BASE_DIR, [symbol])[symbol]
    cache_path = PREDICTION_CACHE_DIR / f"{symbol}.json"
    if not cache_path.is_file():
        raise DeploymentConfigurationError(f"{symbol}: operational cache is required for deployment refresh.")
    result = refresh_deployment_symbol(symbol, df, approved, json.loads(cache_path.read_text()))
    return result, {"approved_configuration": approved}


def _parse_approved_configuration(symbol: str, payload: dict[str, object]) -> tuple:
    """Convert strict manifest metadata into model-specific configuration types."""
    try:
        lag_data = payload["lag_reg"]
        arima_data = payload["arima"]
        lstm_data = payload["lstm"]
        lag_config = lag_regression.LagRegressionDeploymentConfig(
            alpha=float(lag_data["alpha"]),
            candidate_features=tuple(str(value) for value in lag_data["candidate_features"]),
            pacf_selected_lags=tuple(int(value) for value in lag_data["pacf_selected_lags"]),
        )
        arima_config = arima_model.ARIMAConfiguration(
            order=tuple(int(value) for value in arima_data["order"]),
            trend=str(arima_data["trend"]),
        )
        lstm_config = lstm_model.LSTMConfig(
            lookback=int(lstm_data["lookback"]),
            hidden_size=int(lstm_data["hidden_size"]),
            learning_rate=float(lstm_data["learning_rate"]),
            batch_size=int(lstm_data["batch_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentConfigurationError(
            f"{symbol}: approved deployment configuration metadata is malformed."
        ) from exc
    if len(arima_config.order) != 3:
        raise DeploymentConfigurationError(f"{symbol}: approved ARIMA order must contain p, d, and q.")
    if (
        lstm_config.lookback not in lstm_model.LOOKBACK_GRID
        or lstm_config.hidden_size not in lstm_model.HIDDEN_UNITS_GRID
        or lstm_config.learning_rate not in lstm_model.LEARNING_RATE_GRID
        or lstm_config.batch_size not in lstm_model.BATCH_SIZE_GRID
    ):
        raise DeploymentConfigurationError(
            f"{symbol}: approved LSTM configuration is outside the supported model-family grid."
        )
    return lag_config, arima_config, lstm_config


def _approved_configuration_payload(lag_config, arima_config, lstm_config) -> dict[str, object]:
    return {
        "lag_reg": {
            "alpha": float(lag_config.alpha),
            "candidate_features": list(lag_config.candidate_features),
            "pacf_selected_lags": list(lag_config.pacf_selected_lags),
        },
        "arima": {"order": list(arima_config.order), "trend": arima_config.trend},
        "lstm": {
            "lookback": lstm_config.lookback,
            "hidden_size": lstm_config.hidden_size,
            "learning_rate": lstm_config.learning_rate,
            "batch_size": lstm_config.batch_size,
        },
    }


def refresh_deployment_symbol(
    symbol: str,
    df: pd.DataFrame,
    approved: dict[str, object],
    existing_cache: dict,
) -> dict:
    """Refit approved models and update predictions without formal evaluation."""
    lag_config, arima_config, lstm_config = _parse_approved_configuration(symbol, approved)
    if not isinstance(existing_cache.get("metrics"), dict) or not isinstance(existing_cache.get("deployment_backtest"), dict):
        raise DeploymentConfigurationError(
            f"{symbol}: existing operational cache lacks approved metrics or deployment_backtest."
        )
    log.info("Deployment refresh for %s (%d rows); formal evaluation is disabled.", symbol, len(df))
    lag_artifact = lag_regression.refit_deployment_lag_regression(df, lag_config)
    arima_fitted = arima_model.refit_deployment_arima(df, arima_config)
    lstm_artifact = lstm_model.train_deployment_lstm(df, lstm_config)
    lag_regression.save(lag_artifact, LAG_MODELS_DIR / f"{symbol}.pkl")
    arima_model.save(arima_fitted, ARIMA_MODELS_DIR / f"{symbol}.pkl")
    lstm_model.save(lstm_artifact, LSTM_MODELS_DIR / f"{symbol}.pth")
    result = dict(existing_cache)
    result["next_close"] = {
        "lag": round(lag_regression.predict_next(lag_artifact, df), 2),
        "arima": round(arima_model.predict_next(arima_fitted), 2),
        "lstm": round(lstm_model.predict_next(lstm_artifact, df), 2),
    }
    return result


def train_and_select_all(raw_dir: Path = RAW_DIR, *, strict: bool = False) -> dict[str, str]:
    """Backward-compatible alias for deployment refresh; it never retunes."""
    return refresh_deployment_all(raw_dir=raw_dir, strict=strict)


def refresh_deployment_all(raw_dir: Path = RAW_DIR, *, strict: bool = False) -> dict[str, str]:
    """Scheduled-safe refit of approved configurations on current data."""
    log.info("Starting deployment refresh; no formal evaluation or retuning will run.")
    _ensure_dirs()
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
            raise RuntimeError("Strict deployment refresh universe check failed — " + "; ".join(problems))
        csv_paths = [csv_by_symbol[symbol] for symbol in EXPECTED_TICKERS]
    elif not csv_paths:
        log.warning("No CSVs found in %s — nothing to refresh.", raw_dir)
        return {}

    symbols = [path.stem.upper() for path in csv_paths]
    approved = load_approved_deployment_configurations(BASE_DIR, symbols)
    if not BEST_MODELS_PATH.is_file():
        raise DeploymentConfigurationError("Deployment refresh requires the approved best_models.json mapping.")
    approved_families = json.loads(BEST_MODELS_PATH.read_text())
    missing_families = sorted(set(symbols) - set(approved_families))
    if missing_families:
        raise DeploymentConfigurationError(
            "Deployment refresh lacks approved model-family metadata for: " + ", ".join(missing_families)
        )

    best_models: dict[str, str] = {}

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
            cache_path = PREDICTION_CACHE_DIR / f"{symbol}.json"
            if not cache_path.is_file():
                raise DeploymentConfigurationError(f"{symbol}: operational cache is missing.")
            result = refresh_deployment_symbol(
                symbol,
                df,
                approved[symbol],
                json.loads(cache_path.read_text()),
            )
        except Exception as exc:
            failures[symbol] = str(exc)
            log.exception("Deployment refresh failed for %s — skipping.", symbol)
            continue

        best_models[symbol] = approved_families[symbol]
        (PREDICTION_CACHE_DIR / f"{symbol}.json").write_text(json.dumps(result, indent=2))
        log.info("Deployment refresh preserved %s approved model family: %s.", symbol, best_models[symbol])

    if strict and failures:
        detail = "; ".join(f"{symbol}: {message}" for symbol, message in sorted(failures.items()))
        raise RuntimeError(
            f"Strict deployment refresh failed for {len(failures)} ticker(s); "
            f"{len(best_models)}/{len(EXPECTED_TICKERS)} completed. {detail}"
        )

    if strict and set(best_models) != set(EXPECTED_TICKERS):
        missing = sorted(set(EXPECTED_TICKERS) - set(best_models))
        raise RuntimeError(
            f"Strict deployment refresh produced only {len(best_models)}/{len(EXPECTED_TICKERS)} "
            f"tickers; missing: {', '.join(missing)}"
        )

    deployment_artifacts = {
        symbol: {
            "lag_regression": str((LAG_MODELS_DIR / f"{symbol}.pkl").relative_to(BASE_DIR)),
            "arima": str((ARIMA_MODELS_DIR / f"{symbol}.pkl").relative_to(BASE_DIR)),
            "lstm": str((LSTM_MODELS_DIR / f"{symbol}.pth").relative_to(BASE_DIR)),
        }
        for symbol in best_models
    }
    manifest = write_deployment_manifest(
        BASE_DIR,
        deployment_artifacts,
        approved_configurations=approved,
        operation="refresh",
    )
    log.info("Deployment refresh wrote manifest to %s", manifest)

    return best_models


def retune_deployment_challengers(
    raw_dir: Path = RAW_DIR,
    *,
    symbols: list[str],
    run_id: str | None = None,
) -> Path:
    """Manually tune challenger configurations without promoting them."""
    if not symbols:
        raise ValueError("Deployment retuning requires an explicit symbol list.")
    requested = [symbol.upper() for symbol in symbols]
    if len(set(requested)) != len(requested):
        raise ValueError("Deployment retuning symbol list contains duplicates.")
    challenger_id = create_run_id(run_id)
    challenger_dir = BASE_DIR / "models" / "deployment" / "challengers" / challenger_id
    if challenger_dir.exists():
        raise FileExistsError(f"Deployment challenger run {challenger_id} already exists.")
    challenger_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "mode": "deployment",
        "operation": "retune",
        "status": "challenger_only",
        "automatic_promotion": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configurations": {},
    }
    log.info("Starting manual deployment retuning for %s; formal evidence is disabled.", requested)
    for symbol in requested:
        df = validate_ohlcv_csv(raw_dir / f"{symbol}.csv")
        lag_artifact = lag_regression.train_deployment_lag_regression(df)
        lag_config = lag_regression.deployment_config_from_artifact(lag_artifact)
        arima_artifact, arima_config = arima_model.retune_deployment_arima(df)
        lstm_artifact, lstm_config, lstm_diagnostics = lstm_model.retune_deployment_lstm(df, symbol)
        lag_regression.save(lag_artifact, challenger_dir / "lag_regression" / f"{symbol}.pkl")
        arima_model.save(arima_artifact, challenger_dir / "arima" / f"{symbol}.pkl")
        lstm_model.save(lstm_artifact, challenger_dir / "lstm" / f"{symbol}.pth")
        manifest["configurations"][symbol] = {
            **_approved_configuration_payload(lag_config, arima_config, lstm_config),
            "lstm_tuning_diagnostics": lstm_diagnostics,
        }
    path = challenger_dir / "challenger_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    log.info("Deployment retuning completed challenger-only run %s.", challenger_id)
    return path


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

    parser = argparse.ArgumentParser(description="Run deployment refresh/retuning or formal evaluation.")
    parser.add_argument(
        "--mode",
        choices=("deployment", "deployment-refresh", "deployment-retune", "formal"),
        default="deployment-refresh",
        help="'deployment' remains a compatibility alias for deployment-refresh.",
    )
    parser.add_argument("--symbols", nargs="+", help="Required for formal and deployment-retune modes.")
    parser.add_argument("--run-id", help="Optional formal or challenger-run identifier.")
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
    elif args.mode == "deployment-retune":
        if args.strict:
            parser.error("--strict is not supported for manual challenger retuning.")
        challenger = retune_deployment_challengers(
            symbols=args.symbols or [],
            run_id=args.run_id,
        )
        print(challenger)
    else:
        if args.symbols or args.run_id:
            parser.error("--symbols and --run-id are not deployment-refresh options.")
        mapping = refresh_deployment_all(strict=args.strict)
        print(json.dumps(mapping, indent=2, sort_keys=True))
