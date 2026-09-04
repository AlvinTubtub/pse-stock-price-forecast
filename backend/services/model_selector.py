"""Separated formal, deployment-refresh, retuning, and approval workflows.

Scheduled deployment refresh reads approved configurations, refits on current
validated data, and preserves prior research metrics and model-family choices.
Formal evaluation, manual challenger retuning, and manually confirmed approval
have separate entrypoints and artifact destinations.

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
import math
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from services.data_validator import CSVValidationError, validate_ohlcv_csv
from services.artifact_runs import (
    DeploymentConfigurationError,
    FormalRunWriter,
    create_run_id,
    deployment_current_dir,
    file_sha256,
    git_repository_commit,
    git_worktree_is_dirty,
    load_approved_deployment_configurations,
    source_data_manifest,
    write_deployment_manifest,
)
from services.pdf_pipeline.config import TARGET_COMPANIES
from services.evaluation import build_naive_formal_forecasts, compute_canonical_formal_metrics, evaluate_naive, run_formal_residual_diagnostics, run_formal_statistical_tests
from services.corporate_actions import build_corporate_action_evidence, load_verified_event_registry
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

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SAFE_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,31}")
_DEPLOYMENT_ARTIFACTS = {
    "lag_regression": ("lag_reg", ".pkl"),
    "arima": ("arima", ".pkl"),
    "lstm": ("lstm", ".pth"),
}


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

def evaluate_formal_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    verified_corporate_actions: tuple[dict[str, object], ...] = (),
) -> dict:
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
        "tuning_metadata": lag.tuning_metadata,
        "selected_features": lag.artifact.selected_features,
        "coefficients": {name: float(value) for name, value in zip(lag.artifact.candidate_features, lag.artifact.model.coef_)},
        **run_formal_residual_diagnostics(forecasts["lag_reg"]["error"], include_ljung_box=True),
    }
    lstm_metadata = {"training_metadata": lstm.metadata, **run_formal_residual_diagnostics(forecasts["lstm"]["error"])}
    corporate_action_evidence = build_corporate_action_evidence(
        forecasts,
        development_close,
        verified_events=verified_corporate_actions,
    )
    return {"plan": plan, "development_cv_plan": development_cv_plan, "forecasts": forecasts, "metrics": metrics, "diagnostics": {"lag_reg": lag_metadata, "arima": arima.diagnostics, "lstm": lstm_metadata, "corporate_actions": corporate_action_evidence}, "development_close": development_close, "mase_denominator": mase_denominator, "lstm_config": lstm.selected_config, "backtests": {"lag_reg": lag.backtest, "arima": arima.backtest, "lstm": lstm.backtest}}


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
    if not math.isfinite(lag_config.alpha) or lag_config.alpha <= 0:
        raise DeploymentConfigurationError(
            f"{symbol}: approved Lag Regression alpha must be a positive finite number."
        )
    if not lag_config.candidate_features or len(set(lag_config.candidate_features)) != len(
        lag_config.candidate_features
    ):
        raise DeploymentConfigurationError(
            f"{symbol}: approved Lag Regression features must be non-empty and unique."
        )
    unsupported_features = sorted(
        set(lag_config.candidate_features) - set(lag_regression.REGRESSION_FEATURE_COLUMNS)
    )
    if unsupported_features:
        raise DeploymentConfigurationError(
            f"{symbol}: approved Lag Regression features are unsupported: {unsupported_features}."
        )
    if any(lag <= 0 for lag in lag_config.pacf_selected_lags) or len(
        set(lag_config.pacf_selected_lags)
    ) != len(lag_config.pacf_selected_lags):
        raise DeploymentConfigurationError(
            f"{symbol}: approved PACF lags must be positive and unique."
        )
    p, d, q = arima_config.order
    allowed_trends = {0: {"n", "c"}, 1: {"n", "t"}, 2: {"n"}}
    if p not in range(4) or d not in range(3) or q not in range(4):
        raise DeploymentConfigurationError(
            f"{symbol}: approved ARIMA order is outside the supported candidate grid."
        )
    if arima_config.trend not in allowed_trends[d]:
        raise DeploymentConfigurationError(
            f"{symbol}: approved ARIMA trend is invalid for differencing order {d}."
        )
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


def _load_json_without_duplicates(path: Path, description: str) -> dict[str, object]:
    """Load a JSON object and reject duplicate keys at every nesting level."""
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DeploymentConfigurationError(
                    f"{description} contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentConfigurationError(f"Cannot read valid {description} at {path}.") from exc
    if not isinstance(payload, dict):
        raise DeploymentConfigurationError(f"{description} must be a JSON object.")
    return payload


def _validate_challenger_run_id(run_id: str) -> str:
    """Return a safe challenger run ID that cannot escape its parent directory."""
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise DeploymentConfigurationError(
            "Challenger run ID must be one safe path component containing only letters, "
            "numbers, dots, underscores, or hyphens."
        )
    return run_id


def _validate_approval_inputs(
    challenger_run_id: str,
    symbols: list[str] | None,
) -> dict[str, object]:
    """Validate a complete challenger approval before deployment state can change."""
    safe_run_id = _validate_challenger_run_id(challenger_run_id)
    challengers_root = BASE_DIR / "models" / "deployment" / "challengers"
    challenger_dir = challengers_root / safe_run_id
    if challenger_dir.resolve().parent != challengers_root.resolve():
        raise DeploymentConfigurationError("Challenger run path escapes the challengers directory.")
    manifest_path = challenger_dir / "challenger_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DeploymentConfigurationError(
            f"Challenger run {safe_run_id} lacks a regular challenger_manifest.json."
        )
    manifest = _load_json_without_duplicates(manifest_path, "challenger manifest")
    required_identity = {
        "mode": "deployment",
        "operation": "retune",
        "status": "challenger_only",
        "automatic_promotion": False,
    }
    for field, expected in required_identity.items():
        if manifest.get(field) != expected or (
            field == "automatic_promotion" and manifest.get(field) is not False
        ):
            raise DeploymentConfigurationError(
                f"Challenger manifest requires {field}={expected!r}."
            )
    configurations = manifest.get("configurations")
    if not isinstance(configurations, dict) or not configurations:
        raise DeploymentConfigurationError(
            "Challenger manifest requires a non-empty per-symbol configurations object."
        )
    available = list(configurations)
    invalid_available = sorted(
        symbol for symbol in available
        if not isinstance(symbol, str) or not _SAFE_SYMBOL.fullmatch(symbol)
    )
    if invalid_available:
        raise DeploymentConfigurationError(
            "Challenger manifest contains malformed or unexpected symbol(s): "
            + ", ".join(map(str, invalid_available))
        )
    unexpected_available = sorted(set(available) - set(EXPECTED_TICKERS))
    if unexpected_available:
        raise DeploymentConfigurationError(
            "Challenger manifest contains unexpected deployment symbol(s): "
            + ", ".join(unexpected_available)
        )
    for artifact_name, (_configuration_name, suffix) in _DEPLOYMENT_ARTIFACTS.items():
        artifact_dir = challenger_dir / artifact_name
        if artifact_dir.exists() and not artifact_dir.is_dir():
            raise DeploymentConfigurationError(
                f"Challenger {artifact_name} artifact location must be a directory."
            )
        artifact_symbols = {path.stem for path in artifact_dir.glob(f"*{suffix}")}
        unexpected_artifacts = sorted(artifact_symbols - set(available))
        if unexpected_artifacts:
            raise DeploymentConfigurationError(
                f"Challenger {artifact_name} directory contains unexpected symbol artifact(s): "
                + ", ".join(unexpected_artifacts)
            )
    if symbols is None:
        requested = sorted(available)
    else:
        requested = [symbol.upper() for symbol in symbols]
        if not requested:
            raise DeploymentConfigurationError("Deployment approval symbol list cannot be empty.")
        if len(set(requested)) != len(requested):
            raise DeploymentConfigurationError("Deployment approval symbol list contains duplicates.")
        invalid_requested = sorted(symbol for symbol in requested if not _SAFE_SYMBOL.fullmatch(symbol))
        if invalid_requested:
            raise DeploymentConfigurationError(
                "Deployment approval contains malformed symbol(s): " + ", ".join(invalid_requested)
            )
        missing = sorted(set(requested) - set(available))
        if missing:
            raise DeploymentConfigurationError(
                "Requested symbol(s) are absent from the challenger manifest: " + ", ".join(missing)
            )

    normalized_configurations: dict[str, dict[str, object]] = {}
    source_artifacts: dict[str, dict[str, Path]] = {}
    challenger_root = challenger_dir.resolve()
    for symbol in requested:
        raw_configuration = configurations[symbol]
        if not isinstance(raw_configuration, dict):
            raise DeploymentConfigurationError(f"{symbol}: challenger configuration must be an object.")
        lag_config, arima_config, lstm_config = _parse_approved_configuration(symbol, raw_configuration)
        normalized_configurations[symbol] = _approved_configuration_payload(
            lag_config, arima_config, lstm_config
        )
        source_artifacts[symbol] = {}
        for artifact_name, (_configuration_name, suffix) in _DEPLOYMENT_ARTIFACTS.items():
            source = challenger_dir / artifact_name / f"{symbol}{suffix}"
            try:
                source_resolved = source.resolve(strict=True)
            except OSError as exc:
                raise DeploymentConfigurationError(
                    f"{symbol}: challenger {artifact_name} artifact is missing."
                ) from exc
            if source.is_symlink() or not source_resolved.is_file() or challenger_root not in source_resolved.parents:
                raise DeploymentConfigurationError(
                    f"{symbol}: challenger {artifact_name} artifact is not a safe regular file."
                )
            if source_resolved.stat().st_size == 0:
                raise DeploymentConfigurationError(
                    f"{symbol}: challenger {artifact_name} artifact is empty."
                )
            source_artifacts[symbol][artifact_name] = source_resolved

    if not BEST_MODELS_PATH.is_file() or BEST_MODELS_PATH.is_symlink():
        raise DeploymentConfigurationError(
            "Deployment approval requires the existing approved best_models.json mapping."
        )
    approved_families = _load_json_without_duplicates(BEST_MODELS_PATH, "best-model family mapping")
    missing_families = sorted(set(requested) - set(approved_families))
    allowed_families = {"Lag-Informed Regression", "ARIMA", "LSTM"}
    malformed_families = sorted(
        symbol for symbol in requested
        if symbol in approved_families
        and approved_families[symbol] not in allowed_families
    )
    if missing_families or malformed_families:
        details = []
        if missing_families:
            details.append("missing: " + ", ".join(missing_families))
        if malformed_families:
            details.append("malformed: " + ", ".join(malformed_families))
        raise DeploymentConfigurationError(
            "Approval cannot preserve the existing approved model family (" + "; ".join(details) + ")."
        )

    current_manifest_path = DEPLOYMENT_CURRENT_DIR / "deployment_manifest.json"
    if current_manifest_path.exists():
        if not current_manifest_path.is_file() or current_manifest_path.is_symlink():
            raise DeploymentConfigurationError("Current deployment manifest must be a regular file.")
        current_manifest = _load_json_without_duplicates(
            current_manifest_path, "current deployment manifest"
        )
    else:
        current_manifest = {}
    existing_artifacts = current_manifest.get("artifacts", {})
    existing_configurations = current_manifest.get("approved_configurations", {})
    if not isinstance(existing_artifacts, dict) or not isinstance(existing_configurations, dict):
        raise DeploymentConfigurationError(
            "Current deployment manifest has malformed artifacts or approved_configurations."
        )
    return {
        "run_id": safe_run_id,
        "requested": requested,
        "source_artifacts": source_artifacts,
        "approved_configurations": {**existing_configurations, **normalized_configurations},
        "artifacts": dict(existing_artifacts),
    }


def approve_deployment_challenger(
    challenger_run_id: str,
    *,
    symbols: list[str] | None = None,
    confirmed: bool = False,
) -> Path:
    """Manually promote validated challenger artifacts without changing model families."""
    if not confirmed:
        raise DeploymentConfigurationError(
            "Deployment approval requires the explicit --confirm-approved flag."
        )
    plan = _validate_approval_inputs(challenger_run_id, symbols)
    requested = plan["requested"]
    log.info(
        "Starting manually confirmed deployment approval for challenger %s: %s.",
        plan["run_id"], requested,
    )
    deployment_root = BASE_DIR / "models" / "deployment"
    deployment_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".approval-", dir=deployment_root))
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    replaced: list[Path] = []
    try:
        # Copy and hash-check every artifact before deployment-current is modified.
        for symbol in requested:
            plan["artifacts"][symbol] = {}
            for artifact_name, (_configuration_name, suffix) in _DEPLOYMENT_ARTIFACTS.items():
                source = plan["source_artifacts"][symbol][artifact_name]
                destination = DEPLOYMENT_CURRENT_DIR / artifact_name / f"{symbol}{suffix}"
                staged_path = staging_dir / "new" / artifact_name / f"{symbol}{suffix}"
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged_path)
                if staged_path.stat().st_size == 0 or file_sha256(staged_path) != file_sha256(source):
                    raise DeploymentConfigurationError(
                        f"{symbol}: staged {artifact_name} artifact failed integrity validation."
                    )
                staged[destination] = staged_path
                plan["artifacts"][symbol][artifact_name] = str(destination.relative_to(BASE_DIR))

        for index, destination in enumerate(staged):
            if destination.is_symlink():
                raise DeploymentConfigurationError(
                    f"Refusing to replace symbolic-link deployment artifact: {destination}."
                )
            if destination.exists():
                backup = staging_dir / "backup" / str(index)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
                backups[destination] = backup
            else:
                backups[destination] = None

        for destination, staged_path in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, destination)
            replaced.append(destination)

        manifest_path = write_deployment_manifest(
            BASE_DIR,
            plan["artifacts"],
            approved_configurations=plan["approved_configurations"],
            operation="approval",
            source_challenger_run_id=plan["run_id"],
            approved_symbols=requested,
            manually_approved=True,
        )
        log.info(
            "Manual deployment approval completed for challenger %s; model-family choices were preserved.",
            plan["run_id"],
        )
        return manifest_path
    except Exception:
        for destination in reversed(replaced):
            backup = backups[destination]
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        log.exception(
            "Deployment approval for challenger %s failed; promoted artifacts were rolled back.",
            plan["run_id"],
        )
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


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
    expected_data_cutoff: str | None = None,
    expected_row_count: int | None = None,
    resume: bool = False,
    corporate_actions_file: Path | None = None,
) -> Path:
    """Create or resume one immutable formal run without deployment writes.

    A caller must explicitly name its company universe.  This prevents a
    scheduled deployment workflow from accidentally launching a full formal
    research experiment.  All requested raw files are validated and hashed
    before a run directory is created.  Completed company checkpoints are
    reusable only with the same commit, source hashes, cutoff, row count, and
    symbol universe.
    """
    if not symbols:
        raise ValueError("Formal evaluation requires an explicit symbol list (for example: BPI).")
    requested = [symbol.upper() for symbol in symbols]
    if len(set(requested)) != len(requested):
        raise ValueError("Formal evaluation symbol list contains duplicates.")
    if expected_row_count is not None and expected_row_count <= 1:
        raise ValueError("Expected formal source row count must be greater than one.")
    canonical_universe = set(requested) == set(EXPECTED_TICKERS)
    if canonical_universe and corporate_actions_file is None:
        raise ValueError(
            "The canonical 15-company formal run requires a reviewed --corporate-actions-file; "
            "do not claim that an omitted registry means no events occurred."
        )
    expected_cutoff = None
    if expected_data_cutoff is not None:
        try:
            expected_cutoff = pd.Timestamp(expected_data_cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected formal data cutoff must be a valid date.") from exc
        if expected_cutoff.tzinfo is not None:
            expected_cutoff = expected_cutoff.tz_localize(None)
        expected_cutoff = expected_cutoff.normalize()

    git_dirty = git_worktree_is_dirty()
    if git_dirty:
        raise RuntimeError("Formal evaluation requires a clean Git worktree; commit or stash changes first.")

    raw_dir = Path(raw_dir)
    validated_data: dict[str, pd.DataFrame] = {}
    source_provenance: dict[str, dict[str, object]] = {}
    cutoff_dates: list[pd.Timestamp] = []
    log.info("Formal preflight: validating %d frozen source file(s) in %s.", len(requested), raw_dir)
    for symbol in requested:
        csv_path = raw_dir / f"{symbol}.csv"
        df = validate_ohlcv_csv(csv_path)
        dates = pd.to_datetime(df["Date"], errors="raise")
        last_date = pd.Timestamp(dates.max()).normalize()
        if expected_row_count is not None and len(df) != expected_row_count:
            raise ValueError(
                f"{symbol}: expected {expected_row_count} frozen rows, found {len(df)}."
            )
        if expected_cutoff is not None and last_date != expected_cutoff:
            raise ValueError(
                f"{symbol}: expected frozen cutoff {expected_cutoff.date()}, found {last_date.date()}."
            )
        validated_data[symbol] = df
        source_provenance[symbol] = source_data_manifest(csv_path, df)
        cutoff_dates.append(last_date)
        log.info(
            "Formal preflight passed for %s: %d rows, cutoff %s, sha256 %s.",
            symbol,
            len(df),
            last_date.date(),
            source_provenance[symbol]["sha256"],
        )

    if corporate_actions_file is None:
        corporate_actions, corporate_action_registry = load_verified_event_registry(None, requested)
    else:
        preflight_plans = {
            symbol: create_formal_evaluation_plan(validated_data[symbol], symbol)
            for symbol in requested
        }
        corporate_actions, corporate_action_registry = load_verified_event_registry(
            corporate_actions_file,
            requested,
            required_start_date=min(plan.holdout_start_date for plan in preflight_plans.values()),
            required_end_date=max(plan.holdout_end_date for plan in preflight_plans.values()),
        )

    formal_run_id = create_run_id(run_id)
    writer = FormalRunWriter(BASE_DIR, formal_run_id)
    if writer.path.exists() and not resume:
        raise FileExistsError(
            f"Formal run {formal_run_id} already exists. Use --resume only for its unchanged inputs."
        )
    resume_contract = {
        "formal_evaluation_version": "1.0",
        "repository_commit": git_repository_commit(),
        "symbols": requested,
        "expected_data_cutoff": str(expected_cutoff.date()) if expected_cutoff is not None else None,
        "expected_row_count": expected_row_count,
        "source_files": {
            symbol: {
                "sha256": source_provenance[symbol]["sha256"],
                "row_count": source_provenance[symbol]["row_count"],
                "first_date": str(pd.Timestamp(source_provenance[symbol]["first_date"]).date()),
                "last_date": str(pd.Timestamp(source_provenance[symbol]["last_date"]).date()),
            }
            for symbol in requested
        },
        "corporate_action_registry": corporate_action_registry,
    }
    resumed = writer.create_or_resume(resume_contract)
    log.info(
        "%s formal run %s with %d requested company/companies.",
        "Resuming" if resumed else "Starting",
        formal_run_id,
        len(requested),
    )
    formal_by_symbol: dict[str, dict] = {}
    plans: dict[str, FormalEvaluationPlan] = {}
    started_at = time.monotonic()
    try:
        for index, symbol in enumerate(requested, start=1):
            if writer.has_company_checkpoint(symbol):
                log.info(
                    "Formal company %d/%d %s: loading completed checkpoint.",
                    index,
                    len(requested),
                    symbol,
                )
                payload = writer.read_company_checkpoint(symbol)
            else:
                company_started_at = time.monotonic()
                log.info(
                    "Formal company %d/%d %s: evaluation started.",
                    index,
                    len(requested),
                    symbol,
                )
                if corporate_actions[symbol]:
                    payload = evaluate_formal_symbol(
                        symbol,
                        validated_data[symbol],
                        verified_corporate_actions=corporate_actions[symbol],
                    )
                else:
                    payload = evaluate_formal_symbol(symbol, validated_data[symbol])
                writer.write_company_checkpoint(symbol, payload, source_provenance[symbol])
                payload = writer.read_company_checkpoint(symbol)
                elapsed = time.monotonic() - company_started_at
                remaining = len(requested) - index
                log.info(
                    "Formal company %d/%d %s: checkpoint completed in %.1f minutes; "
                    "rough remaining estimate %.1f minutes.",
                    index,
                    len(requested),
                    symbol,
                    elapsed / 60,
                    (elapsed * remaining) / 60,
                )
            plans[symbol] = payload["plan"]
            formal_by_symbol[symbol] = payload
        writer.write_split_manifest(plans)
        writer.write_data_manifest(source_provenance)
        writer.write_methodology_manifest(
            str(max(cutoff_dates).date()),
            requested,
            git_dirty=git_dirty,
            corporate_action_registry=corporate_action_registry,
        )
        statistics = run_formal_statistical_tests(formal_by_symbol)
        for symbol in requested:
            stage1 = statistics["per_company"][symbol]["dm_squared_error"]["stage1_vs_naive"]
            flags = {item["model_a"]: {"beats_naive_rmse": item["beats_naive_rmse"], "significantly_beats_naive": item["significantly_beats_naive"]} for item in stage1}
            for model_key in ("lag_reg", "arima", "lstm"):
                formal_by_symbol[symbol]["diagnostics"][model_key].update(flags[model_key])
            writer.write_company(symbol, formal_by_symbol[symbol]["forecasts"], formal_by_symbol[symbol]["metrics"], formal_by_symbol[symbol]["diagnostics"])
        writer.write_statistics(statistics)
        writer.mark_evaluation_complete()
        finalized = writer.finalize()
        log.info(
            "Finalized formal run %s in %.1f minutes.",
            writer.path,
            (time.monotonic() - started_at) / 60,
        )
        return finalized
    except Exception:
        log.exception(
            "Formal run %s failed before finalization; deployment artifacts were not modified. "
            "After correcting the environment, rerun the same command with --resume.",
            writer.run_id,
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Run deployment refresh, retuning, manual approval, or formal evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "deployment",
            "deployment-refresh",
            "deployment-retune",
            "deployment-approve",
            "formal",
        ),
        default="deployment-refresh",
        help="'deployment' remains a compatibility alias for deployment-refresh.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Required for formal/retune; optional for approval (defaults to the challenger universe).",
    )
    parser.add_argument("--run-id", help="Optional formal or challenger-run identifier.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Formal-only frozen OHLCV directory; required to prevent use of moving production data.",
    )
    parser.add_argument(
        "--expected-data-cutoff",
        help="Formal-only required last trading date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--expected-row-count",
        type=int,
        help="Formal-only required row count for every requested company CSV.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete formal run after verifying its code and frozen-input contract.",
    )
    parser.add_argument(
        "--corporate-actions-file",
        type=Path,
        help="Formal-only reviewed JSON registry of verified corporate-action dates.",
    )
    parser.add_argument(
        "--challenger-run-id",
        help="Required source challenger run identifier for deployment-approve.",
    )
    parser.add_argument(
        "--confirm-approved",
        action="store_true",
        help="Required explicit confirmation for deployment-approve.",
    )
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
        if args.challenger_run_id or args.confirm_approved:
            parser.error("Challenger approval options are valid only for deployment-approve.")
        if args.raw_dir is None or args.expected_data_cutoff is None or args.expected_row_count is None:
            parser.error(
                "formal mode requires --raw-dir, --expected-data-cutoff, and --expected-row-count."
            )
        finalized = run_formal_evaluation(
            raw_dir=args.raw_dir,
            symbols=args.symbols or [],
            run_id=args.run_id,
            expected_data_cutoff=args.expected_data_cutoff,
            expected_row_count=args.expected_row_count,
            resume=args.resume,
            corporate_actions_file=args.corporate_actions_file,
        )
        print(finalized)
    elif args.mode == "deployment-retune":
        if args.raw_dir or args.expected_data_cutoff or args.expected_row_count or args.resume or args.corporate_actions_file:
            parser.error("Frozen-data and resume options are valid only for formal mode.")
        if args.strict:
            parser.error("--strict is not supported for manual challenger retuning.")
        if args.challenger_run_id or args.confirm_approved:
            parser.error("Challenger approval options are valid only for deployment-approve.")
        challenger = retune_deployment_challengers(
            symbols=args.symbols or [],
            run_id=args.run_id,
        )
        print(challenger)
    elif args.mode == "deployment-approve":
        if args.raw_dir or args.expected_data_cutoff or args.expected_row_count or args.resume or args.corporate_actions_file:
            parser.error("Frozen-data and resume options are valid only for formal mode.")
        if args.strict or args.run_id:
            parser.error("--strict and --run-id are not deployment-approve options.")
        if not args.challenger_run_id:
            parser.error("deployment-approve requires --challenger-run-id.")
        if not args.confirm_approved:
            parser.error("deployment-approve requires --confirm-approved.")
        manifest = approve_deployment_challenger(
            args.challenger_run_id,
            symbols=args.symbols,
            confirmed=args.confirm_approved,
        )
        print(manifest)
    else:
        if (
            args.symbols
            or args.run_id
            or args.challenger_run_id
            or args.confirm_approved
            or args.raw_dir
            or args.expected_data_cutoff
            or args.expected_row_count
            or args.resume
            or args.corporate_actions_file
        ):
            parser.error("Approval, symbol, and run-ID options are not deployment-refresh options.")
        mapping = refresh_deployment_all(strict=args.strict)
        print(json.dumps(mapping, indent=2, sort_keys=True))
