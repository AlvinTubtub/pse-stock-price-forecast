"""One complete fresh-training lifecycle and persisted-inference preparation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
from pathlib import Path
from typing import Any, Final

import joblib

from config.companies import get_company
from config.model_config import DEFAULT_MODEL_CONFIG, ModelConfig, ModelId
from config.settings import SETTINGS, as_manila_time, manila_now
from src.artifacts.manager import ArtifactManager
from src.data.calendar import PSETradingCalendar
from src.data.split import build_company_evaluation_plan
from src.data.validator import OhlcvRecord, require_chronological_records
from src.evaluation.evaluator import (
    CompanyEvaluation,
    ModelPredictionOutput,
    evaluate_prediction_outputs,
)
from src.export.frontend_exporter import CompanyFrontendArtifacts
from src.features.regression_features import build_regression_dataset
from src.inference.next_day import (
    CompanyNextDayForecast,
    predict_next_day_from_artifacts,
    predict_next_day_from_fresh_refit,
)
from src.training.production_refit import (
    ProductionRefitResult,
    ProductionSelections,
    compute_artifact_sha256,
    refit_all_principal_models,
    selections_from_evaluation_results,
)
from src.training.train_arima import train_arima_for_evaluation
from src.training.train_lir import train_lir_for_evaluation
from src.training.train_lstm import train_lstm_for_evaluation


LOGGER = logging.getLogger(__name__)
EVALUATION_ARTIFACT_SCHEMA_ID: Final[str] = "forecastph.company-evaluation"
EVALUATION_ARTIFACT_SCHEMA_VERSION: Final[int] = 1
FORECAST_ARTIFACT_SCHEMA_ID: Final[str] = "forecastph.next-day-forecast"
FORECAST_ARTIFACT_SCHEMA_VERSION: Final[int] = 1


class OrchestrationError(RuntimeError):
    """Raised when a lifecycle or its new artifacts are incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class CompanyTrainingResult:
    symbol: str
    frontend_artifacts: CompanyFrontendArtifacts
    selections: ProductionSelections
    production_refit: ProductionRefitResult
    evaluation_path: Path
    forecast_path: Path


def _artifact_root(artifacts_root: Path | None) -> Path:
    return SETTINGS.artifacts_dir if artifacts_root is None else Path(artifacts_root)


def _symbol_directory(root: Path, symbol: str, *, create: bool = True) -> Path:
    company = get_company(symbol)
    resolved_root = root.resolve()
    directory = (resolved_root / company.symbol).resolve()
    if directory.parent != resolved_root:
        raise OrchestrationError("Unsafe company artifact directory")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    with temporary.open("r", encoding="utf-8") as source:
        decoded = json.load(source)
    if not isinstance(decoded, dict):
        temporary.unlink(missing_ok=True)
        raise OrchestrationError("Generated artifact metadata must be a JSON object")
    temporary.replace(path)


def persist_company_evaluation(
    evaluation: CompanyEvaluation,
    records: Sequence[OhlcvRecord],
    *,
    artifacts_root: Path | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Persist one new canonical evaluation with checksum and source boundary."""

    history = tuple(records)
    require_chronological_records(history)
    company = get_company(evaluation.symbol)
    if evaluation.backtest.symbol != company.symbol:
        raise OrchestrationError("Evaluation symbol is inconsistent")
    directories = ArtifactManager(_artifact_root(artifacts_root)).ensure_directories()
    directory = _symbol_directory(directories.evaluations, company.symbol)
    evaluation_path = directory / "evaluation.joblib"
    temporary = directory / ".evaluation.joblib.tmp"
    joblib.dump(evaluation, temporary)
    temporary.replace(evaluation_path)
    metadata_path = directory / "evaluation.metadata.json"
    try:
        timestamp = manila_now() if created_at is None else as_manila_time(created_at)
    except ValueError as exc:
        raise OrchestrationError(
            "Evaluation artifact timestamp must be timezone-aware"
        ) from exc
    payload: dict[str, object] = {
        "schema_id": EVALUATION_ARTIFACT_SCHEMA_ID,
        "schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
        "symbol": company.symbol,
        "created_at": timestamp.isoformat(),
        "data_row_count": len(history),
        "data_through": history[-1].trading_date.isoformat(),
        "evaluation_start": evaluation.backtest.target_dates[0].isoformat(),
        "evaluation_end": evaluation.backtest.target_dates[-1].isoformat(),
        "evaluation_count": len(evaluation.backtest.target_dates),
        "source": "chronological_out_of_sample_evaluation",
        "artifact_file": evaluation_path.name,
        "artifact_sha256": compute_artifact_sha256(evaluation_path),
    }
    _atomic_json(metadata_path, payload)
    LOGGER.info(
        "Persisted canonical evaluation symbol=%s dates=%d path=%s",
        company.symbol,
        len(evaluation.backtest.target_dates),
        evaluation_path,
    )
    return evaluation_path


def _evaluation_metadata(payload: Mapping[str, Any], symbol: str) -> dict[str, object]:
    required = {
        "schema_id",
        "schema_version",
        "symbol",
        "created_at",
        "data_row_count",
        "data_through",
        "evaluation_start",
        "evaluation_end",
        "evaluation_count",
        "source",
        "artifact_file",
        "artifact_sha256",
    }
    if not required <= set(payload):
        raise OrchestrationError("Evaluation artifact metadata is incomplete")
    if (
        payload["schema_id"] != EVALUATION_ARTIFACT_SCHEMA_ID
        or payload["schema_version"] != EVALUATION_ARTIFACT_SCHEMA_VERSION
        or payload["symbol"] != symbol
        or payload["source"] != "chronological_out_of_sample_evaluation"
        or payload["artifact_file"] != "evaluation.joblib"
    ):
        raise OrchestrationError("Evaluation artifact metadata is incompatible")
    try:
        created_at = datetime.fromisoformat(payload["created_at"])
        date.fromisoformat(payload["data_through"])
        date.fromisoformat(payload["evaluation_start"])
        date.fromisoformat(payload["evaluation_end"])
    except (TypeError, ValueError) as exc:
        raise OrchestrationError("Evaluation artifact dates are invalid") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise OrchestrationError("Evaluation created_at must be timezone-aware")
    for field in ("data_row_count", "evaluation_count"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise OrchestrationError(f"Evaluation {field} must be positive")
    checksum = payload["artifact_sha256"]
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise OrchestrationError("Evaluation artifact checksum is invalid")
    return dict(payload)


def load_company_evaluation(
    symbol: str,
    records: Sequence[OhlcvRecord],
    *,
    artifacts_root: Path | None = None,
) -> CompanyEvaluation:
    """Load only a compatible new evaluation artifact for frontend reuse."""

    company = get_company(symbol)
    history = tuple(records)
    require_chronological_records(history)
    evaluations_root = ArtifactManager(
        _artifact_root(artifacts_root)
    ).directories.evaluations
    directory = _symbol_directory(evaluations_root, company.symbol, create=False)
    metadata_path = directory / "evaluation.metadata.json"
    try:
        with metadata_path.open("r", encoding="utf-8") as source:
            raw_payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestrationError(
            f"Cannot load evaluation metadata for {company.symbol}"
        ) from exc
    if not isinstance(raw_payload, dict):
        raise OrchestrationError("Evaluation artifact metadata must be an object")
    payload = _evaluation_metadata(raw_payload, company.symbol)
    row_count = payload["data_row_count"]
    trained_through = date.fromisoformat(payload["data_through"])
    if len(history) < row_count or history[row_count - 1].trading_date != trained_through:
        raise OrchestrationError("Raw history does not contain the evaluation boundary")
    evaluation_path = directory / payload["artifact_file"]
    if not evaluation_path.is_file():
        raise OrchestrationError("Evaluation artifact is missing")
    if compute_artifact_sha256(evaluation_path) != payload["artifact_sha256"]:
        raise OrchestrationError("Evaluation artifact checksum mismatch")
    try:
        evaluation = joblib.load(evaluation_path)
    except Exception as exc:
        raise OrchestrationError("Cannot load canonical evaluation artifact") from exc
    if not isinstance(evaluation, CompanyEvaluation) or evaluation.symbol != company.symbol:
        raise OrchestrationError("Loaded evaluation has an incompatible type or symbol")
    if (
        len(evaluation.backtest.target_dates) != payload["evaluation_count"]
        or evaluation.backtest.target_dates[0].isoformat() != payload["evaluation_start"]
        or evaluation.backtest.target_dates[-1].isoformat() != payload["evaluation_end"]
    ):
        raise OrchestrationError("Loaded evaluation disagrees with its metadata")
    return evaluation


def persist_next_day_forecast(
    forecast: CompanyNextDayForecast,
    *,
    artifacts_root: Path | None = None,
) -> Path:
    """Persist the latest new three-model forecast as an observable JSON artifact."""

    company = get_company(forecast.symbol)
    predictions = tuple(forecast.predictions)
    models = {prediction.model for prediction in predictions}
    required = {ModelId.LAG_REGRESSION, ModelId.ARIMA, ModelId.LSTM}
    if len(predictions) != 3 or models != required:
        raise OrchestrationError("Forecast artifact requires exactly three principal models")
    for prediction in predictions:
        if (
            prediction.symbol != company.symbol
            or prediction.origin_date != forecast.origin_date
            or prediction.forecast_for != forecast.forecast_for
            or prediction.forecast_for <= prediction.origin_date
            or prediction.inference_at.tzinfo is None
            or prediction.inference_at.utcoffset() is None
        ):
            raise OrchestrationError("Forecast prediction metadata is inconsistent")
    directory = ArtifactManager(_artifact_root(artifacts_root)).ensure_directories().forecasts
    path = directory / f"{company.symbol}.json"
    payload = {
        "schema_id": FORECAST_ARTIFACT_SCHEMA_ID,
        "schema_version": FORECAST_ARTIFACT_SCHEMA_VERSION,
        **forecast.as_dict(),
    }
    _atomic_json(path, payload)
    LOGGER.info("Persisted next-day forecast symbol=%s path=%s", company.symbol, path)
    return path


def train_company_lifecycle(
    symbol: str,
    records: Sequence[OhlcvRecord],
    *,
    calendar: PSETradingCalendar,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    artifacts_root: Path | None = None,
) -> CompanyTrainingResult:
    """Tune, evaluate, refit, forecast, and persist one company from raw data."""

    company = get_company(symbol)
    history = tuple(records)
    require_chronological_records(history)
    LOGGER.info("Starting complete fresh lifecycle symbol=%s rows=%d", company.symbol, len(history))
    plan = build_company_evaluation_plan(company.symbol, history, model_config=model_config)
    regression_dataset = build_regression_dataset(
        history, model_config.lag_regression.features
    )
    lir = train_lir_for_evaluation(
        regression_dataset, plan, config=model_config.lag_regression
    )
    arima = train_arima_for_evaluation(history, plan, config=model_config.arima)
    lstm = train_lstm_for_evaluation(history, plan, config=model_config.lstm)
    evaluation = evaluate_prediction_outputs(
        plan,
        (
            ModelPredictionOutput(
                symbol=lir.symbol,
                model=ModelId.LAG_REGRESSION,
                target_dates=lir.target_dates,
                actual_closes=lir.actual_closes,
                predicted_closes=lir.predicted_closes,
            ),
            ModelPredictionOutput(
                symbol=arima.symbol,
                model=ModelId.ARIMA,
                target_dates=arima.target_dates,
                actual_closes=arima.actual_closes,
                predicted_closes=arima.predicted_closes,
            ),
            ModelPredictionOutput(
                symbol=lstm.symbol,
                model=ModelId.LSTM,
                target_dates=lstm.target_dates,
                actual_closes=lstm.actual_closes,
                predicted_closes=lstm.predicted_closes,
            ),
        ),
    )
    selections = selections_from_evaluation_results(lir, arima, lstm)
    production_refit = refit_all_principal_models(
        company.symbol,
        selections,
        history,
        model_config=model_config,
        fresh=True,
    )
    forecast = predict_next_day_from_fresh_refit(
        production_refit,
        history,
        calendar=calendar,
    )
    evaluation_path = persist_company_evaluation(
        evaluation,
        history,
        artifacts_root=artifacts_root,
    )
    forecast_path = persist_next_day_forecast(
        forecast,
        artifacts_root=artifacts_root,
    )
    frontend_artifacts = CompanyFrontendArtifacts(
        records=history,
        evaluation=evaluation,
        next_day_forecast=forecast,
    )
    LOGGER.info("Completed complete fresh lifecycle symbol=%s", company.symbol)
    return CompanyTrainingResult(
        symbol=company.symbol,
        frontend_artifacts=frontend_artifacts,
        selections=selections,
        production_refit=production_refit,
        evaluation_path=evaluation_path,
        forecast_path=forecast_path,
    )


def forecast_company_from_artifacts(
    symbol: str,
    records: Sequence[OhlcvRecord],
    *,
    calendar: PSETradingCalendar,
    artifacts_root: Path | None = None,
) -> CompanyFrontendArtifacts:
    """Load new persisted models/evaluation and forecast without tuning or fitting."""

    company = get_company(symbol)
    history = tuple(records)
    require_chronological_records(history)
    evaluation = load_company_evaluation(
        company.symbol,
        history,
        artifacts_root=artifacts_root,
    )
    forecast = predict_next_day_from_artifacts(
        company.symbol,
        history,
        calendar=calendar,
    )
    persist_next_day_forecast(forecast, artifacts_root=artifacts_root)
    return CompanyFrontendArtifacts(
        records=history,
        evaluation=evaluation,
        next_day_forecast=forecast,
    )
