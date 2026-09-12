"""Fail-closed validation for the Phase 14 production artifact package."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Final

import torch

from config.companies import COMPANIES
from config.model_config import ModelId
from config.settings import SETTINGS, as_manila_time, manila_now
from src.artifacts.io import atomic_write_json
from src.data.loader import load_company_history
from src.evaluation.backtest import REQUIRED_EVALUATION_MODELS
from src.inference.predictor import (
    load_production_model,
    validate_artifact_against_history,
)
from src.training.orchestration import load_company_evaluation
from src.training.production_refit import PRINCIPAL_MODELS
from src.training.train_arima import (
    ARIMA_EVALUATION_SCHEMA_ID,
    ARIMA_EVALUATION_SCHEMA_VERSION,
)
from src.training.train_lir import (
    LIR_EVALUATION_SCHEMA_ID,
    LIR_EVALUATION_SCHEMA_VERSION,
)
from src.training.train_lstm import (
    LSTM_EVALUATION_SCHEMA_ID,
    LSTM_EVALUATION_SCHEMA_VERSION,
)


LOGGER = logging.getLogger(__name__)
PRODUCTION_MANIFEST_SCHEMA_ID: Final[str] = "forecastph.production-artifact-manifest"
PRODUCTION_MANIFEST_SCHEMA_VERSION: Final[int] = 1
PRODUCTION_ARTIFACT_GENERATION: Final[str] = "phase14"
PRODUCTION_ARTIFACT_NAME: Final[str] = "forecastph-backend-artifacts"
PRODUCTION_WORKFLOW_FILE: Final[str] = "train_models.yml"
PRODUCTION_MANIFEST_FILENAME: Final[str] = "production_manifest.json"
MANIFEST_MODEL_FAMILIES: Final[tuple[str, ...]] = (
    "lag_regression",
    "arima",
    "lstm",
)
_SHA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


class ProductionArtifactValidationError(RuntimeError):
    """Raised when a Phase 14 package is missing, malformed, or incompatible."""


@dataclass(frozen=True, slots=True)
class ProductionManifest:
    run_id: str
    source_commit: str
    source_branch: str
    created_at: datetime
    training_data_through: date
    symbols: tuple[str, ...]
    model_families: tuple[str, ...] = MANIFEST_MODEL_FAMILIES
    schema_id: str = PRODUCTION_MANIFEST_SCHEMA_ID
    schema_version: int = PRODUCTION_MANIFEST_SCHEMA_VERSION
    artifact_generation: str = PRODUCTION_ARTIFACT_GENERATION
    artifact_name: str = PRODUCTION_ARTIFACT_NAME
    workflow_file: str = PRODUCTION_WORKFLOW_FILE

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "artifact_generation": self.artifact_generation,
            "artifact_name": self.artifact_name,
            "workflow_file": self.workflow_file,
            "run_id": self.run_id,
            "source_commit": self.source_commit,
            "source_branch": self.source_branch,
            "created_at": self.created_at.isoformat(),
            "training_data_through": self.training_data_through.isoformat(),
            "symbols": list(self.symbols),
            "symbol_count": len(self.symbols),
            "model_families": list(self.model_families),
        }


@dataclass(frozen=True, slots=True)
class ProductionArtifactReport:
    artifacts_root: Path
    symbols: tuple[str, ...]
    training_data_through: date
    validated_file_count: int
    manifest: ProductionManifest | None


def configured_symbols() -> tuple[str, ...]:
    """Return the exact, sorted universe encoded in a production manifest."""

    return tuple(sorted(company.symbol for company in COMPANIES))


def _reject_nonstandard_number(value: str) -> None:
    raise ProductionArtifactValidationError(
        f"Non-standard JSON number is forbidden: {value}"
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source, parse_constant=_reject_nonstandard_number)
    except ProductionArtifactValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionArtifactValidationError(f"Cannot read valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProductionArtifactValidationError(f"JSON artifact must be an object: {path}")
    return payload


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProductionArtifactValidationError(f"Manifest field {field} is invalid")
    return value


def parse_production_manifest(payload: Mapping[str, Any]) -> ProductionManifest:
    """Strictly parse identity, provenance, universe, and model-family fields."""

    exact_fields = {
        "schema_id",
        "schema_version",
        "artifact_generation",
        "artifact_name",
        "workflow_file",
        "run_id",
        "source_commit",
        "source_branch",
        "created_at",
        "training_data_through",
        "symbols",
        "symbol_count",
        "model_families",
    }
    if set(payload) != exact_fields:
        missing = sorted(exact_fields - set(payload))
        extra = sorted(set(payload) - exact_fields)
        raise ProductionArtifactValidationError(
            f"Production manifest fields mismatch; missing={missing} extra={extra}"
        )
    if (
        payload["schema_id"] != PRODUCTION_MANIFEST_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != PRODUCTION_MANIFEST_SCHEMA_VERSION
        or payload["artifact_generation"] != PRODUCTION_ARTIFACT_GENERATION
        or payload["artifact_name"] != PRODUCTION_ARTIFACT_NAME
        or payload["workflow_file"] != PRODUCTION_WORKFLOW_FILE
    ):
        raise ProductionArtifactValidationError(
            "Production manifest identity, schema, or generation is incompatible"
        )
    run_id = _required_string(payload, "run_id")
    source_commit = _required_string(payload, "source_commit").lower()
    source_branch = _required_string(payload, "source_branch")
    if _SHA_PATTERN.fullmatch(source_commit) is None:
        raise ProductionArtifactValidationError("Manifest source_commit must be a full Git SHA")
    if not run_id.isdecimal():
        raise ProductionArtifactValidationError("Manifest run_id must be a GitHub Actions run ID")
    try:
        created_at = datetime.fromisoformat(payload["created_at"])
        training_data_through = date.fromisoformat(payload["training_data_through"])
    except (TypeError, ValueError) as exc:
        raise ProductionArtifactValidationError("Manifest date fields are invalid") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ProductionArtifactValidationError("Manifest created_at must be timezone-aware")
    if training_data_through.isoformat() != payload["training_data_through"]:
        raise ProductionArtifactValidationError(
            "Manifest training_data_through must use YYYY-MM-DD"
        )
    symbols_value = payload["symbols"]
    model_families_value = payload["model_families"]
    if not isinstance(symbols_value, list) or any(
        not isinstance(symbol, str) for symbol in symbols_value
    ):
        raise ProductionArtifactValidationError("Manifest symbols must be a string array")
    if not isinstance(model_families_value, list) or any(
        not isinstance(model, str) for model in model_families_value
    ):
        raise ProductionArtifactValidationError(
            "Manifest model_families must be a string array"
        )
    symbols = tuple(symbols_value)
    model_families = tuple(model_families_value)
    if symbols != configured_symbols():
        raise ProductionArtifactValidationError(
            "Manifest symbol universe does not match configured companies"
        )
    if (
        type(payload["symbol_count"]) is not int
        or payload["symbol_count"] != len(symbols)
    ):
        raise ProductionArtifactValidationError("Manifest symbol_count is invalid")
    if model_families != MANIFEST_MODEL_FAMILIES:
        raise ProductionArtifactValidationError(
            "Manifest model families do not match the three principal models"
        )
    return ProductionManifest(
        run_id=run_id,
        source_commit=source_commit,
        source_branch=source_branch,
        created_at=created_at,
        training_data_through=training_data_through,
        symbols=symbols,
        model_families=model_families,
    )


def load_production_manifest(
    artifacts_root: Path = SETTINGS.artifacts_dir,
    *,
    expected_run_id: str | None = None,
    expected_source_commit: str | None = None,
    expected_source_branch: str | None = None,
) -> ProductionManifest:
    """Load a manifest and optionally bind it to the selected Actions run."""

    path = Path(artifacts_root) / PRODUCTION_MANIFEST_FILENAME
    manifest = parse_production_manifest(_load_json_object(path))
    expected = {
        "run_id": expected_run_id,
        "source_commit": expected_source_commit.lower() if expected_source_commit else None,
        "source_branch": expected_source_branch,
    }
    actual = {
        "run_id": manifest.run_id,
        "source_commit": manifest.source_commit,
        "source_branch": manifest.source_branch,
    }
    for field, expected_value in expected.items():
        if expected_value is not None and actual[field] != expected_value:
            raise ProductionArtifactValidationError(
                f"Manifest {field} does not match the selected training run"
            )
    return manifest


def expected_artifact_paths(
    artifacts_root: Path,
    *,
    require_forecasts: bool,
) -> tuple[Path, ...]:
    """List every current file required for complete all-company operation."""

    root = Path(artifacts_root)
    paths: list[Path] = []
    for symbol in configured_symbols():
        paths.extend(
            (
                root / "evaluations" / "lir" / f"{symbol}.json",
                root / "evaluations" / "arima" / f"{symbol}.json",
                root / "evaluations" / "lstm" / f"{symbol}.json",
                root / "evaluations" / "lstm" / f"{symbol}.pt",
                root / "evaluations" / symbol / "evaluation.joblib",
                root / "evaluations" / symbol / "evaluation.metadata.json",
            )
        )
        for model in PRINCIPAL_MODELS:
            extension = ".pt" if model is ModelId.LSTM else ".joblib"
            paths.extend(
                (
                    root / "models" / symbol / f"{model.value}{extension}",
                    root / "models" / symbol / f"{model.value}.metadata.json",
                )
            )
        if require_forecasts:
            paths.append(root / "forecasts" / f"{symbol}.json")
    return tuple(paths)


def _require_finite_metrics(evaluation: object, symbol: str) -> None:
    backtest = evaluation.backtest
    expected_dates = evaluation.plan.evaluation_target_dates
    if backtest.target_dates != expected_dates or not expected_dates:
        raise ProductionArtifactValidationError(
            f"Canonical evaluation target dates are invalid for {symbol}"
        )
    if {model for model, _ in backtest.records_by_model} != set(
        REQUIRED_EVALUATION_MODELS
    ):
        raise ProductionArtifactValidationError(
            f"Canonical evaluation model set is incomplete for {symbol}"
        )
    for model in REQUIRED_EVALUATION_MODELS:
        records = backtest.records_for(model)
        if tuple(record.target_date for record in records) != expected_dates:
            raise ProductionArtifactValidationError(
                f"Canonical evaluation dates are misaligned for {symbol}/{model.value}"
            )
        metrics = evaluation.metrics_for(model)
        if not all(
            math.isfinite(value)
            for value in (metrics.rmse, metrics.mae, metrics.mase, metrics.r2)
        ):
            raise ProductionArtifactValidationError(
                f"Canonical evaluation metrics are non-finite for {symbol}/{model.value}"
            )


def _validate_model_evidence(
    artifacts_root: Path,
    symbol: str,
    expected_dates: tuple[date, ...],
) -> None:
    definitions = (
        ("lir", LIR_EVALUATION_SCHEMA_ID, LIR_EVALUATION_SCHEMA_VERSION),
        ("arima", ARIMA_EVALUATION_SCHEMA_ID, ARIMA_EVALUATION_SCHEMA_VERSION),
        ("lstm", LSTM_EVALUATION_SCHEMA_ID, LSTM_EVALUATION_SCHEMA_VERSION),
    )
    expected_date_strings = [value.isoformat() for value in expected_dates]
    for directory, schema_id, schema_version in definitions:
        path = artifacts_root / "evaluations" / directory / f"{symbol}.json"
        payload = _load_json_object(path)
        if (
            payload.get("schema_id") != schema_id
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != schema_version
            or payload.get("symbol") != symbol
            or payload.get("evaluation_target_dates") != expected_date_strings
        ):
            raise ProductionArtifactValidationError(
                f"Model-specific evaluation evidence is incompatible: {path}"
            )
    lstm_metadata = _load_json_object(
        artifacts_root / "evaluations" / "lstm" / f"{symbol}.json"
    )
    if lstm_metadata.get("model_state_file") != f"{symbol}.pt":
        raise ProductionArtifactValidationError(
            f"LSTM evaluation state filename is incompatible for {symbol}"
        )
    state_path = artifacts_root / "evaluations" / "lstm" / f"{symbol}.pt"
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ProductionArtifactValidationError(
            f"Cannot load LSTM evaluation state for {symbol}"
        ) from exc
    if (
        not isinstance(state, dict)
        or state.get("schema_id") != LSTM_EVALUATION_SCHEMA_ID
        or type(state.get("schema_version")) is not int
        or state.get("schema_version") != LSTM_EVALUATION_SCHEMA_VERSION
        or not state.get("model_state_dict")
    ):
        raise ProductionArtifactValidationError(
            f"LSTM evaluation state is incompatible for {symbol}"
        )


def _validate_forecast(artifacts_root: Path, symbol: str, latest_date: date) -> None:
    path = artifacts_root / "forecasts" / f"{symbol}.json"
    payload = _load_json_object(path)
    predictions = payload.get("predictions")
    if (
        payload.get("schema_id") != "forecastph.next-day-forecast"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("symbol") != symbol
        or payload.get("origin_date") != latest_date.isoformat()
        or not isinstance(predictions, list)
        or len(predictions) != len(PRINCIPAL_MODELS)
    ):
        raise ProductionArtifactValidationError(f"Forecast artifact is incompatible: {path}")
    expected_models = {model.value for model in PRINCIPAL_MODELS}
    if {item.get("model") for item in predictions if isinstance(item, dict)} != expected_models:
        raise ProductionArtifactValidationError(
            f"Forecast model set is incomplete for {symbol}"
        )
    forecast_for = payload.get("forecastFor")
    if not isinstance(forecast_for, str):
        raise ProductionArtifactValidationError(f"Forecast date is missing for {symbol}")
    try:
        parsed_forecast_date = date.fromisoformat(forecast_for)
        if parsed_forecast_date.isoformat() != forecast_for:
            raise ProductionArtifactValidationError(
                f"Forecast date must use YYYY-MM-DD for {symbol}"
            )
        if parsed_forecast_date <= latest_date or parsed_forecast_date.weekday() >= 5:
            raise ProductionArtifactValidationError(
                f"Forecast date is not a future weekday for {symbol}"
            )
    except ValueError as exc:
        raise ProductionArtifactValidationError(
            f"Forecast date is invalid for {symbol}"
        ) from exc
    for prediction in predictions:
        if not isinstance(prediction, dict) or any(
            not isinstance(prediction.get(field), (int, float))
            or isinstance(prediction.get(field), bool)
            or not math.isfinite(float(prediction[field]))
            for field in ("origin_close", "predicted_delta", "predicted_close")
        ):
            raise ProductionArtifactValidationError(
                f"Forecast numeric values are invalid for {symbol}"
            )
        if (
            prediction.get("symbol") != symbol
            or prediction.get("origin_date") != latest_date.isoformat()
            or prediction.get("forecastFor") != forecast_for
        ):
            raise ProductionArtifactValidationError(
                f"Forecast prediction identity is inconsistent for {symbol}"
            )
        try:
            inference_at = datetime.fromisoformat(prediction["inference_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionArtifactValidationError(
                f"Forecast inference timestamp is invalid for {symbol}"
            ) from exc
        if inference_at.tzinfo is None or inference_at.utcoffset() is None:
            raise ProductionArtifactValidationError(
                f"Forecast inference timestamp must be timezone-aware for {symbol}"
            )
        if not math.isclose(
            float(prediction["predicted_close"]),
            float(prediction["origin_close"]) + float(prediction["predicted_delta"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ProductionArtifactValidationError(
                f"Forecast Close reconstruction is invalid for {symbol}"
            )


def validate_production_artifacts(
    artifacts_root: Path = SETTINGS.artifacts_dir,
    *,
    require_manifest: bool = True,
    require_forecasts: bool = False,
    require_current_data: bool = False,
    expected_run_id: str | None = None,
    expected_source_commit: str | None = None,
    expected_source_branch: str | None = None,
) -> ProductionArtifactReport:
    """Validate the exact universe and every artifact required by persisted inference."""

    root = Path(artifacts_root).resolve()
    manifest = (
        load_production_manifest(
            root,
            expected_run_id=expected_run_id,
            expected_source_commit=expected_source_commit,
            expected_source_branch=expected_source_branch,
        )
        if require_manifest
        else None
    )
    expected_paths = expected_artifact_paths(root, require_forecasts=require_forecasts)
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        relative = [str(path.relative_to(root)) for path in missing]
        raise ProductionArtifactValidationError(
            f"Production artifact package is incomplete; missing={relative}"
        )

    trained_through_dates: set[date] = set()
    validated_files = 0
    for symbol in configured_symbols():
        history = load_company_history(symbol)
        evaluation = load_company_evaluation(symbol, history, artifacts_root=root)
        _require_finite_metrics(evaluation, symbol)
        _validate_model_evidence(root, symbol, evaluation.backtest.target_dates)
        trained_dates: set[date] = set()
        row_counts: set[int] = set()
        for model in PRINCIPAL_MODELS:
            artifact = load_production_model(symbol, model, artifacts_root=root)
            validate_artifact_against_history(artifact, history)
            trained_dates.add(artifact.metadata.trained_through)
            row_counts.add(artifact.metadata.data_row_count)
        if len(trained_dates) != 1 or len(row_counts) != 1:
            raise ProductionArtifactValidationError(
                f"Production model training boundaries disagree for {symbol}"
            )
        trained_through = next(iter(trained_dates))
        row_count = next(iter(row_counts))
        if require_current_data and (
            trained_through != history[-1].trading_date or row_count != len(history)
        ):
            raise ProductionArtifactValidationError(
                f"Production models are not fitted through current raw data for {symbol}"
            )
        trained_through_dates.add(trained_through)
        if require_forecasts:
            _validate_forecast(root, symbol, history[-1].trading_date)
        validated_files += 13 if require_forecasts else 12

    if len(trained_through_dates) != 1:
        raise ProductionArtifactValidationError(
            "Production training boundary differs across configured companies"
        )
    training_data_through = next(iter(trained_through_dates))
    if manifest is not None and manifest.training_data_through != training_data_through:
        raise ProductionArtifactValidationError(
            "Manifest training_data_through disagrees with production models"
        )
    LOGGER.info(
        "Validated Phase 14 production artifacts symbols=%d files=%d trained_through=%s",
        len(configured_symbols()),
        validated_files,
        training_data_through,
    )
    return ProductionArtifactReport(
        artifacts_root=root,
        symbols=configured_symbols(),
        training_data_through=training_data_through,
        validated_file_count=validated_files,
        manifest=manifest,
    )


def create_production_manifest(
    report: ProductionArtifactReport,
    *,
    run_id: str,
    source_commit: str,
    source_branch: str,
    created_at: datetime | None = None,
) -> Path:
    """Atomically create the manifest only after artifact validation succeeds."""

    timestamp = manila_now() if created_at is None else as_manila_time(created_at)
    manifest = ProductionManifest(
        run_id=run_id,
        source_commit=source_commit.lower(),
        source_branch=source_branch,
        created_at=timestamp,
        training_data_through=report.training_data_through,
        symbols=report.symbols,
    )
    validated = parse_production_manifest(manifest.as_dict())
    path = report.artifacts_root / PRODUCTION_MANIFEST_FILENAME
    atomic_write_json(path, validated.as_dict())
    LOGGER.info("Created Phase 14 production manifest path=%s run_id=%s", path, run_id)
    return path
