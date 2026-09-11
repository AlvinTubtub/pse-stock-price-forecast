"""Fresh all-data refitting and versioned production-model persistence."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import torch

from config.companies import get_company
from config.model_config import DEFAULT_MODEL_CONFIG, ModelConfig, ModelId
from config.settings import SETTINGS, manila_now
from src.data.loader import load_company_history
from src.data.validator import OhlcvRecord, require_chronological_records
from src.features.regression_features import build_regression_dataset
from src.models.arima import ArimaSpecification
from src.models.lstm import LstmSpecification, validate_model_state
from src.training.train_arima import (
    ArimaEvaluationResult,
    refit_arima_for_production,
)
from src.training.train_lir import LIREvaluationResult, refit_lir_for_production
from src.training.train_lstm import (
    LstmEvaluationResult,
    refit_lstm_for_production,
)


LOGGER = logging.getLogger(__name__)
MODEL_ARTIFACT_SCHEMA_ID = "forecastph.production-model"
MODEL_ARTIFACT_SCHEMA_VERSION = 1
MODEL_IMPLEMENTATION_VERSION = "backend-from-scratch-v1"
PRINCIPAL_MODELS = (
    ModelId.LAG_REGRESSION,
    ModelId.ARIMA,
    ModelId.LSTM,
)
ARTIFACT_FORMATS: Mapping[ModelId, tuple[str, str]] = {
    ModelId.LAG_REGRESSION: ("joblib", ".joblib"),
    ModelId.ARIMA: ("joblib", ".joblib"),
    ModelId.LSTM: ("pytorch_state_dict", ".pt"),
}


class ProductionRefitError(RuntimeError):
    """Raised when fresh production refitting or persistence cannot complete."""


class ModelArtifactCompatibilityError(ValueError):
    """Raised before loading an incompatible or malformed production artifact."""


@dataclass(frozen=True, slots=True)
class ProductionSelections:
    """Hyperparameters selected before, and separate from, production refitting."""

    lir_alpha: float
    arima: ArimaSpecification
    lstm: LstmSpecification

    def __post_init__(self) -> None:
        if not math.isfinite(self.lir_alpha) or self.lir_alpha <= 0:
            raise ValueError("Selected LIR alpha must be finite and positive")


def selections_from_evaluation_results(
    lir: LIREvaluationResult,
    arima: ArimaEvaluationResult,
    lstm: LstmEvaluationResult,
) -> ProductionSelections:
    """Extract only validated hyperparameter choices, never fitted evaluation state."""

    symbols = {lir.symbol, arima.symbol, lstm.symbol}
    if len(symbols) != 1:
        raise ProductionRefitError("Evaluation selections belong to different companies")
    return ProductionSelections(
        lir_alpha=lir.tuning.chosen_alpha,
        arima=arima.tuning.selected_specification,
        lstm=lstm.tuning.selected_specification,
    )


@dataclass(frozen=True, slots=True)
class ProductionModelMetadata:
    """Compatibility and provenance metadata stored beside one model file."""

    symbol: str
    model_family: ModelId
    trained_through: date
    hyperparameters: dict[str, object]
    created_at: datetime
    data_row_count: int
    artifact_format: str
    artifact_file: str
    artifact_sha256: str
    schema_id: str = MODEL_ARTIFACT_SCHEMA_ID
    schema_version: int = MODEL_ARTIFACT_SCHEMA_VERSION
    implementation_version: str = MODEL_IMPLEMENTATION_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "implementation_version": self.implementation_version,
            "symbol": self.symbol,
            "model_family": self.model_family.value,
            "trained_through": self.trained_through.isoformat(),
            "hyperparameters": self.hyperparameters,
            "created_at": self.created_at.isoformat(),
            "data_row_count": self.data_row_count,
            "artifact_format": self.artifact_format,
            "artifact_file": self.artifact_file,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProductionModelArtifact:
    model_family: ModelId
    fitted_model: object
    metadata: ProductionModelMetadata
    model_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class ProductionRefitResult:
    symbol: str
    trained_through: date
    data_row_count: int
    artifacts: tuple[ProductionModelArtifact, ...]

    def artifact_for(self, model: ModelId) -> ProductionModelArtifact:
        try:
            return {artifact.model_family: artifact for artifact in self.artifacts}[model]
        except KeyError as exc:
            raise ProductionRefitError(f"Missing production model {model.value}") from exc


def compute_artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models_root() -> Path:
    root = (SETTINGS.artifacts_dir / "models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _symbol_directory(root: Path, symbol: str) -> Path:
    company = get_company(symbol)
    directory = (root / company.symbol).resolve()
    if directory.parent != root:
        raise ProductionRefitError("Unsafe production artifact symbol path")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_metadata(path: Path, metadata: ProductionModelMetadata) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(
            metadata.as_dict(),
            output,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output.write("\n")


def _persist_model(
    *,
    root: Path,
    symbol: str,
    model: ModelId,
    fitted_model: object,
    hyperparameters: dict[str, object],
    trained_through: date,
    data_row_count: int,
    created_at: datetime,
) -> ProductionModelArtifact:
    artifact_format, extension = ARTIFACT_FORMATS[model]
    directory = _symbol_directory(root, symbol)
    model_path = directory / f"{model.value}{extension}"
    metadata_path = directory / f"{model.value}.metadata.json"
    if model is ModelId.LSTM:
        validate_model_state(fitted_model)
        torch.save(
            {
                "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
                "specification": fitted_model.specification.as_dict(),
                "selected_epoch_count": fitted_model.epoch_count,
                "seed": fitted_model.seed,
                "training_size": fitted_model.training_size,
                "scaler": fitted_model.scaler.state_dict(),
                "model_state_dict": fitted_model.cpu_state_dict(),
            },
            model_path,
        )
    else:
        joblib.dump(fitted_model, model_path)
    metadata = ProductionModelMetadata(
        symbol=symbol,
        model_family=model,
        trained_through=trained_through,
        hyperparameters=hyperparameters,
        created_at=created_at,
        data_row_count=data_row_count,
        artifact_format=artifact_format,
        artifact_file=model_path.name,
        artifact_sha256=compute_artifact_sha256(model_path),
    )
    _write_metadata(metadata_path, metadata)
    LOGGER.info(
        "Persisted production model symbol=%s model=%s path=%s",
        symbol,
        model.value,
        model_path,
    )
    return ProductionModelArtifact(
        model_family=model,
        fitted_model=fitted_model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def validate_model_artifact_metadata(
    payload: Mapping[str, Any],
    *,
    expected_symbol: str,
    expected_model: ModelId,
) -> ProductionModelMetadata:
    """Validate schema, identity, paths, timestamps, and required hyperparameters."""

    if payload.get("schema_id") != MODEL_ARTIFACT_SCHEMA_ID:
        raise ModelArtifactCompatibilityError("Unsupported model artifact schema_id")
    if payload.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactCompatibilityError("Unsupported model artifact schema_version")
    if payload.get("implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        raise ModelArtifactCompatibilityError("Unsupported model implementation version")
    if payload.get("symbol") != expected_symbol:
        raise ModelArtifactCompatibilityError("Model artifact symbol mismatch")
    if payload.get("model_family") != expected_model.value:
        raise ModelArtifactCompatibilityError("Model artifact family mismatch")
    expected_format, expected_extension = ARTIFACT_FORMATS[expected_model]
    if payload.get("artifact_format") != expected_format:
        raise ModelArtifactCompatibilityError("Model artifact format mismatch")
    artifact_file = payload.get("artifact_file")
    if (
        not isinstance(artifact_file, str)
        or Path(artifact_file).name != artifact_file
        or artifact_file != f"{expected_model.value}{expected_extension}"
    ):
        raise ModelArtifactCompatibilityError("Unsafe or incompatible artifact filename")
    artifact_sha256 = payload.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ModelArtifactCompatibilityError("Invalid artifact SHA-256")
    try:
        trained_through = date.fromisoformat(payload["trained_through"])
        created_at = datetime.fromisoformat(payload["created_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactCompatibilityError(
            "Invalid trained-through date or creation timestamp"
        ) from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ModelArtifactCompatibilityError("Creation timestamp must be timezone-aware")
    data_row_count = payload.get("data_row_count")
    if (
        isinstance(data_row_count, bool)
        or not isinstance(data_row_count, int)
        or data_row_count < 1
    ):
        raise ModelArtifactCompatibilityError("data_row_count must be a positive integer")
    hyperparameters = payload.get("hyperparameters")
    if not isinstance(hyperparameters, dict) or not hyperparameters:
        raise ModelArtifactCompatibilityError("Model hyperparameters are missing")
    required_hyperparameters = {
        ModelId.LAG_REGRESSION: {
            "alpha",
            "feature_names",
            "pacf_selected_lags",
            "feature_config",
        },
        ModelId.ARIMA: {"order", "trend"},
        ModelId.LSTM: {
            "lookback",
            "hidden_size",
            "learning_rate",
            "batch_size",
            "selected_epoch_count",
            "seed",
        },
    }[expected_model]
    if not required_hyperparameters <= set(hyperparameters):
        raise ModelArtifactCompatibilityError("Required model hyperparameters are missing")
    return ProductionModelMetadata(
        symbol=expected_symbol,
        model_family=expected_model,
        trained_through=trained_through,
        hyperparameters=hyperparameters,
        created_at=created_at,
        data_row_count=data_row_count,
        artifact_format=expected_format,
        artifact_file=artifact_file,
        artifact_sha256=artifact_sha256,
    )


def refit_all_principal_models(
    symbol: str,
    selections: ProductionSelections,
    records: Sequence[OhlcvRecord] | None = None,
    *,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
    fresh: bool = True,
) -> ProductionRefitResult:
    """Freshly refit and persist all three models; existing files are never loaded."""

    if not fresh:
        raise ProductionRefitError(
            "Production refit is fresh-only; artifact loading belongs to inference"
        )
    company = get_company(symbol)
    source_records = (
        load_company_history(company.symbol) if records is None else tuple(records)
    )
    require_chronological_records(source_records)
    trained_through = source_records[-1].trading_date
    created_at = manila_now()
    root = _models_root()
    LOGGER.info(
        "Starting fresh production refit symbol=%s rows=%d trained_through=%s",
        company.symbol,
        len(source_records),
        trained_through,
    )

    regression_dataset = build_regression_dataset(
        source_records,
        model_config.lag_regression.features,
    )
    lir = refit_lir_for_production(
        regression_dataset,
        chosen_alpha=selections.lir_alpha,
        config=model_config.lag_regression,
    )
    arima_fit = refit_arima_for_production(
        source_records,
        selected_specification=selections.arima,
        config=model_config.arima,
    )
    lstm_fit = refit_lstm_for_production(
        source_records,
        selected_specification=selections.lstm,
        config=model_config.lstm,
    )
    fitted_by_model = {
        ModelId.LAG_REGRESSION: (
            lir,
            {
                "alpha": selections.lir_alpha,
                "feature_names": list(lir.fit_metadata.feature_names),
                "selected_features": list(lir.fit_metadata.selected_features),
                "pacf_selected_lags": list(lir.pacf_selected_lags),
                "feature_config": asdict(model_config.lag_regression.features),
            },
        ),
        ModelId.ARIMA: (arima_fit.model, selections.arima.as_dict()),
        ModelId.LSTM: (
            lstm_fit.fitted,
            {
                **selections.lstm.as_dict(),
                "selected_epoch_count": lstm_fit.fitted.epoch_count,
                "seed": lstm_fit.fitted.seed,
            },
        ),
    }
    artifacts = tuple(
        _persist_model(
            root=root,
            symbol=company.symbol,
            model=model,
            fitted_model=fitted_by_model[model][0],
            hyperparameters=fitted_by_model[model][1],
            trained_through=trained_through,
            data_row_count=len(source_records),
            created_at=created_at,
        )
        for model in PRINCIPAL_MODELS
    )
    LOGGER.info("Completed fresh production refit symbol=%s models=3", company.symbol)
    return ProductionRefitResult(
        symbol=company.symbol,
        trained_through=trained_through,
        data_row_count=len(source_records),
        artifacts=artifacts,
    )
