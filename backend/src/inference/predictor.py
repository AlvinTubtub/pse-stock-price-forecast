"""Compatible production-artifact loading and family-specific one-step inference."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch

from config.companies import get_company
from config.model_config import ModelId, RegressionFeatureConfig
from config.settings import SETTINGS
from src.artifacts.manager import ArtifactDirectories
from src.data.validator import OhlcvRecord, require_chronological_records
from src.features.regression_features import build_regression_origin_features
from src.models.arima import FittedArimaModel
from src.models.lstm import (
    DeltaScaler,
    FittedLstmModel,
    LstmSpecification,
    UnivariateDeltaLSTM,
    validate_model_state,
)
from src.training.production_refit import (
    ARTIFACT_FORMATS,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    ModelArtifactCompatibilityError,
    ProductionModelArtifact,
    ProductionModelMetadata,
    compute_artifact_sha256,
    validate_model_artifact_metadata,
)
from src.training.train_lir import LIRFittedModel


@dataclass(frozen=True, slots=True)
class ModelForecast:
    model: ModelId
    predicted_delta: float
    predicted_close: float


def _artifact_directory(models_root: Path, symbol: str) -> Path:
    root = models_root.resolve()
    directory = (root / symbol).resolve()
    if directory.parent != root:
        raise ModelArtifactCompatibilityError("Unsafe model artifact directory")
    return directory


def _load_metadata(
    *,
    symbol: str,
    model: ModelId,
    models_root: Path,
) -> tuple[ProductionModelMetadata, Path, Path]:
    directory = _artifact_directory(models_root, symbol)
    metadata_path = directory / f"{model.value}.metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactCompatibilityError(
            f"Cannot read compatible metadata for {symbol}/{model.value}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModelArtifactCompatibilityError("Model metadata must be a JSON object")
    metadata = validate_model_artifact_metadata(
        payload,
        expected_symbol=symbol,
        expected_model=model,
    )
    model_path = (directory / metadata.artifact_file).resolve()
    if model_path.parent != directory or not model_path.is_file():
        raise ModelArtifactCompatibilityError("Model artifact file is missing or unsafe")
    if compute_artifact_sha256(model_path) != metadata.artifact_sha256:
        raise ModelArtifactCompatibilityError("Model artifact checksum mismatch")
    return metadata, model_path, metadata_path


def _load_lstm(
    model_path: Path,
    metadata: ProductionModelMetadata,
) -> FittedLstmModel:
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelArtifactCompatibilityError("Cannot load LSTM state artifact") from exc
    if not isinstance(checkpoint, dict):
        raise ModelArtifactCompatibilityError("LSTM checkpoint must be a mapping")
    if checkpoint.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactCompatibilityError("LSTM checkpoint schema mismatch")
    hyperparameters = metadata.hyperparameters
    try:
        specification = LstmSpecification(
            lookback=int(hyperparameters["lookback"]),
            hidden_size=int(hyperparameters["hidden_size"]),
            learning_rate=float(hyperparameters["learning_rate"]),
            batch_size=int(hyperparameters["batch_size"]),
        )
        epoch_count = int(hyperparameters["selected_epoch_count"])
        seed = int(hyperparameters["seed"])
        scaler_payload = checkpoint["scaler"]
        scaler = DeltaScaler(
            mean=float(scaler_payload["mean"]),
            scale=float(scaler_payload["scale"]),
            observations=int(scaler_payload["observations"]),
        )
        network = UnivariateDeltaLSTM(specification.hidden_size)
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        fitted = FittedLstmModel(
            network=network,
            scaler=scaler,
            specification=specification,
            epoch_count=epoch_count,
            seed=seed,
            training_size=int(checkpoint["training_size"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ModelArtifactCompatibilityError("LSTM checkpoint contents are incompatible") from exc
    if checkpoint.get("specification") != specification.as_dict():
        raise ModelArtifactCompatibilityError("LSTM specification metadata mismatch")
    if checkpoint.get("selected_epoch_count") != epoch_count or checkpoint.get("seed") != seed:
        raise ModelArtifactCompatibilityError("LSTM training metadata mismatch")
    if seed != 42:
        raise ModelArtifactCompatibilityError("Production LSTM seed must be 42")
    validate_model_state(fitted)
    return fitted


def load_production_model(
    symbol: str,
    model: ModelId,
    *,
    artifacts_root: Path | None = None,
) -> ProductionModelArtifact:
    """Validate metadata/checksum first, then load one current model artifact."""

    company = get_company(symbol)
    if model not in ARTIFACT_FORMATS:
        raise ModelArtifactCompatibilityError(f"Unsupported production model {model.value}")
    root = ArtifactDirectories.from_root(
        SETTINGS.artifacts_dir if artifacts_root is None else Path(artifacts_root)
    ).models
    metadata, model_path, metadata_path = _load_metadata(
        symbol=company.symbol,
        model=model,
        models_root=root,
    )
    if model is ModelId.LSTM:
        fitted_model = _load_lstm(model_path, metadata)
    else:
        try:
            fitted_model = joblib.load(model_path)
        except Exception as exc:
            raise ModelArtifactCompatibilityError(
                f"Cannot load {model.value} model artifact"
            ) from exc
        expected_type = (
            LIRFittedModel if model is ModelId.LAG_REGRESSION else FittedArimaModel
        )
        if not isinstance(fitted_model, expected_type):
            raise ModelArtifactCompatibilityError(
                f"Loaded {model.value} artifact has an incompatible model class"
            )
        hyperparameters = metadata.hyperparameters
        try:
            if model is ModelId.LAG_REGRESSION and (
                float(hyperparameters["alpha"]) != fitted_model.fit_metadata.alpha
                or tuple(hyperparameters["feature_names"])
                != fitted_model.fit_metadata.feature_names
                or tuple(hyperparameters["pacf_selected_lags"])
                != fitted_model.pacf_selected_lags
            ):
                raise ModelArtifactCompatibilityError(
                    "LIR metadata does not match model state"
                )
            if model is ModelId.ARIMA and (
                tuple(hyperparameters["order"]) != fitted_model.specification.order
                or hyperparameters["trend"] != fitted_model.specification.trend
            ):
                raise ModelArtifactCompatibilityError(
                    "ARIMA metadata does not match model state"
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelArtifactCompatibilityError(
                f"{model.value} hyperparameter metadata is incompatible"
            ) from exc
    return ProductionModelArtifact(
        model_family=model,
        fitted_model=fitted_model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def _regression_feature_config(payload: object) -> RegressionFeatureConfig:
    if not isinstance(payload, Mapping):
        raise ModelArtifactCompatibilityError("LIR feature configuration is missing")
    values = dict(payload)
    tuple_fields = (
        "return_lags",
        "rolling_return_windows",
        "volume_windows",
        "raw_price_lags",
    )
    try:
        for field in tuple_fields:
            if field in values:
                values[field] = tuple(values[field])
        return RegressionFeatureConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ModelArtifactCompatibilityError(
            "LIR feature configuration is incompatible"
        ) from exc


def validate_artifact_against_history(
    artifact: ProductionModelArtifact,
    records: Sequence[OhlcvRecord],
) -> tuple[OhlcvRecord, ...]:
    """Require history to contain the unchanged model-training boundary or newer rows."""

    history = tuple(records)
    require_chronological_records(history)
    metadata = artifact.metadata
    if len(history) < metadata.data_row_count:
        raise ModelArtifactCompatibilityError(
            "Current history ends before the model training boundary"
        )
    if history[metadata.data_row_count - 1].trading_date != metadata.trained_through:
        raise ModelArtifactCompatibilityError(
            "Model trained-through date does not match the historical boundary"
        )
    return history


def predict_with_production_model(
    artifact: ProductionModelArtifact,
    records: Sequence[OhlcvRecord],
) -> ModelForecast:
    """Generate one next-session Close forecast from one compatible fresh model."""

    history = validate_artifact_against_history(artifact, records)
    origin_close = history[-1].close
    model = artifact.model_family
    if model is ModelId.LAG_REGRESSION:
        fitted = artifact.fitted_model
        if not isinstance(fitted, LIRFittedModel):
            raise ModelArtifactCompatibilityError("Invalid in-memory LIR model")
        feature_config = _regression_feature_config(
            artifact.metadata.hyperparameters.get("feature_config")
        )
        origin = build_regression_origin_features(history, feature_config)
        by_name = dict(zip(origin.feature_names, origin.feature_values, strict=True))
        try:
            matrix = np.asarray(
                [[by_name[name] for name in fitted.fit_metadata.feature_names]],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise ModelArtifactCompatibilityError(
                f"Missing LIR inference feature {exc.args[0]}"
            ) from exc
        predicted_delta = float(fitted.model.predict_delta(matrix)[0])
        predicted_close = origin_close + predicted_delta
    elif model is ModelId.ARIMA:
        fitted = artifact.fitted_model
        if not isinstance(fitted, FittedArimaModel):
            raise ModelArtifactCompatibilityError("Invalid in-memory ARIMA model")
        for observation in history[artifact.metadata.data_row_count :]:
            fitted = fitted.append_actual(observation.close)
        predicted_close = fitted.forecast_one()
        predicted_delta = predicted_close - origin_close
    elif model is ModelId.LSTM:
        fitted = artifact.fitted_model
        if not isinstance(fitted, FittedLstmModel):
            raise ModelArtifactCompatibilityError("Invalid in-memory LSTM model")
        closes = np.asarray([record.close for record in history], dtype=np.float64)
        deltas = np.diff(closes)
        if len(deltas) < fitted.specification.lookback:
            raise ModelArtifactCompatibilityError("Insufficient history for LSTM lookback")
        input_deltas = deltas[-fitted.specification.lookback :]
        predicted_delta = float(fitted.predict_delta_sequences(input_deltas)[0])
        predicted_close = origin_close + predicted_delta
    else:
        raise ModelArtifactCompatibilityError(f"Unsupported model family {model.value}")
    if not math.isfinite(predicted_delta) or not math.isfinite(predicted_close):
        raise RuntimeError(f"{model.value} produced a non-finite production forecast")
    return ModelForecast(
        model=model,
        predicted_delta=predicted_delta,
        predicted_close=predicted_close,
    )
