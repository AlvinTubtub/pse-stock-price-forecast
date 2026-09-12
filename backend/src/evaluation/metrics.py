"""Authoritative full-precision metrics for aligned OOS Close forecasts."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math

import numpy as np


class MetricError(ValueError):
    """Raised when a metric input or shared MASE scale is invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """One model's unrounded company-level OOS evaluation metrics."""

    rmse: float
    mae: float
    mase: float
    r2: float
    observations: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mase": self.mase,
            "r2": self.r2,
            "observations": self.observations,
        }


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 1:
        raise MetricError(f"{name} cannot be empty")
    if not np.isfinite(vector).all():
        raise MetricError(f"{name} must contain only finite values")
    return vector


def compute_mase_denominator(development_closes: Sequence[float]) -> float:
    """Compute the one-step naive MAE scale once from development Close levels."""

    closes = _finite_vector(development_closes, name="development_closes")
    if closes.size < 2:
        raise MetricError("MASE requires at least two development Close values")
    denominator = float(np.mean(np.abs(np.diff(closes))))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise MetricError("MASE denominator must be finite and strictly positive")
    return denominator


def compute_evaluation_metrics(
    actual_closes: Sequence[float],
    predicted_closes: Sequence[float],
    *,
    mase_denominator: float,
) -> EvaluationMetrics:
    """Compute RMSE, MAE, MASE, and standard finite R-squared without rounding."""

    actual = _finite_vector(actual_closes, name="actual_closes")
    predicted = _finite_vector(predicted_closes, name="predicted_closes")
    if actual.shape != predicted.shape:
        raise MetricError("actual_closes and predicted_closes must have equal length")
    if not math.isfinite(mase_denominator) or mase_denominator <= 0.0:
        raise MetricError("mase_denominator must be finite and strictly positive")

    residuals = predicted - actual
    squared_error_sum = float(np.sum(np.square(residuals)))
    rmse = float(math.sqrt(squared_error_sum / actual.size))
    mae = float(np.mean(np.abs(residuals)))
    total_sum_of_squares = float(np.sum(np.square(actual - np.mean(actual))))
    if total_sum_of_squares == 0.0:
        r2 = 1.0 if squared_error_sum == 0.0 else 0.0
    else:
        r2 = float(1.0 - squared_error_sum / total_sum_of_squares)
    return EvaluationMetrics(
        rmse=rmse,
        mae=mae,
        mase=float(mae / mase_denominator),
        r2=r2,
        observations=int(actual.size),
    )


def serialize_metrics(metrics: EvaluationMetrics) -> str:
    """Serialize native float values without display rounding or NaN extensions."""

    return json.dumps(
        metrics.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
