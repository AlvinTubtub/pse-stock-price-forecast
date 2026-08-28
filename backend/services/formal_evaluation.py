"""Strict, date-indexed formal hold-out forecast validation.

This module intentionally has no forecasting-model imports, so formal data
contracts can be unit-tested without loading optional training dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from services.time_series_cv import FormalEvaluationPlan

FORMAL_MODEL_KEYS = ("lag_reg", "arima", "lstm", "naive")
FORMAL_FORECAST_COLUMNS = (
    "symbol", "model", "origin_date", "target_date", "actual_close", "predicted_close", "error",
)


class FormalHoldoutAlignmentError(ValueError):
    """Raised when a model does not forecast the exact frozen hold-out window."""


def validate_formal_holdout_alignment(
    forecasts_by_model: dict[str, pd.DataFrame],
    plan: FormalEvaluationPlan,
    *,
    required_models: tuple[str, ...] = FORMAL_MODEL_KEYS,
) -> dict[str, pd.DataFrame]:
    """Strictly validate formal forecasts by ``target_date``, never position.

    It rejects missing, extra, duplicate, unordered, wrongly-labelled, or
    disagreeing actual values. Callers must correct the producing model rather
    than truncate, pad, or right-align its arrays.
    """
    expected_dates = tuple(pd.Timestamp(d) for d in plan.holdout_target_dates)
    expected_set = set(expected_dates)
    validated: dict[str, pd.DataFrame] = {}
    reference_actual: pd.Series | None = None

    for model_key in required_models:
        if model_key not in forecasts_by_model:
            raise FormalHoldoutAlignmentError(f"{plan.symbol}: missing formal forecasts for {model_key}.")
        frame = forecasts_by_model[model_key].copy()
        missing_columns = set(FORMAL_FORECAST_COLUMNS) - set(frame.columns)
        if missing_columns:
            raise FormalHoldoutAlignmentError(f"{model_key}: missing formal forecast columns: {sorted(missing_columns)}.")
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
        frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
        if frame["target_date"].duplicated().any():
            raise FormalHoldoutAlignmentError(f"{plan.symbol}/{model_key}: duplicate formal target_date values.")
        if len(frame) != plan.holdout_count:
            raise FormalHoldoutAlignmentError(
                f"{plan.symbol}/{model_key}: expected {plan.holdout_count} hold-out predictions, got {len(frame)}."
            )
        if not frame["target_date"].is_monotonic_increasing:
            raise FormalHoldoutAlignmentError(f"{plan.symbol}/{model_key}: formal target_date values are not chronological.")
        actual_dates = set(frame["target_date"])
        if actual_dates != expected_set:
            missing = sorted(expected_set - actual_dates)
            extra = sorted(actual_dates - expected_set)
            raise FormalHoldoutAlignmentError(
                f"{plan.symbol}/{model_key}: formal target-date mismatch; missing={missing}, extra={extra}."
            )
        if not (frame["origin_date"] < frame["target_date"]).all():
            raise FormalHoldoutAlignmentError(f"{plan.symbol}/{model_key}: every origin_date must precede target_date.")
        if frame["symbol"].astype(str).str.upper().ne(plan.symbol).any():
            raise FormalHoldoutAlignmentError(f"{plan.symbol}/{model_key}: incorrect symbol in formal forecasts.")
        if frame["model"].astype(str).ne(model_key).any():
            raise FormalHoldoutAlignmentError(f"{plan.symbol}/{model_key}: incorrect model key in formal forecasts.")

        frame = frame.sort_values("target_date").reset_index(drop=True)
        expected_error = frame["actual_close"].astype(float) - frame["predicted_close"].astype(float)
        if not np.allclose(frame["error"].astype(float), expected_error, equal_nan=False):
            raise FormalHoldoutAlignmentError(f"{plan.symbol}/{model_key}: error must equal actual_close - predicted_close.")
        indexed_actual = frame.set_index("target_date")["actual_close"].astype(float).reindex(expected_dates)
        if reference_actual is None:
            reference_actual = indexed_actual
        elif not np.allclose(indexed_actual.to_numpy(), reference_actual.to_numpy(), equal_nan=False):
            raise FormalHoldoutAlignmentError(
                f"{plan.symbol}/{model_key}: actual_close disagrees with another model for the same target_date."
            )
        validated[model_key] = frame

    return validated
