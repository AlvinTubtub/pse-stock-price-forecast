"""Phase 1 tests for the centralized, target-date formal evaluation plan."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.evaluation import build_naive_formal_forecasts, compute_canonical_formal_metrics, compute_metrics
from services.formal_evaluation import FormalHoldoutAlignmentError, validate_formal_holdout_alignment
from services.time_series_cv import build_forecast_rows, create_development_cv_date_plan, create_formal_evaluation_plan


def _ohlcv(n: int = 21) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=n, freq="B")
    close = np.linspace(100.0, 120.0, n) + np.sin(np.arange(n)) * 0.123456
    return pd.DataFrame({"Date": dates, "Close": close})


def _forecast_frame(df: pd.DataFrame, plan, model: str, adjustment: float = 0.25) -> pd.DataFrame:
    rows = build_forecast_rows(df, plan.symbol)
    rows = rows.loc[rows["target_date"].isin(plan.holdout_target_dates)].copy()
    rows["model"] = model
    rows["predicted_close"] = rows["actual_close"] - adjustment
    rows["error"] = rows["actual_close"] - rows["predicted_close"]
    return rows[["symbol", "model", "origin_date", "target_date", "actual_close", "predicted_close", "error"]]


def _valid_forecasts(df: pd.DataFrame, plan) -> dict[str, pd.DataFrame]:
    forecasts = {key: _forecast_frame(df, plan, key) for key in ("lag_reg", "arima", "lstm")}
    forecasts["naive"] = build_naive_formal_forecasts(df, plan)
    return forecasts


def test_common_target_date_split_and_boundary_are_deterministic():
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "bpi")
    rows = build_forecast_rows(df, "BPI")

    assert len(rows) == len(df) - 1
    assert rows.iloc[0]["origin_date"] == df.iloc[0]["Date"]
    assert rows.iloc[0]["target_date"] == df.iloc[1]["Date"]
    assert rows.iloc[0]["actual_close"] == df.iloc[1]["Close"]
    assert rows.iloc[0]["target_delta"] == pytest.approx(df.iloc[1]["Close"] - df.iloc[0]["Close"])
    assert plan.total_forecast_rows == 20
    assert plan.holdout_count == 3  # round(20 * 0.15)
    assert plan.development_count == 17
    assert set(plan.development_target_dates).isdisjoint(plan.holdout_target_dates)
    assert plan.development_end_date < plan.holdout_start_date
    assert plan.holdout_origin_dates[0] == plan.development_end_date


def test_development_cv_plan_freezes_five_common_lookback_30_target_folds():
    df = _ohlcv(101)
    formal_plan = create_formal_evaluation_plan(df, "BPI")
    holdout_dates_before = formal_plan.holdout_target_dates
    cv_plan = create_development_cv_date_plan(formal_plan, maximum_lookback=30)

    assert cv_plan.fold_count == 5
    assert len(cv_plan.folds) == 5
    assert cv_plan.common_target_dates == formal_plan.development_target_dates[30:]
    assert cv_plan.common_origin_dates == formal_plan.development_origin_dates[30:]
    assert formal_plan.holdout_target_dates == holdout_dates_before
    assert set(cv_plan.common_target_dates).isdisjoint(formal_plan.holdout_target_dates)

    validation_dates = tuple(
        target
        for fold in cv_plan.folds
        for target in fold.validation_target_dates
    )
    for fold in cv_plan.folds:
        assert fold.training_target_dates[-1] < fold.validation_target_dates[0]
        assert len(fold.training_origin_dates) == len(fold.training_target_dates)
        assert len(fold.validation_origin_dates) == len(fold.validation_target_dates)

    dates = tuple(pd.to_datetime(df["Date"]))
    for lookback in (5, 10, 20, 30):
        available_targets = set(dates[lookback + 1:])
        assert set(validation_dates) <= available_targets


def test_alignment_accepts_same_target_dates_and_naive_uses_origin_close():
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "BPI")
    forecasts = _valid_forecasts(df, plan)
    validated = validate_formal_holdout_alignment(forecasts, plan)

    assert set(validated) == {"lag_reg", "arima", "lstm", "naive"}
    naive = validated["naive"]
    origin_close = df.set_index("Date")["Close"]
    assert np.allclose(naive["predicted_close"], origin_close.reindex(naive["origin_date"]).to_numpy())


@pytest.mark.parametrize(
    "model, mutate, message",
    [
        ("lstm", lambda frame: frame.iloc[:-1], "expected"),
        ("arima", lambda frame: pd.concat([frame, frame.iloc[[-1]].assign(target_date=frame["target_date"].iloc[-1] + pd.Timedelta(days=1))]), "expected"),
        ("lag_reg", lambda frame: pd.concat([frame, frame.iloc[[-1]]]), "duplicate"),
        ("arima", lambda frame: frame.assign(actual_close=lambda x: x["actual_close"] + 1.0, error=lambda x: x["error"] + 1.0), "actual_close"),
    ],
)
def test_alignment_rejects_missing_extra_duplicate_and_actual_mismatches(model, mutate, message):
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "BPI")
    forecasts = _valid_forecasts(df, plan)
    forecasts[model] = mutate(forecasts[model])

    with pytest.raises(FormalHoldoutAlignmentError, match=message):
        validate_formal_holdout_alignment(forecasts, plan)


def test_validator_never_rescues_mismatched_lengths_by_position():
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "BPI")
    forecasts = _valid_forecasts(df, plan)
    forecasts["lag_reg"] = forecasts["lag_reg"].iloc[1:].copy()

    with pytest.raises(FormalHoldoutAlignmentError):
        validate_formal_holdout_alignment(forecasts, plan)


def test_metrics_are_full_precision_numeric_floats():
    metrics = compute_metrics([1.0, 2.0, 4.0], [1.111111, 1.777777, 4.333333], y_train=[0.0, 1.0, 2.0, 4.0])

    for key in ("rmse", "mae", "mase", "r2"):
        assert isinstance(metrics[key], float)
    assert metrics["rmse"] != round(metrics["rmse"], 4)


def test_canonical_formal_metrics_replace_model_specific_mase_values():
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "BPI")
    forecasts = _valid_forecasts(df, plan)
    development_close = df.loc[df["Date"] <= plan.development_end_date, "Close"].to_numpy()
    existing = {
        model: {"mase": 999.0, "legacy_extra": model}
        for model in forecasts
    }

    denominator, metrics = compute_canonical_formal_metrics(
        forecasts,
        development_close,
        existing_metrics=existing,
    )

    expected_denominator = float(np.mean(np.abs(np.diff(development_close))))
    assert denominator == pytest.approx(expected_denominator)
    for model, frame in forecasts.items():
        expected_mae = float(np.mean(np.abs(frame["actual_close"] - frame["predicted_close"])))
        assert metrics[model]["mase"] == pytest.approx(expected_mae / expected_denominator)
        assert metrics[model]["mase"] != 999.0
        assert metrics[model]["legacy_extra"] == model
        assert all(isinstance(metrics[model][key], float) for key in ("rmse", "mae", "mase", "r2"))
