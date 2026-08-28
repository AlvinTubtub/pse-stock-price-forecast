"""Shared expanding-window rolling-origin time-series cross-validation.

Used by both the Lag-Informed Regression model (LASSO lambda selection)
and the ARIMA order search, so both follow the exact same validation
scheme the capstone paper specifies — one implementation, no drift.

``sklearn.model_selection.TimeSeriesSplit`` already *is* expanding-window
rolling-origin CV: each fold's training set grows to include everything
before the validation fold, and folds are strictly chronological (no
shuffling, no look-ahead). This module just centralizes the fold count
policy (5 folds when there's enough history, fewer only when the series
is too short) so every caller stays consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_N_SPLITS = 5
FORMAL_HOLDOUT_FRACTION = 0.15


class FormalEvaluationPlanError(ValueError):
    """Raised when a company cannot produce an unambiguous formal split."""


@dataclass(frozen=True)
class FormalEvaluationPlan:
    """Frozen, target-date-based 85/15 formal evaluation plan for one company.

    A forecasting row is ``origin_date=Date[t]`` and
    ``target_date=Date[t+1]``.  The split is deliberately made over those
    target rows, rather than raw OHLCV rows, so the first hold-out target is
    never accidentally used by a development observation.
    """

    symbol: str
    development_origin_dates: tuple[pd.Timestamp, ...]
    development_target_dates: tuple[pd.Timestamp, ...]
    holdout_origin_dates: tuple[pd.Timestamp, ...]
    holdout_target_dates: tuple[pd.Timestamp, ...]
    development_start_date: pd.Timestamp
    development_end_date: pd.Timestamp
    holdout_start_date: pd.Timestamp
    holdout_end_date: pd.Timestamp
    development_count: int
    holdout_count: int
    split_ratio: float
    total_forecast_rows: int


def create_formal_evaluation_plan(
    df: pd.DataFrame,
    symbol: str,
    *,
    holdout_fraction: float = FORMAL_HOLDOUT_FRACTION,
) -> FormalEvaluationPlan:
    """Create the single formal date manifest used by every model for a symbol.

    ``df`` must already be chronologically ordered validated OHLCV data, but
    this helper validates the Date column again because an invalid manifest is
    more dangerous than a loud failure.  The returned plan retains dates as
    ``Timestamp`` values; callers must not re-create a split from array sizes.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise FormalEvaluationPlanError("holdout_fraction must be strictly between 0 and 1.")
    if "Date" not in df or "Close" not in df:
        raise FormalEvaluationPlanError("Formal evaluation requires Date and Close columns.")
    if len(df) < 3:
        raise FormalEvaluationPlanError(f"{symbol}: at least three OHLCV rows are required.")

    dates = pd.to_datetime(df["Date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise FormalEvaluationPlanError(f"{symbol}: Date values must be unique and chronological.")
    if pd.to_numeric(df["Close"], errors="coerce").isna().any():
        raise FormalEvaluationPlanError(f"{symbol}: Close values must be numeric for formal evaluation.")

    origins = tuple(pd.Timestamp(d) for d in dates.iloc[:-1])
    targets = tuple(pd.Timestamp(d) for d in dates.iloc[1:])
    total = len(targets)
    holdout_count = max(1, int(round(total * holdout_fraction)))
    development_count = total - holdout_count
    if development_count < 1:
        raise FormalEvaluationPlanError(f"{symbol}: split leaves no development forecasting rows.")

    boundary = development_count
    return FormalEvaluationPlan(
        symbol=symbol.upper(),
        development_origin_dates=origins[:boundary],
        development_target_dates=targets[:boundary],
        holdout_origin_dates=origins[boundary:],
        holdout_target_dates=targets[boundary:],
        development_start_date=targets[0],
        development_end_date=targets[boundary - 1],
        holdout_start_date=targets[boundary],
        holdout_end_date=targets[-1],
        development_count=development_count,
        holdout_count=holdout_count,
        split_ratio=1.0 - holdout_fraction,
        total_forecast_rows=total,
    )


def build_forecast_rows(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return date-indexed one-step forecasting rows without choosing a split."""
    dates = pd.to_datetime(df["Date"], errors="raise")
    close = pd.to_numeric(df["Close"], errors="raise").to_numpy(dtype=float)
    return pd.DataFrame({
        "symbol": symbol.upper(),
        "origin_date": dates.iloc[:-1].to_numpy(),
        "target_date": dates.iloc[1:].to_numpy(),
        "actual_close": close[1:],
        "target_delta": close[1:] - close[:-1],
    })


def development_ohlcv_for_plan(df: pd.DataFrame, plan: FormalEvaluationPlan) -> pd.DataFrame:
    """Return raw rows ending at the final development *target* date.

    This is the model-preparation contract for later phases: model-specific
    warm-up may remove only earlier development rows, never plan hold-out
    dates.
    """
    dates = pd.to_datetime(df["Date"], errors="raise")
    mask = dates <= plan.development_end_date
    prepared = df.loc[mask].copy()
    if prepared.empty or pd.Timestamp(prepared["Date"].iloc[-1]) != plan.development_end_date:
        raise FormalEvaluationPlanError(f"{plan.symbol}: development boundary is absent from OHLCV data.")
    return prepared


def n_splits_for(n_samples: int, max_splits: int = DEFAULT_N_SPLITS, min_fold_size: int = 10) -> int:
    """Picks a safe number of expanding-window folds for a series of this
    length, capping at ``max_splits`` (5, per the paper) but shrinking for
    short series so every fold still has at least ``min_fold_size`` rows.
    """
    if n_samples < (min_fold_size * 2):
        return 2
    return max(2, min(max_splits, n_samples // min_fold_size))


def expanding_window_splitter(n_samples: int, max_splits: int = DEFAULT_N_SPLITS, min_fold_size: int = 10) -> TimeSeriesSplit:
    """Returns a ``TimeSeriesSplit`` configured for expanding-window
    rolling-origin cross-validation over ``n_samples`` chronological rows."""
    return TimeSeriesSplit(n_splits=n_splits_for(n_samples, max_splits, min_fold_size))
