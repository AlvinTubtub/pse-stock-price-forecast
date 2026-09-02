"""Shared expanding-window rolling-origin time-series cross-validation.

Used by formal model preparation so chronological split policy and explicit
validation target dates have one implementation rather than model-local
positional splits.

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


@dataclass(frozen=True)
class DevelopmentCVFold:
    """One explicit chronological development fold keyed by forecast dates."""

    fold_number: int
    training_origin_dates: tuple[pd.Timestamp, ...]
    training_target_dates: tuple[pd.Timestamp, ...]
    validation_origin_dates: tuple[pd.Timestamp, ...]
    validation_target_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class DevelopmentCVDatePlan:
    """Common formal-development folds usable by every approved lookback.

    The common date universe starts only after ``maximum_lookback`` forecast
    pairs.  A model may use fewer history values inside each sequence, but it
    must select and score rows using these frozen target-date folds.
    """

    symbol: str
    maximum_lookback: int
    fold_count: int
    common_origin_dates: tuple[pd.Timestamp, ...]
    common_target_dates: tuple[pd.Timestamp, ...]
    folds: tuple[DevelopmentCVFold, ...]


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


def create_development_cv_date_plan(
    plan: FormalEvaluationPlan,
    *,
    maximum_lookback: int = 30,
    fold_count: int = DEFAULT_N_SPLITS,
) -> DevelopmentCVDatePlan:
    """Freeze common expanding-window CV dates inside formal development.

    The formal 85/15 plan remains untouched.  Leading development targets are
    excluded only to guarantee that the largest approved lookback can form a
    complete sequence.  Five folds are required by default; insufficient data
    raises instead of silently reducing the formal fold count.
    """
    if maximum_lookback < 1:
        raise FormalEvaluationPlanError("maximum_lookback must be at least one forecast pair.")
    if fold_count < 2:
        raise FormalEvaluationPlanError("Development CV requires at least two folds.")
    if len(plan.development_target_dates) != len(plan.development_origin_dates):
        raise FormalEvaluationPlanError(f"{plan.symbol}: development origin and target dates are misaligned.")

    common_origins = plan.development_origin_dates[maximum_lookback:]
    common_targets = plan.development_target_dates[maximum_lookback:]
    if len(common_targets) <= fold_count:
        raise FormalEvaluationPlanError(
            f"{plan.symbol}: {len(common_targets)} common development targets cannot produce {fold_count} folds."
        )
    if set(common_targets).intersection(plan.holdout_target_dates):
        raise FormalEvaluationPlanError(f"{plan.symbol}: development CV target dates overlap the formal holdout.")

    splitter = TimeSeriesSplit(n_splits=fold_count)
    folds: list[DevelopmentCVFold] = []
    for fold_number, (training_indices, validation_indices) in enumerate(
        splitter.split(common_targets), start=1
    ):
        training_targets = tuple(common_targets[index] for index in training_indices)
        validation_targets = tuple(common_targets[index] for index in validation_indices)
        training_origins = tuple(common_origins[index] for index in training_indices)
        validation_origins = tuple(common_origins[index] for index in validation_indices)
        if not training_targets or not validation_targets or training_targets[-1] >= validation_targets[0]:
            raise FormalEvaluationPlanError(f"{plan.symbol}: development CV fold {fold_number} is not chronological.")
        folds.append(DevelopmentCVFold(
            fold_number=fold_number,
            training_origin_dates=training_origins,
            training_target_dates=training_targets,
            validation_origin_dates=validation_origins,
            validation_target_dates=validation_targets,
        ))

    return DevelopmentCVDatePlan(
        symbol=plan.symbol,
        maximum_lookback=maximum_lookback,
        fold_count=fold_count,
        common_origin_dates=tuple(common_origins),
        common_target_dates=tuple(common_targets),
        folds=tuple(folds),
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
