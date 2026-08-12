"""Chronological TRAIN -> VALIDATION -> FINAL UNSEEN TEST split.

This module owns exactly one cut: the boundary between "everything the
models are allowed to see while training and selecting hyperparameters"
(TRAIN+VALIDATION) and "the final unseen test window" (FINAL TEST). Rows on
the FINAL TEST side of that boundary are never passed to any training,
scaling, feature-fitting, or model-selection code in this suite.

The existing per-model training code (services/forecasting/*.py) already
performs its own internal TRAIN/VALIDATION split (expanding-window CV
folds for LASSO's lambda and ARIMA's order search, and a train/validation/
test holdout for the LSTM's grid search) — this module's job is only to
make sure the FINAL TEST rows are removed *before* any of that code ever
runs, so "validation" inside services/forecasting/*.py and "final test"
here can never be confused with one another.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitError(ValueError):
    """Raised when a requested split cannot be produced safely."""


@dataclass(frozen=True)
class ChronologicalSplit:
    ticker: str
    trainval: pd.DataFrame
    final_test: pd.DataFrame

    @property
    def final_test_dates(self) -> list[pd.Timestamp]:
        return list(self.final_test["Date"])

    @property
    def trainval_last_close(self) -> float:
        return float(self.trainval["Close"].iloc[-1])


def chronological_split(
    df: pd.DataFrame,
    ticker: str,
    *,
    final_test_fraction: float = 0.15,
    final_test_start: str | None = None,
    final_test_end: str | None = None,
    min_trainval_rows: int = 60,
    min_final_test_rows: int = 5,
) -> ChronologicalSplit:
    """Split a validated, date-sorted OHLCV dataframe into TRAIN+VALIDATION
    and FINAL TEST, purely chronologically (rows are never shuffled or
    reordered).

    If ``final_test_start`` is given, the final-test window is exactly the
    rows within ``[final_test_start, final_test_end]`` (inclusive;
    ``final_test_end`` defaults to the ticker's last available date), and
    everything strictly before ``final_test_start`` becomes TRAIN+VALIDATION.
    Otherwise, the most recent ``final_test_fraction`` of rows (by count)
    become the final-test window.
    """
    if df.empty:
        raise SplitError(f"{ticker}: no rows to split.")
    if not df["Date"].is_monotonic_increasing:
        raise SplitError(f"{ticker}: input rows are not sorted chronologically.")
    if df["Date"].duplicated().any():
        raise SplitError(f"{ticker}: duplicate dates in input — cannot split safely.")

    n = len(df)

    if final_test_start:
        start_ts = pd.Timestamp(final_test_start)
        end_ts = pd.Timestamp(final_test_end) if final_test_end else df["Date"].max()
        if end_ts < start_ts:
            raise SplitError(f"{ticker}: final_test_end ({end_ts.date()}) is before final_test_start ({start_ts.date()}).")
        trainval = df[df["Date"] < start_ts].reset_index(drop=True)
        final_test = df[(df["Date"] >= start_ts) & (df["Date"] <= end_ts)].reset_index(drop=True)
    else:
        if not (0.0 < final_test_fraction < 1.0):
            raise SplitError(f"{ticker}: final_test_fraction must be in (0, 1), got {final_test_fraction}.")
        n_test = max(min_final_test_rows, int(round(n * final_test_fraction)))
        trainval = df.iloc[: n - n_test].reset_index(drop=True)
        final_test = df.iloc[n - n_test :].reset_index(drop=True)

    if len(trainval) < min_trainval_rows:
        raise SplitError(
            f"{ticker}: only {len(trainval)} TRAIN+VALIDATION rows after the split "
            f"(need >= {min_trainval_rows}). Provide more history or shrink the final-test window."
        )
    if len(final_test) < min_final_test_rows:
        raise SplitError(
            f"{ticker}: only {len(final_test)} FINAL TEST rows after the split "
            f"(need >= {min_final_test_rows}). Provide more history or shrink the final-test window."
        )
    if not final_test.empty and not trainval.empty and trainval["Date"].max() >= final_test["Date"].min():
        raise SplitError(
            f"{ticker}: TRAIN+VALIDATION (up to {trainval['Date'].max().date()}) is not strictly "
            f"before FINAL TEST (from {final_test['Date'].min().date()})."
        )

    return ChronologicalSplit(ticker=ticker, trainval=trainval, final_test=final_test)
