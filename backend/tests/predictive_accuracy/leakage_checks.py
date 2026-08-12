"""Automated leakage / integrity checks for the final-test evaluation.

Every function here raises ``LeakageError`` with a specific, actionable
message on failure. The evaluation run (see ``run_evaluation.py``) calls
these at every stage where leakage could plausibly be introduced and never
catches ``LeakageError`` silently — a failed check aborts the run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """Raised when an integrity check detects (or cannot rule out) leakage."""


def assert_no_date_overlap(trainval: pd.DataFrame, final_test: pd.DataFrame, ticker: str) -> None:
    """No date may appear in both windows, and TRAIN+VALIDATION must end
    strictly before FINAL TEST begins."""
    if trainval.empty or final_test.empty:
        return
    if trainval["Date"].max() >= final_test["Date"].min():
        raise LeakageError(
            f"[{ticker}] TRAIN+VALIDATION dates overlap or extend into FINAL TEST: "
            f"trainval max={trainval['Date'].max().date()}, test min={final_test['Date'].min().date()}."
        )
    overlap = set(trainval["Date"]) & set(final_test["Date"])
    if overlap:
        raise LeakageError(
            f"[{ticker}] {len(overlap)} duplicate date(s) shared between "
            f"TRAIN+VALIDATION and FINAL TEST: {sorted(overlap)[:5]}."
        )


def assert_fit_input_excludes_dates(fit_dates: pd.Series | list, forbidden_dates: pd.Series | list, ticker: str, stage: str) -> None:
    """``fit_dates`` is every date actually present in the data a
    scaler/model/feature-selection step was fit on. Fails loudly if any
    forbidden (final-test) date leaked into it."""
    leaked = set(pd.Series(fit_dates)) & set(pd.Series(forbidden_dates))
    if leaked:
        example = sorted(leaked)[:5]
        raise LeakageError(
            f"[{ticker}] {stage}: {len(leaked)} FINAL TEST date(s) were present in the data "
            f"used to fit it — e.g. {example}."
        )


def assert_identical_test_dates(dates_by_model: dict[str, list], ticker: str) -> None:
    """Every model (and the naive baseline) must be scored on exactly the
    same set of final-test dates, in the same order."""
    keys = list(dates_by_model.keys())
    if len(keys) < 2:
        return
    reference_key = keys[0]
    reference = [pd.Timestamp(d) for d in dates_by_model[reference_key]]
    for key in keys[1:]:
        candidate = [pd.Timestamp(d) for d in dates_by_model[key]]
        if candidate != reference:
            ref_set, cand_set = set(reference), set(candidate)
            raise LeakageError(
                f"[{ticker}] model '{key}' ({len(candidate)} dates) was scored on different "
                f"final-test dates than '{reference_key}' ({len(reference)} dates). "
                f"Only-in-'{reference_key}': {sorted(ref_set - cand_set)[:3]}; "
                f"only-in-'{key}': {sorted(cand_set - ref_set)[:3]}."
            )


def assert_naive_uses_prior_close_only(naive_pred: np.ndarray, prior_close: np.ndarray, ticker: str) -> None:
    """The naive baseline must be exactly tomorrow = today — nothing more
    elaborate (no drift, no smoothing) is allowed to call itself 'naive'."""
    naive_pred = np.asarray(naive_pred, dtype=float)
    prior_close = np.asarray(prior_close, dtype=float)
    if naive_pred.shape != prior_close.shape:
        raise LeakageError(
            f"[{ticker}] naive baseline shape mismatch: predictions={naive_pred.shape}, "
            f"prior close={prior_close.shape}."
        )
    if not np.allclose(naive_pred, prior_close, equal_nan=False):
        max_diff = float(np.max(np.abs(naive_pred - prior_close)))
        raise LeakageError(
            f"[{ticker}] naive baseline predictions deviate from the immediately preceding "
            f"observed close by up to {max_diff} — naive baseline must be tomorrow = today, exactly."
        )


def assert_model_selection_before_final_test(selection_source: str, ticker: str, model_key: str) -> None:
    """Every model's hyperparameters/order/architecture must have been
    chosen using only TRAIN+VALIDATION data (the naive baseline has no
    hyperparameters to select, so it is exempt)."""
    allowed = {"trainval", "naive"}
    if selection_source not in allowed:
        raise LeakageError(
            f"[{ticker}/{model_key}] model-selection source was '{selection_source}', "
            f"expected one of {sorted(allowed)} — selection must happen before the final test is touched."
        )


def assert_scaler_fit_row_count(scaler_fit_n: int, trainval_n: int, ticker: str, scaler_name: str) -> None:
    """A training-only scaler/transformer must be fit on exactly the
    TRAIN+VALIDATION row count — not more (which would mean it saw
    final-test rows) and not fewer (which would mean it silently dropped
    trainval rows without us noticing)."""
    if scaler_fit_n != trainval_n:
        raise LeakageError(
            f"[{ticker}] {scaler_name} was fit on {scaler_fit_n} rows but TRAIN+VALIDATION "
            f"has {trainval_n} rows — scaler must be fit on TRAIN+VALIDATION only, no more, no fewer."
        )


def assert_no_future_values_in_features(feature_frame_dates: pd.Series, as_of_date: pd.Timestamp, ticker: str) -> None:
    """Defensive check for hand-rolled feature windows: every date feeding
    a single prediction must be on or before the day *before* the date
    being predicted (features may use same-day-or-earlier OHLCV, never the
    target day's own close)."""
    future = [d for d in pd.Series(feature_frame_dates) if pd.Timestamp(d) >= as_of_date]
    if future:
        raise LeakageError(
            f"[{ticker}] feature window for predicting {as_of_date.date()} includes "
            f"{len(future)} date(s) on/after the target date: {sorted(future)[:3]}."
        )
