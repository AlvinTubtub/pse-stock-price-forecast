"""Per-model TRAIN+VALIDATION -> FREEZE -> FINAL-TEST prediction runners.

Each runner:

  1. Trains (and, internally, tunes/selects) using *only* the TRAIN+
     VALIDATION dataframe, by calling straight into the existing
     ``services/forecasting/*.py`` training code — reused unmodified — with
     a dataframe that already excludes every final-test row (see
     ``splits.chronological_split``). PACF lag selection, LASSO's CV-
     selected lambda, ARIMA's CV order search, and the LSTM's grid search
     therefore never see a final-test row: it is not in the object they
     were called with.
  2. Freezes the resulting artifact — no further fitting happens anywhere
     below this point.
  3. Produces predictions for exactly the requested final-test dates,
     reusing each model's existing feature/reconstruction functions.
     Predicting date T uses realized OHLCV data up to and including T-1
     (standard walk-forward forecasting, matching how these models are
     actually deployed) — this is not leakage; see the README's "Leakage
     controls" section for the fit-vs-input distinction. The frozen
     artifact's parameters themselves are never re-fit on any final-test
     observation.

All four runners return the same ``ModelRun`` shape so ``run_evaluation.py``
and ``metrics.py`` never need to know which model produced which
predictions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from services.feature_engineering import build_full_features, reconstruct_price
from services.forecasting import arima_model, lag_regression, lstm_model

from .leakage_checks import assert_naive_uses_prior_close_only, assert_no_future_values_in_features
from .splits import ChronologicalSplit

log = logging.getLogger(__name__)


@dataclass
class ModelRun:
    model_key: str
    ticker: str
    dates: list  # calendar dates being predicted, chronological
    y_true: np.ndarray
    y_pred: np.ndarray
    base_close: np.ndarray  # actual close on the trading day immediately before each predicted date
    y_train_reference: np.ndarray  # TRAIN+VALIDATION Close series, for MASE scaling
    selection_source: str  # "trainval" (tuned) or "naive" (no tuning to do)
    fallback: bool = False  # True if an optional dependency (statsmodels/torch) was unavailable
    extra: dict = field(default_factory=dict)

    def restrict_to_dates(self, keep_dates: set) -> "ModelRun":
        keep_dates = {pd.Timestamp(d) for d in keep_dates}
        mask = [pd.Timestamp(d) in keep_dates for d in self.dates]
        idx = np.where(mask)[0]
        return ModelRun(
            model_key=self.model_key,
            ticker=self.ticker,
            dates=[self.dates[i] for i in idx],
            y_true=self.y_true[idx],
            y_pred=self.y_pred[idx],
            base_close=self.base_close[idx],
            y_train_reference=self.y_train_reference,
            selection_source=self.selection_source,
            fallback=self.fallback,
            extra=self.extra,
        )


def align_runs_to_common_dates(runs: dict[str, ModelRun]) -> dict[str, ModelRun]:
    """Restrict every model's run to the intersection of dates every other
    model was also able to produce a prediction for, so every model
    (including naive) is scored on exactly the same final-test dates."""
    date_sets = [set(pd.Timestamp(d) for d in run.dates) for run in runs.values()]
    common = set.intersection(*date_sets) if date_sets else set()
    return {key: run.restrict_to_dates(common) for key, run in runs.items()}


# --------------------------------------------------------------------------
# Naive baseline
# --------------------------------------------------------------------------

def run_naive(full_df: pd.DataFrame, split: ChronologicalSplit) -> ModelRun:
    """tomorrow = today, scored on the final-test dates. Uses nothing but
    the immediately preceding observed close."""
    merged = full_df.reset_index(drop=True)
    prior_close = merged["Close"].shift(1)
    test_dates = set(split.final_test["Date"])
    mask = merged["Date"].isin(test_dates) & prior_close.notna()

    dates = list(merged.loc[mask, "Date"])
    y_true = merged.loc[mask, "Close"].to_numpy(dtype=float)
    y_pred = prior_close.loc[mask].to_numpy(dtype=float)

    assert_naive_uses_prior_close_only(y_pred, prior_close.loc[mask].to_numpy(dtype=float), split.ticker)

    return ModelRun(
        model_key="naive",
        ticker=split.ticker,
        dates=dates,
        y_true=y_true,
        y_pred=y_pred,
        base_close=y_pred.copy(),
        y_train_reference=split.trainval["Close"].to_numpy(dtype=float),
        selection_source="naive",
        extra={},
    )


# --------------------------------------------------------------------------
# Lag-Informed Regression / LASSO
# --------------------------------------------------------------------------

def run_lag_regression(full_df: pd.DataFrame, split: ChronologicalSplit) -> ModelRun:
    ticker = split.ticker

    # Trains + selects (PACF lag selection, scaler, LassoCV lambda) using
    # ONLY split.trainval — this function never receives final-test rows.
    artifact, _trainval_metrics, _next, _backtest, _test_actual, _test_pred = lag_regression.train(split.trainval)

    features_full = build_full_features(full_df)
    target_dates = full_df["Date"].shift(-1)  # date whose close each row's target_delta predicts
    test_dates = set(split.final_test["Date"])
    mask = target_dates.isin(test_dates) & features_full["target_delta"].notna()
    idx = features_full.index[mask]

    if len(idx) == 0:
        raise ValueError(f"[{ticker}] lag_regression: no final-test rows had a computable feature row.")

    X = features_full.loc[idx, artifact.candidate_features]
    if X.isna().any().any():
        raise ValueError(f"[{ticker}] lag_regression: NaNs in final-test feature rows — cannot predict.")

    scaled = artifact.scaler.transform(X)
    pred_delta = artifact.model.predict(scaled)
    base_close = full_df.loc[idx, "Close"].to_numpy(dtype=float)
    pred_close = reconstruct_price(base_close, pred_delta)
    actual_close = reconstruct_price(base_close, features_full.loc[idx, "target_delta"].to_numpy(dtype=float))
    dates = list(target_dates.loc[idx])  # the date each row's target_delta actually predicts

    return ModelRun(
        model_key="lag_reg",
        ticker=ticker,
        dates=dates,
        y_true=np.asarray(actual_close, dtype=float),
        y_pred=np.asarray(pred_close, dtype=float),
        base_close=base_close,
        y_train_reference=split.trainval["Close"].to_numpy(dtype=float),
        selection_source="trainval",
        extra={
            "pacf_selected_lags": artifact.pacf_selected_lags,
            "n_candidate_features": len(artifact.candidate_features),
            "n_selected_features": len(artifact.selected_features),
        },
    )


# --------------------------------------------------------------------------
# ARIMA
# --------------------------------------------------------------------------

def run_arima(full_df: pd.DataFrame, split: ChronologicalSplit) -> ModelRun:
    ticker = split.ticker

    # CV order search + final fit happen entirely inside split.trainval —
    # this function never receives final-test rows.
    fitted, order, _trainval_metrics, _next, _backtest, _test_actual, _test_pred = arima_model.train(split.trainval)

    final_close = split.final_test["Close"].to_numpy(dtype=float)
    dates = list(split.final_test["Date"])
    base_close = np.concatenate([[split.trainval_last_close], final_close[:-1]]) if len(final_close) else np.array([])

    if fitted is None:
        # statsmodels unavailable in this environment: services/forecasting
        # /arima_model.py itself falls back to a drift model rather than
        # crashing. Mirror that same deterministic fallback here, using
        # only TRAIN+VALIDATION-derived drift (no final-test data enters
        # the drift estimate).
        drift = float(split.trainval["Close"].diff().mean())
        y_pred = split.trainval_last_close + np.cumsum(np.full(len(final_close), drift))
        return ModelRun(
            model_key="arima", ticker=ticker, dates=dates,
            y_true=final_close, y_pred=y_pred, base_close=base_close,
            y_train_reference=split.trainval["Close"].to_numpy(dtype=float),
            selection_source="trainval", fallback=True,
            extra={"order": list(order), "reason": "statsmodels not installed in this environment"},
        )

    y_pred = arima_model._walk_forward_forecast(fitted, final_close)
    return ModelRun(
        model_key="arima", ticker=ticker, dates=dates,
        y_true=final_close, y_pred=np.asarray(y_pred, dtype=float), base_close=base_close,
        y_train_reference=split.trainval["Close"].to_numpy(dtype=float),
        selection_source="trainval", fallback=False,
        extra={"order": list(order)},
    )


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

def _lstm_frame_with_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature frame as ``lstm_model._build_feature_frame(df,
    require_target=True)``, but keeping ``Date`` and the target's own
    ``target_date`` alongside it so predictions can be aligned back to
    specific calendar dates (the upstream helper resets/discards the
    index and never exposes dates)."""
    engineered = build_full_features(df)
    indicator_cols = [c for c in lstm_model.FEATURE_COLUMNS if c not in ("Open", "High", "Low", "Close", "Volume")]
    frame = df[["Date", "Open", "High", "Low", "Close", "Volume"]].join(engineered[indicator_cols])
    frame["target_delta"] = engineered["target_delta"]
    frame["target_date"] = df["Date"].shift(-1)
    return frame.dropna(subset=[c for c in frame.columns if c != "target_date"] + ["target_date"]).reset_index(drop=True)


def run_lstm(full_df: pd.DataFrame, split: ChronologicalSplit) -> ModelRun:
    ticker = split.ticker

    # Grid search + training happen entirely inside split.trainval — this
    # function never receives final-test rows.
    artifact, _trainval_metrics, _next, _backtest, _test_actual, _test_pred = lstm_model.train(split.trainval)

    if artifact is None or not lstm_model.HAS_TORCH:
        # torch unavailable, or not enough TRAIN+VALIDATION history for
        # even the smallest grid config: services/forecasting/lstm_model.py
        # itself falls back to naive-style scoring rather than crashing.
        # Mirror the same fallback deterministically on the final test.
        naive_run = run_naive(full_df, split)
        naive_run.model_key = "lstm"
        naive_run.selection_source = "trainval"
        naive_run.fallback = True
        naive_run.extra = {"reason": "torch not installed, or insufficient TRAIN+VALIDATION history, in this environment"}
        return naive_run

    import torch  # local import: only reached when HAS_TORCH is True

    frame = _lstm_frame_with_dates(full_df)
    seq_len = artifact["seq_len"]
    test_dates = set(split.final_test["Date"])

    X_raw = frame[artifact["feature_columns"]].to_numpy(dtype="float32")
    y_raw = frame["target_delta"].to_numpy(dtype="float32")
    close_raw = frame["Close"].to_numpy(dtype="float32")
    target_dates = frame["target_date"].to_numpy()

    xs, ys, closes, dts = [], [], [], []
    for i in range(len(X_raw) - seq_len + 1):
        end = i + seq_len - 1
        td = pd.Timestamp(target_dates[end])
        if td in test_dates:
            window_dates = frame["Date"].iloc[i : i + seq_len]
            assert_no_future_values_in_features(window_dates, td, ticker)
            xs.append(X_raw[i : i + seq_len])
            ys.append(y_raw[end])
            closes.append(close_raw[end])
            dts.append(td)

    if not xs:
        raise ValueError(
            f"[{ticker}] lstm: no final-test window could be built (seq_len={seq_len} may exceed "
            f"available TRAIN+VALIDATION history)."
        )

    X_seq = np.asarray(xs, dtype="float32")
    y_seq = np.asarray(ys, dtype="float32")
    close_seq = np.asarray(closes, dtype="float32")

    x_scaler, y_scaler = artifact["x_scaler"], artifact["y_scaler"]
    X_scaled = x_scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape).astype("float32")

    model = lstm_model._LSTMNet(input_size=artifact["input_size"], hidden_size=artifact["hidden_size"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.inference_mode():
        pred_scaled = model(torch.tensor(X_scaled)).squeeze(-1).numpy()

    pred_delta = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    pred_close = reconstruct_price(close_seq, pred_delta)
    actual_close = reconstruct_price(close_seq, y_seq)

    return ModelRun(
        model_key="lstm",
        ticker=ticker,
        dates=dts,
        y_true=np.asarray(actual_close, dtype=float),
        y_pred=np.asarray(pred_close, dtype=float),
        base_close=np.asarray(close_seq, dtype=float),
        y_train_reference=split.trainval["Close"].to_numpy(dtype=float),
        selection_source="trainval",
        fallback=False,
        extra={"seq_len": int(seq_len), "hidden_size": int(artifact["hidden_size"])},
    )


RUNNERS = {
    "naive": run_naive,
    "lag_reg": run_lag_regression,
    "arima": run_arima,
    "lstm": run_lstm,
}
