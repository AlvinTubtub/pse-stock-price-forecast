"""Leakage-safe Lag-Informed Regression formal and deployment workflows."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from services.evaluation import compute_metrics
from services.feature_engineering import REGRESSION_FEATURE_COLUMNS, RETURN_LAG_COLUMNS, build_full_features, reconstruct_price
from services.time_series_cv import FormalEvaluationPlan, development_ohlcv_for_plan, expanding_window_splitter

try:
    from statsmodels.tsa.stattools import pacf as _pacf
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover - optional dependency may be incompatible
    HAS_STATSMODELS = False

log = logging.getLogger(__name__)
MAX_PACF_LAG = 20
PACF_FALLBACK_LAGS = (1, 2, 3, 5)
LASSO_ALPHA_GRID = tuple(np.logspace(-4, 1, 30))


@dataclass
class LagRegressionArtifact:
    """Persisted deployment artifact; compatible with daily inference."""
    scaler: StandardScaler
    model: Lasso
    candidate_features: list[str] = field(default_factory=list)
    pacf_selected_lags: list[int] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    alpha: float = 0.0


@dataclass
class FormalLagRegressionResult:
    """Development-only formal fit and exact date-indexed hold-out rows."""
    artifact: LagRegressionArtifact
    metrics: dict[str, float]
    forecasts: pd.DataFrame
    backtest: list[float]


def select_pacf_return_lags(return_series: pd.Series, max_lag: int = MAX_PACF_LAG, alpha: float = 0.05) -> list[int]:
    """Select lags from *training-only daily returns* using 1.96/sqrt(n)."""
    series = pd.Series(return_series, dtype=float).dropna()
    if not HAS_STATSMODELS or len(series) < max_lag * 3 or series.nunique() <= 1:
        return list(PACF_FALLBACK_LAGS)
    try:
        values = _pacf(series, nlags=max_lag, method="ywm")
    except Exception:  # pragma: no cover
        log.warning("PACF computation failed; using deterministic fallback lags.", exc_info=True)
        return list(PACF_FALLBACK_LAGS)
    threshold = norm.ppf(1 - alpha / 2) / np.sqrt(len(series))
    selected = [lag for lag in range(1, max_lag + 1) if abs(values[lag]) > threshold]
    return selected or list(PACF_FALLBACK_LAGS)


pacf_select_lags = select_pacf_return_lags  # backwards-compatible name


def _candidate_columns(pacf_lags: list[int]) -> list[str]:
    selected = {f"return_lag_{lag}" for lag in pacf_lags}
    return [column for column in REGRESSION_FEATURE_COLUMNS if column not in set(RETURN_LAG_COLUMNS) - selected]


def _usable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build traceable supervised rows without backfill or other imputation."""
    features = build_full_features(df)
    required = list(REGRESSION_FEATURE_COLUMNS) + ["target_delta", "origin_date", "target_date", "daily_return"]
    return features.dropna(subset=required).copy()


def _select_alpha(features: pd.DataFrame) -> float:
    """Manual expanding CV: PACF and scaler are fit inside each fold only."""
    if len(features) < 20:
        raise ValueError("At least 20 usable development rows are required for regression CV.")
    splitter = expanding_window_splitter(len(features))
    scores = np.full(len(LASSO_ALPHA_GRID), np.inf, dtype=float)
    for alpha_index, alpha in enumerate(LASSO_ALPHA_GRID):
        fold_rmses: list[float] = []
        for train_idx, validation_idx in splitter.split(features):
            fold_train, fold_validation = features.iloc[train_idx], features.iloc[validation_idx]
            columns = _candidate_columns(select_pacf_return_lags(fold_train["daily_return"]))
            scaler = StandardScaler().fit(fold_train[columns])
            model = Lasso(alpha=float(alpha), max_iter=50_000, tol=1e-3, random_state=42)
            model.fit(scaler.transform(fold_train[columns]), fold_train["target_delta"].to_numpy(dtype=float))
            predicted_delta = model.predict(scaler.transform(fold_validation[columns]))
            errors = fold_validation["target_delta"].to_numpy(dtype=float) - predicted_delta
            fold_rmses.append(float(np.sqrt(np.mean(errors**2))))
        if fold_rmses:
            scores[alpha_index] = float(np.mean(fold_rmses))
    if not np.isfinite(scores).any():
        raise RuntimeError("No LASSO alpha completed all regression CV folds.")
    return float(LASSO_ALPHA_GRID[int(np.argmin(scores))])


def _fit_final(features: pd.DataFrame, alpha: float) -> LagRegressionArtifact:
    lags = select_pacf_return_lags(features["daily_return"])
    columns = _candidate_columns(lags)
    scaler = StandardScaler().fit(features[columns])
    model = Lasso(alpha=alpha, max_iter=50_000, tol=1e-3, random_state=42)
    model.fit(scaler.transform(features[columns]), features["target_delta"].to_numpy(dtype=float))
    selected = [name for name, coefficient in zip(columns, model.coef_) if coefficient != 0]
    return LagRegressionArtifact(scaler, model, columns, lags, selected, alpha)


def _formal_forecast_rows(df: pd.DataFrame, plan: FormalEvaluationPlan, artifact: LagRegressionArtifact) -> pd.DataFrame:
    features = build_full_features(df)
    rows = features.loc[features["target_date"].isin(plan.holdout_target_dates)].copy()
    if len(rows) != plan.holdout_count or rows["target_date"].duplicated().any():
        raise ValueError(f"{plan.symbol}: regression cannot construct every required hold-out row.")
    if set(rows["target_date"]) != set(plan.holdout_target_dates):
        raise ValueError(f"{plan.symbol}: regression formal target dates do not match the plan.")
    if rows[artifact.candidate_features].isna().any().any():
        raise ValueError(f"{plan.symbol}: predictor warm-up prevents a required regression hold-out forecast.")
    predicted_delta = artifact.model.predict(artifact.scaler.transform(rows[artifact.candidate_features]))
    actual_close = reconstruct_price(df.loc[rows.index, "Close"], rows["target_delta"])
    predicted_close = reconstruct_price(df.loc[rows.index, "Close"], predicted_delta)
    result = pd.DataFrame({
        "symbol": plan.symbol, "model": "lag_reg",
        "origin_date": pd.to_datetime(rows["origin_date"]).to_numpy(),
        "target_date": pd.to_datetime(rows["target_date"]).to_numpy(),
        "actual_close": actual_close, "predicted_close": predicted_close,
    }).sort_values("target_date").reset_index(drop=True)
    result["error"] = result["actual_close"] - result["predicted_close"]
    return result


def train_formal_lag_regression(df: pd.DataFrame, plan: FormalEvaluationPlan) -> FormalLagRegressionResult:
    """Fit only development rows, then forecast every frozen hold-out date."""
    development = development_ohlcv_for_plan(df, plan)
    development_features = _usable_features(development)
    alpha = _select_alpha(development_features)
    artifact = _fit_final(development_features, alpha)
    forecasts = _formal_forecast_rows(df, plan, artifact)
    development_actual = reconstruct_price(development.loc[development_features.index, "Close"], development_features["target_delta"])
    metrics = compute_metrics(forecasts["actual_close"], forecasts["predicted_close"], y_train=development_actual)
    log.info("Formal lag regression %s: alpha=%g, hold-out rows=%d", plan.symbol, alpha, len(forecasts))
    return FormalLagRegressionResult(artifact, metrics, forecasts, forecasts["predicted_close"].tolist())


def train_deployment_lag_regression(df: pd.DataFrame) -> LagRegressionArtifact:
    """Fit the persisted next-day artifact on all currently approved data."""
    features = _usable_features(df)
    return _fit_final(features, _select_alpha(features))


def train(df: pd.DataFrame):
    """Legacy deployment API; formal callers must use train_formal_lag_regression."""
    artifact = train_deployment_lag_regression(df)
    return artifact, {}, predict_next(artifact, df), [], [], []


def save(artifact: LagRegressionArtifact, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load(path) -> LagRegressionArtifact:
    return joblib.load(path)


def predict_next(artifact: LagRegressionArtifact, df: pd.DataFrame) -> float:
    """Predict next Close without future-value imputation."""
    features = build_full_features(df)
    last_row = features.iloc[[-1]][artifact.candidate_features]
    if last_row.isna().any().any():
        raise ValueError("Latest regression predictors are incomplete; no future-value imputation is permitted.")
    next_delta = float(artifact.model.predict(artifact.scaler.transform(last_row))[0])
    return float(df["Close"].iloc[-1] + next_delta)
