"""Phase 2 regression leakage, feature, and formal-date tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.feature_engineering import REGRESSION_FEATURE_COLUMNS, _rsi, build_full_features
from services.formal_evaluation import validate_formal_holdout_alignment
from services.time_series_cv import create_formal_evaluation_plan


def _load_lag_module():
    path = Path(__file__).resolve().parent.parent / "services" / "forecasting" / "lag_regression.py"
    spec = importlib.util.spec_from_file_location("phase2_lag_regression", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lag = _load_lag_module()


def _ohlcv(n: int = 130) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    rng = np.random.default_rng(17)
    close = 100 + np.cumsum(rng.normal(0.15, 0.8, n))
    opening = close + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "Date": dates, "Open": opening, "High": np.maximum(opening, close) + 0.5,
        "Low": np.minimum(opening, close) - 0.5, "Close": close,
        "Volume": rng.integers(10_000, 100_000, n),
    })


@pytest.mark.parametrize(
    ("close", "expected"),
    [
        (pd.Series(np.arange(1.0, 41.0)), 100.0),
        (pd.Series(np.arange(40.0, 0.0, -1.0)), 0.0),
        (pd.Series(np.full(40, 10.0)), 50.0),
    ],
)
def test_rsi_edge_cases_after_warmup(close, expected):
    rsi = _rsi(close, 14)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[-1] == pytest.approx(expected)


def test_regression_features_preserve_dates_and_never_backfill_warmup():
    features = build_full_features(_ohlcv(45))
    assert set(REGRESSION_FEATURE_COLUMNS).issubset(features.columns)
    assert {"origin_date", "target_date", "target_delta"}.issubset(features.columns)
    assert features.loc[0, "ma_10"] != features.loc[10, "ma_10"]
    assert pd.isna(features.loc[0, "ma_10"])
    assert pd.isna(features.loc[0, "rsi_14"])


def test_pacf_is_called_per_fold_with_training_daily_returns(monkeypatch):
    features = lag._usable_features(_ohlcv())
    calls: list[np.ndarray] = []

    def select(returns, **_kwargs):
        calls.append(np.asarray(returns, dtype=float).copy())
        return [1, 2]

    monkeypatch.setattr(lag, "LASSO_ALPHA_GRID", (0.01, 0.1))
    monkeypatch.setattr(lag, "select_pacf_return_lags", select)
    lag._select_alpha(features)

    # Two alphas × expanding folds: each call receives only that fold's
    # daily-return series, never target_delta or later validation values.
    n_folds = lag.expanding_window_splitter(len(features)).get_n_splits()
    assert len(calls) == 2 * n_folds
    assert all(len(call) < len(features) for call in calls)
    assert all(np.isfinite(call).all() for call in calls)


def test_fold_scalers_exclude_extreme_validation_values(monkeypatch):
    features = lag._usable_features(_ohlcv())
    features.loc[features.index[-1], "log_volume"] = 1_000_000.0
    fitted_means: list[np.ndarray] = []
    base_scaler = lag.StandardScaler

    class RecordingScaler(base_scaler):
        def fit(self, X, y=None):
            result = super().fit(X, y)
            fitted_means.append(self.mean_.copy())
            return result

    monkeypatch.setattr(lag, "LASSO_ALPHA_GRID", (0.01,))
    monkeypatch.setattr(lag, "StandardScaler", RecordingScaler)
    monkeypatch.setattr(lag, "select_pacf_return_lags", lambda _returns: [1, 2])
    lag._select_alpha(features)

    assert fitted_means
    assert all(mean.max() < 1_000_000.0 for mean in fitted_means)


def test_formal_regression_uses_exact_plan_dates_and_development_only(monkeypatch):
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "BPI")
    monkeypatch.setattr(lag, "LASSO_ALPHA_GRID", (0.01, 0.1))
    monkeypatch.setattr(lag, "select_pacf_return_lags", lambda _returns: [1, 2, 3])

    formal = lag.train_formal_lag_regression(df, plan)
    validated = validate_formal_holdout_alignment({"lag_reg": formal.forecasts}, plan, required_models=("lag_reg",))

    assert list(validated["lag_reg"]["target_date"]) == list(plan.holdout_target_dates)
    assert formal.backtest == list(formal.forecasts["predicted_close"])
    # Mutating hold-out OHLCV cannot affect formal alpha/PACF selection.
    changed = df.copy()
    holdout_mask = changed["Date"] >= plan.holdout_start_date
    changed.loc[holdout_mask, ["Open", "High", "Low", "Close", "Volume"]] *= 100.0
    changed_formal = lag.train_formal_lag_regression(changed, plan)
    assert changed_formal.artifact.alpha == formal.artifact.alpha
    assert changed_formal.artifact.pacf_selected_lags == formal.artifact.pacf_selected_lags


def test_formal_regression_rejects_a_missing_required_target_date(monkeypatch):
    df = _ohlcv()
    plan = create_formal_evaluation_plan(df, "BPI")
    monkeypatch.setattr(lag, "LASSO_ALPHA_GRID", (0.01,))
    monkeypatch.setattr(lag, "select_pacf_return_lags", lambda _returns: [1, 2, 3])
    incomplete = df.loc[df["Date"] != plan.holdout_end_date].copy()

    with pytest.raises(ValueError, match="hold-out|boundary"):
        lag.train_formal_lag_regression(incomplete, plan)
