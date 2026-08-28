"""Phase 3 strict formal-ARIMA tests, isolated from local statsmodels ABI issues."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.formal_evaluation import FormalHoldoutAlignmentError, validate_formal_holdout_alignment
from services.time_series_cv import create_formal_evaluation_plan


def _load_arima_module():
    path = Path(__file__).resolve().parent.parent / "services" / "forecasting" / "arima_model.py"
    spec = importlib.util.spec_from_file_location("phase3_arima_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arima = _load_arima_module()


def _df(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    close = 100 + np.arange(n) * 0.2 + np.sin(np.arange(n) / 4)
    return pd.DataFrame({"Date": dates, "Close": close})


def test_candidates_are_bounded_and_adf_only_prioritizes_d():
    orders = arima._candidate_orders(1)
    assert orders[0][1] == 1
    assert all(0 <= p <= 3 and 0 <= d <= 2 and 0 <= q <= 3 for p, d, q in orders)


def test_failed_fold_invalidates_entire_candidate(monkeypatch):
    close = pd.Series(np.arange(70.0))
    arima.HAS_STATSMODELS = True

    class Fit:
        def forecast(self, steps): return pd.Series([1.0])
        def append(self, actual, refit=False): return self

    class Factory:
        def __init__(self, values, order): self.values = values
        def fit(self):
            if len(self.values) > 40:
                raise RuntimeError("forced final-fold failure")
            return Fit()

    monkeypatch.setattr(arima, "ARIMA", Factory, raising=False)
    result = arima._evaluate_candidate(close, (1, 1, 0))
    assert not result.valid
    assert result.successful_fold_count < result.required_fold_count
    assert result.mean_validation_rmse is None


def test_nonconverged_but_finite_fold_is_recorded_under_current_rule(monkeypatch):
    close = pd.Series(np.arange(70.0))
    arima.HAS_STATSMODELS = True

    class Fit:
        mle_retvals = {"converged": False}
        def forecast(self, steps): return pd.Series([1.0])
        def append(self, actual, refit=False): return self

    class Factory:
        def __init__(self, values, order): self.values = values
        def fit(self): return Fit()

    monkeypatch.setattr(arima, "ARIMA", Factory, raising=False)
    result = arima._evaluate_candidate(close, (1, 1, 0))
    assert result.valid is True
    assert result.cv_fold_convergence == (False,) * result.required_fold_count
    assert arima._candidate_diagnostics(result)["all_cv_folds_converged"] is False


def test_all_failed_candidates_raise_without_out_of_range_fallback(monkeypatch):
    arima.HAS_STATSMODELS = True
    monkeypatch.setattr(arima, "is_stationary", lambda _series: False)
    monkeypatch.setattr(arima, "_evaluate_candidate", lambda _close, order: arima.ARIMACandidateResult(order, 5, 0, False, None, ("failed",)))
    with pytest.raises(arima.ARIMAFormalSelectionError):
        arima._select_formal_order(pd.Series(np.arange(70.0)))
    assert arima.DEPLOYMENT_FALLBACK_ORDER != (5, 1, 0)


def test_lowest_full_precision_valid_candidate_wins(monkeypatch):
    arima.HAS_STATSMODELS = True
    monkeypatch.setattr(arima, "_candidate_orders", lambda _d: [(1, 0, 0), (1, 1, 0)])
    monkeypatch.setattr(arima, "is_stationary", lambda _series: True)
    values = {(1, 0, 0): 0.100004, (1, 1, 0): 0.100003}
    monkeypatch.setattr(arima, "_evaluate_candidate", lambda _close, order: arima.ARIMACandidateResult(order, 5, 5, True, values[order]))
    selected, _ = arima._select_formal_order(pd.Series(np.arange(70.0)))
    assert selected == (1, 1, 0)


def test_walk_forward_appends_actuals_after_each_forecast_with_refit_false():
    calls = []
    class Fit:
        def __init__(self, next_value=10.0): self.next_value = next_value
        def forecast(self, steps): return pd.Series([self.next_value])
        def append(self, actual, refit=False):
            calls.append((list(actual), refit))
            return Fit(self.next_value + 1)
    predictions = arima._walk_forward_forecast(Fit(), np.array([11.0, 12.0, 13.0]))
    assert predictions.tolist() == [10.0, 11.0, 12.0]
    assert calls == [([11.0], False), ([12.0], False), ([13.0], False)]


def test_formal_output_uses_exact_plan_dates_and_holdout_ljung_box(monkeypatch):
    df = _df()
    plan = create_formal_evaluation_plan(df, "BPI")
    arima.HAS_STATSMODELS = True

    class Fit:
        mle_retvals = {"converged": True}
        def forecast(self, steps): return pd.Series([101.0])
        def append(self, actual, refit=False): return self
    class Factory:
        def __init__(self, values, order): self.values = values
        def fit(self): return Fit()

    seen = {}
    selected_on = {}
    monkeypatch.setattr(arima, "ARIMA", Factory, raising=False)
    def select_order(development):
        selected_on["close"] = development.copy()
        return (1, 1, 0), [arima.ARIMACandidateResult((1, 1, 0), 5, 5, True, .1, (), (True,) * 5)]
    monkeypatch.setattr(arima, "_select_formal_order", select_order)
    monkeypatch.setattr(arima, "_holdout_ljung_box", lambda errors: seen.setdefault("errors", np.asarray(errors)) or {})
    # Return a concrete diagnostic instead of a truth-value-ambiguous ndarray.
    monkeypatch.setattr(arima, "_holdout_ljung_box", lambda errors: {"diagnostic_target": "holdout_forecast_errors", "lags": [1], "n_errors": len(errors), "statistic": 1.0, "p_value": 0.5})
    formal = arima.train_formal_arima(df, plan)
    validated = validate_formal_holdout_alignment({"arima": formal.forecasts}, plan, required_models=("arima",))
    assert list(validated["arima"]["target_date"]) == list(plan.holdout_target_dates)
    assert len(selected_on["close"]) == plan.development_count + 1
    assert selected_on["close"].iloc[-1] == df.loc[df["Date"] == plan.development_end_date, "Close"].iloc[0]
    assert formal.backtest == list(formal.forecasts["predicted_close"])
    assert formal.diagnostics["diagnostic_target"] == "holdout_forecast_errors"
    assert formal.diagnostics["cv_fold_convergence"] == [True] * 5
    assert formal.diagnostics["all_cv_folds_converged"] is True
    assert formal.diagnostics["final_fit_converged"] is True


def test_wrong_date_is_rejected_even_when_prediction_count_matches():
    df = _df()
    plan = create_formal_evaluation_plan(df, "BPI")
    frame = pd.DataFrame({
        "symbol": "BPI", "model": "arima", "origin_date": plan.holdout_origin_dates,
        "target_date": list(plan.holdout_target_dates[:-1]) + [plan.holdout_target_dates[-1] + pd.Timedelta(days=1)],
        "actual_close": [1.0] * plan.holdout_count, "predicted_close": [1.0] * plan.holdout_count, "error": [0.0] * plan.holdout_count,
    })
    with pytest.raises(FormalHoldoutAlignmentError):
        validate_formal_holdout_alignment({"arima": frame}, plan, required_models=("arima",))
