"""Fast tests for strict audited formal ARIMA behavior (no real model search)."""
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
CONFIG = arima.ARIMAConfiguration((1, 1, 0), "n")


def _df(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    close = 100 + np.arange(n) * 0.2 + np.sin(np.arange(n) / 4)
    return pd.DataFrame({"Date": dates, "Close": close})


def _fold_result(number: int = 1, rmse: float = 0.1):
    attempt = arima.ARIMAFitAttempt(number, "statespace", 500, True, True)
    return arima.ARIMAFoldResult(number, True, True, rmse, (attempt,))


def _candidate(configuration=CONFIG, *, valid: bool = True, rmse: float | None = 0.1):
    folds = tuple(_fold_result(number) for number in range(1, 6)) if valid else ()
    return arima.ARIMACandidateResult(
        configuration=configuration,
        required_fold_count=5,
        successful_fold_count=5 if valid else 0,
        valid=valid,
        mean_validation_rmse=rmse if valid else None,
        failure_reasons=() if valid else ("incomplete",),
        fold_results=folds,
    )


def test_candidate_grid_includes_zero_orders_and_exact_allowed_trends():
    configurations = arima._candidate_configurations(1)
    identities = {(item.order, item.trend) for item in configurations}
    assert ((0, 0, 0), "n") in identities
    assert ((0, 0, 0), "c") in identities
    assert ((0, 1, 0), "n") in identities
    assert ((0, 1, 0), "t") in identities
    assert ((0, 2, 0), "n") in identities
    for configuration in configurations:
        assert configuration.trend in {0: {"n", "c"}, 1: {"n", "t"}, 2: {"n"}}[configuration.order[1]]


def test_successful_retry_uses_only_predeclared_increasing_maxiter(monkeypatch):
    attempts = []

    class Fit:
        def __init__(self, converged): self.mle_retvals = {"converged": converged}

    class Factory:
        def __init__(self, values, order, trend):
            assert order == CONFIG.order and trend == CONFIG.trend

        def fit(self, method, method_kwargs):
            attempts.append((method, method_kwargs["maxiter"]))
            return Fit(len(attempts) == 2)

    monkeypatch.setattr(arima, "ARIMA", Factory, raising=False)
    fit, evidence = arima._fit_with_formal_retries(pd.Series(np.arange(40.0)), CONFIG, context="test")
    assert fit is not None
    assert attempts == [("statespace", 500), ("statespace", 2_000)]
    assert [item.converged for item in evidence] == [False, True]


@pytest.mark.parametrize(
    ("metadata", "expected_status"),
    [({"converged": False}, False), ({}, None)],
    ids=["confirmed-non-convergence", "unavailable-convergence-metadata"],
)
def test_unconfirmed_convergence_invalidates_candidate(monkeypatch, metadata, expected_status):
    class Fit:
        mle_retvals = metadata

    class Factory:
        def __init__(self, values, order, trend): pass
        def fit(self, method, method_kwargs): return Fit()

    monkeypatch.setattr(arima, "ARIMA", Factory, raising=False)
    result = arima._evaluate_candidate(pd.Series(np.arange(70.0)), CONFIG)
    assert result.valid is False
    assert result.successful_fold_count == 0
    assert all(fold.converged is expected_status for fold in result.fold_results)
    assert all(len(fold.attempts) == len(arima.FORMAL_OPTIMIZER_ATTEMPTS) for fold in result.fold_results)


def test_non_finite_predictions_invalidate_candidate(monkeypatch):
    class Fit:
        mle_retvals = {"converged": True}
        def forecast(self, steps): return pd.Series([np.nan])
        def append(self, actual, refit=False): return self

    class Factory:
        def __init__(self, values, order, trend): pass
        def fit(self, method, method_kwargs): return Fit()

    monkeypatch.setattr(arima, "ARIMA", Factory, raising=False)
    result = arima._evaluate_candidate(pd.Series(np.arange(70.0)), CONFIG)
    assert result.valid is False
    assert result.successful_fold_count == 0
    assert all("non-finite" in (fold.failure_reason or "") for fold in result.fold_results)


def test_incomplete_candidate_cannot_win_and_all_incomplete_raise(monkeypatch):
    arima.HAS_STATSMODELS = True
    winner = arima.ARIMAConfiguration((1, 0, 0), "c")
    incomplete = arima.ARIMAConfiguration((0, 0, 0), "n")
    monkeypatch.setattr(arima, "is_stationary", lambda _series: True)
    monkeypatch.setattr(arima, "_candidate_configurations", lambda _d: [incomplete, winner])
    monkeypatch.setattr(
        arima,
        "_evaluate_candidate",
        lambda _close, configuration: _candidate(winner, rmse=2.0) if configuration == winner else _candidate(incomplete, valid=False),
    )
    selected, _ = arima._select_formal_configuration(pd.Series(np.arange(70.0)))
    assert selected == winner

    monkeypatch.setattr(arima, "_evaluate_candidate", lambda _close, configuration: _candidate(configuration, valid=False))
    with pytest.raises(arima.ARIMAFormalSelectionError, match="No ARIMA candidate"):
        arima._select_formal_configuration(pd.Series(np.arange(70.0)))


def test_lowest_full_precision_candidate_wins_with_deterministic_tie_break(monkeypatch):
    arima.HAS_STATSMODELS = True
    first = arima.ARIMAConfiguration((1, 0, 0), "c")
    second = arima.ARIMAConfiguration((1, 1, 0), "n")
    monkeypatch.setattr(arima, "is_stationary", lambda _series: True)
    monkeypatch.setattr(arima, "_candidate_configurations", lambda _d: [first, second])
    values = {first: 0.100004, second: 0.100003}
    monkeypatch.setattr(arima, "_evaluate_candidate", lambda _close, configuration: _candidate(configuration, rmse=values[configuration]))
    selected, _ = arima._select_formal_configuration(pd.Series(np.arange(70.0)))
    assert selected == second


def test_final_development_fit_non_convergence_rejects_formal_symbol(monkeypatch):
    df = _df()
    plan = create_formal_evaluation_plan(df, "BPI")
    monkeypatch.setattr(arima, "_select_formal_configuration", lambda _close: (CONFIG, [_candidate()]))
    failed = arima.ARIMAFitAttempt(2, "statespace", 2_000, True, False, "optimizer_not_converged")
    monkeypatch.setattr(arima, "_fit_with_formal_retries", lambda *_args, **_kwargs: (None, (failed,)))
    with pytest.raises(arima.ARIMAFormalSelectionError, match="final development fit"):
        arima.train_formal_arima(df, plan)


def test_formal_selection_has_no_deployment_fallback(monkeypatch):
    df = _df()
    plan = create_formal_evaluation_plan(df, "BPI")
    monkeypatch.setattr(
        arima,
        "_select_formal_configuration",
        lambda _close: (_ for _ in ()).throw(arima.ARIMAFormalSelectionError("all candidates failed")),
    )
    with pytest.raises(arima.ARIMAFormalSelectionError, match="all candidates failed"):
        arima.train_formal_arima(df, plan)


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


def test_formal_output_keeps_exact_dates_and_json_safe_diagnostics(monkeypatch):
    df = _df()
    plan = create_formal_evaluation_plan(df, "BPI")

    class Fit:
        mle_retvals = {"converged": True}
        def forecast(self, steps): return pd.Series([101.0])
        def append(self, actual, refit=False): return self

    attempts = (arima.ARIMAFitAttempt(1, "statespace", 500, True, True),)
    monkeypatch.setattr(arima, "_select_formal_configuration", lambda _close: (CONFIG, [_candidate()]))
    monkeypatch.setattr(arima, "_fit_with_formal_retries", lambda *_args, **_kwargs: (Fit(), attempts))
    monkeypatch.setattr(arima, "_holdout_ljung_box", lambda errors: {"diagnostic_target": "holdout_forecast_errors", "lags": [1], "n_errors": len(errors), "statistic": 1.0, "p_value": 0.5})

    formal = arima.train_formal_arima(df, plan)
    validated = validate_formal_holdout_alignment({"arima": formal.forecasts}, plan, required_models=("arima",))
    assert list(validated["arima"]["target_date"]) == list(plan.holdout_target_dates)
    assert formal.trend == "n"
    assert formal.diagnostics["selected_configuration"] == {"order": [1, 1, 0], "trend": "n"}
    assert len(formal.diagnostics["cv_fold_results"]) == 5
    assert formal.diagnostics["final_fit_attempts"][0]["converged"] is True


def test_wrong_date_is_rejected_even_when_prediction_count_matches():
    df = _df()
    plan = create_formal_evaluation_plan(df, "BPI")
    frame = pd.DataFrame({
        "symbol": "BPI", "model": "arima", "origin_date": plan.holdout_origin_dates,
        "target_date": list(plan.holdout_target_dates[:-1]) + [plan.holdout_target_dates[-1] + pd.Timedelta(days=1)],
        "actual_close": [1.0] * plan.holdout_count, "predicted_close": [1.0] * plan.holdout_count,
        "error": [0.0] * plan.holdout_count,
    })
    with pytest.raises(FormalHoldoutAlignmentError):
        validate_formal_holdout_alignment({"arima": frame}, plan, required_models=("arima",))
