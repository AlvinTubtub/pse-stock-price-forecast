"""M20 formal hold-out residual diagnostic contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import evaluation


def test_formal_diagnostics_pass_exact_holdout_errors_to_all_statistics(monkeypatch):
    errors = np.array([.1, -.2, .3, -.4, .5, -.6, .7, -.8, .9, -1.0])
    seen = {}

    def fake_ljung(values, lags, return_df):
        seen["ljung"] = values.copy()
        return pd.DataFrame({"lb_stat": [1.5], "lb_pvalue": [.2]})

    def fake_arch(values, nlags):
        seen["arch"] = values.copy()
        return 2.0, .3, 1.0, .4

    def fake_shapiro(values):
        seen["shapiro"] = values.copy()
        return .9, .5

    monkeypatch.setattr(evaluation, "acorr_ljungbox", fake_ljung)
    monkeypatch.setattr(evaluation, "het_arch", fake_arch)
    monkeypatch.setattr(evaluation, "shapiro", fake_shapiro)
    monkeypatch.setattr(evaluation, "HAS_STATSMODELS_DIAGNOSTICS", True)
    result = evaluation.run_formal_residual_diagnostics(errors, include_ljung_box=True)
    assert np.array_equal(seen["ljung"], errors)
    assert np.array_equal(seen["arch"], errors)
    assert np.array_equal(seen["shapiro"], errors)
    assert result["ljung_box"]["diagnostic_target"] == "holdout_forecast_errors"
    assert result["arch_lm"] == {"diagnostic": "arch_lm", "diagnostic_target": "holdout_forecast_errors", "computable": True, "n": 10, "lags": [2], "lm_statistic": 2.0, "lm_p_value": .3, "f_statistic": 1.0, "f_p_value": .4}


def test_insufficient_or_nonfinite_errors_are_structured_noncomputable():
    insufficient = evaluation.run_formal_residual_diagnostics([1.0, 2.0], include_ljung_box=True)
    assert insufficient["ljung_box"]["computable"] is False
    assert insufficient["arch_lm"]["computable"] is False
    nonfinite = evaluation.run_formal_residual_diagnostics([1.0, np.nan, 2.0])
    assert nonfinite["shapiro_wilk"]["computable"] is False
    assert nonfinite["arch_lm"]["diagnostic_target"] == "holdout_forecast_errors"


def test_phase5_attaches_holdout_error_diagnostics_to_each_principal_model():
    dates = pd.date_range("2025-01-01", periods=12)
    frames = {}
    for model, offset in (("lag_reg", .1), ("arima", .2), ("lstm", .3), ("naive", .5)):
        actual = np.arange(12, dtype=float)
        predicted = actual + offset
        frames[model] = pd.DataFrame({"target_date": dates, "actual_close": actual, "predicted_close": predicted, "error": actual - predicted})
    result = evaluation.run_formal_statistical_tests({"BPI": {"forecasts": frames, "development_close": np.arange(20, dtype=float)}})
    for model in ("lag_reg", "arima", "lstm"):
        diagnostics = result["per_company"]["BPI"]["diagnostics"][model]
        assert diagnostics["shapiro_wilk"]["diagnostic_target"] == "holdout_forecast_errors"
        assert diagnostics["arch_lm"]["diagnostic_target"] == "holdout_forecast_errors"
