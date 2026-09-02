"""Regression tests for the frontend deployment-backtest contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import export_forecast_artifacts
from scripts.export_forecast_artifacts import aligned_deployment_backtest_60, reconciled_production_backtest_60
from services.production_history import record_issued_forecast, reconcile_history


def _cache(count: int = 75) -> dict:
    dates = [str(day.date()) for day in pd.date_range("2026-01-02", periods=count, freq="B")]
    actual = [float(index) for index in range(count)]
    return {
        "deployment_backtest": {
            "schema_version": 1,
            "source": "audited_oos_holdout",
            "alignment": "common_target_date",
            "target_dates": dates,
            "actual": actual,
            "by_model": {
                "Lag-Informed Regression": [value + 0.1 for value in actual],
                "ARIMA": [value + 0.2 for value in actual],
                "LSTM": [value + 0.3 for value in actual],
                "Naive baseline": [value + 0.4 for value in actual],
            },
        }
    }


def test_export_uses_latest_60_common_target_dates_for_every_model():
    dates, actual, by_model = aligned_deployment_backtest_60(_cache(), "BPI")

    assert len(dates) == len(actual) == 60
    assert dates[0] == "2026-01-23"
    assert actual == [float(index) for index in range(15, 75)]
    assert set(by_model) == {"Lag-Informed Regression", "ARIMA", "LSTM", "Naive baseline"}
    assert all(len(values) == 60 for values in by_model.values())
    assert by_model["ARIMA"][0] - actual[0] == pytest.approx(0.2)


@pytest.mark.parametrize("mutation", ["missing_model", "wrong_length", "not_oos", "unordered_dates"])
def test_export_rejects_any_series_that_cannot_be_proven_aligned(mutation: str):
    cache = _cache()
    payload = cache["deployment_backtest"]
    if mutation == "missing_model":
        del payload["by_model"]["LSTM"]
    elif mutation == "wrong_length":
        payload["by_model"]["ARIMA"] = payload["by_model"]["ARIMA"][:-1]
    elif mutation == "not_oos":
        payload["source"] = "legacy_backtest"
    else:
        payload["target_dates"][10], payload["target_dates"][11] = payload["target_dates"][11], payload["target_dates"][10]

    with pytest.raises(ValueError):
        aligned_deployment_backtest_60(cache, "BPI")


def test_production_export_includes_only_realized_issued_forecasts_and_keeps_them_aligned(tmp_path, monkeypatch):
    history_dir = tmp_path / "production_history"
    monkeypatch.setattr(export_forecast_artifacts, "PRODUCTION_HISTORY_DIR", history_dir)
    predictions = {"Lag-Informed Regression": 10.1, "ARIMA": 10.2, "LSTM": 10.3}
    record_issued_forecast(history_dir, "BPI", target_date="2026-09-01", issued_at="2026-08-31T16:00:00+08:00", data_as_of="2026-08-31", predictions=predictions)
    later = {key: value + 1 for key, value in predictions.items()}
    record_issued_forecast(history_dir, "BPI", target_date="2026-09-02", issued_at="2026-09-01T16:00:00+08:00", data_as_of="2026-09-01", predictions=later)
    reconcile_history(history_dir, "BPI", pd.DataFrame({"Date": pd.to_datetime(["2026-09-01"]), "Close": [10.0]}), "2026-09-01T16:01:00+08:00")

    dates, actual, by_model = reconciled_production_backtest_60("BPI")

    assert dates == ["2026-09-01"]
    assert actual == [10.0]
    assert by_model == {key: [value] for key, value in predictions.items()}
