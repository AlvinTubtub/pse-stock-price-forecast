"""Fast Phase 6 artifact isolation and integrity tests (no model training)."""
from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import model_selector
from services.artifact_runs import FormalRunIntegrityError, FormalRunWriter, RunMode, deployment_current_dir, write_deployment_manifest
from services.time_series_cv import create_formal_evaluation_plan


def _plan_and_forecasts(n: int = 20) -> tuple:
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=n), "Close": range(n)})
    plan = create_formal_evaluation_plan(df, "BPI")
    actual = list(range(100, 100 + plan.holdout_count))
    forecasts = {}
    for model in ("lag_reg", "arima", "lstm", "naive"):
        frame = pd.DataFrame({"symbol": "BPI", "model": model, "origin_date": plan.holdout_origin_dates, "target_date": plan.holdout_target_dates, "actual_close": actual, "predicted_close": actual, "error": [0.0] * plan.holdout_count})
        forecasts[model] = frame
    return plan, forecasts


def _valid_arima_diagnostics() -> dict:
    attempt = {
        "attempt_number": 1,
        "method": "statespace",
        "maxiter": 500,
        "fit_completed": True,
        "converged": True,
        "failure_reason": None,
    }
    return {
        "selected_order": [1, 1, 0],
        "selected_trend": "n",
        "selected_configuration": {"order": [1, 1, 0], "trend": "n"},
        "optimizer_retry_policy": [{"method": "statespace", "maxiter": 500}],
        "cv_fold_results": [
            {
                "fold_number": fold,
                "successful": True,
                "converged": True,
                "validation_rmse": 0.1,
                "optimizer_attempts": [attempt],
            }
            for fold in range(1, 6)
        ],
        "all_cv_folds_converged": True,
        "final_fit_converged": True,
        "final_fit_attempts": [attempt],
        "candidate_cv": [
            {
                "configuration": {"order": [1, 1, 0], "trend": "n"},
                "valid": True,
            }
        ],
    }


def _complete_writer(tmp_path: Path) -> FormalRunWriter:
    plan, forecasts = _plan_and_forecasts()
    writer = FormalRunWriter(tmp_path, "controlled_run")
    writer.create()
    writer.write_split_manifest({"BPI": plan})
    writer.write_data_manifest({"BPI": {"source_path": "data/raw/BPI.csv", "sha256": "synthetic", "row_count": 20, "first_date": "2025-01-01", "last_date": "2025-01-20"}})
    writer.write_company(
        "BPI",
        forecasts,
        {model: {"rmse": 0.0} for model in forecasts},
        {"arima": _valid_arima_diagnostics()},
    )
    writer.write_methodology_manifest("2025-01-20", ["BPI"])
    writer.write_statistics({"per_company": {}})
    return writer


def test_formal_run_is_versioned_immutable_and_finalizable(tmp_path):
    writer = _complete_writer(tmp_path)
    finalized = writer.finalize()
    assert finalized.is_file()
    assert "per_company/BPI/holdout_predictions.csv" in json.loads(finalized.read_text())["artifact_sha256"]
    assert deployment_current_dir(tmp_path) == tmp_path / "models/deployment/current"
    assert RunMode.FORMAL == "formal"
    with pytest.raises(FileExistsError):
        writer.create()
    with pytest.raises(FormalRunIntegrityError):
        writer.write_statistics({})


def test_methodology_manifest_records_all_research_dependencies(tmp_path):
    writer = FormalRunWriter(tmp_path, "dependency_manifest")
    writer.create()
    writer.write_methodology_manifest("2025-01-20", ["BPI"], git_dirty=False)
    dependencies = json.loads((writer.path / "methodology_manifest.json").read_text())["dependencies"]
    assert set(("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "torch")) <= set(dependencies)
    assert dependencies["scikit-learn"] is not None


def test_incomplete_formal_run_is_rejected(tmp_path):
    plan, forecasts = _plan_and_forecasts()
    writer = FormalRunWriter(tmp_path, "incomplete")
    writer.create()
    writer.write_split_manifest({"BPI": plan})
    writer.write_data_manifest({"BPI": {}})
    writer.write_company("BPI", forecasts, {}, {})
    with pytest.raises(FormalRunIntegrityError, match="missing required"):
        writer.finalize()


def test_deployment_manifest_does_not_modify_finalized_formal_run(tmp_path):
    writer = _complete_writer(tmp_path)
    writer.finalize()
    formal_before = (writer.path / "finalized.json").read_bytes()
    manifest = write_deployment_manifest(tmp_path, {"BPI": {"lag_regression": "models/deployment/current/lag_regression/BPI.pkl"}})
    assert manifest.is_file()
    assert (writer.path / "finalized.json").read_bytes() == formal_before


def test_formal_orchestration_never_writes_deployment_or_dashboard_state(tmp_path, monkeypatch):
    plan, forecasts = _plan_and_forecasts()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "BPI.csv").write_text("Date,Close\n2025-01-01,1\n")
    payload = {
        "plan": plan,
        "forecasts": forecasts,
        "metrics": {model: {"rmse": 0.0} for model in forecasts},
        "diagnostics": {
            "lag_reg": {},
            "arima": _valid_arima_diagnostics(),
            "lstm": {},
        },
        "development_close": [1.0, 2.0],
    }
    monkeypatch.setattr(model_selector, "BASE_DIR", tmp_path)
    source = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=2), "Close": [1.0, 2.0]})
    statistics = {"per_company": {"BPI": {"dm_squared_error": {"stage1_vs_naive": [{"model_a": model, "beats_naive_rmse": True, "significantly_beats_naive": False} for model in ("lag_reg", "arima", "lstm")]}}}}
    with patch.object(model_selector, "git_worktree_is_dirty", return_value=False), patch.object(model_selector, "validate_ohlcv_csv", return_value=source), patch.object(model_selector, "evaluate_formal_symbol", return_value=payload), patch.object(model_selector, "run_formal_statistical_tests", return_value=statistics):
        final = model_selector.run_formal_evaluation(raw_dir=raw_dir, symbols=["BPI"], run_id="formal_only")
    assert final.is_file()
    assert not (tmp_path / "models" / "deployment" / "current").exists()
    assert not (tmp_path / "prediction_cache").exists()
    assert not (tmp_path / "best_models.json").exists()


def test_evaluate_formal_symbol_replaces_model_specific_metrics_with_canonical_values(monkeypatch):
    plan, forecasts = _plan_and_forecasts(100)
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=100), "Close": range(100)})
    stale_metrics = {"rmse": 999.0, "mae": 999.0, "mase": 999.0, "r2": 999.0}
    lag = SimpleNamespace(
        forecasts=forecasts["lag_reg"],
        metrics=stale_metrics,
        artifact=SimpleNamespace(
            alpha=0.1,
            selected_features=[],
            candidate_features=[],
            model=SimpleNamespace(coef_=np.array([])),
        ),
        backtest=[],
    )
    arima = SimpleNamespace(
        forecasts=forecasts["arima"],
        metrics={**stale_metrics, "ljung_box_pvalue": 0.75},
        diagnostics={},
        backtest=[],
    )
    lstm = SimpleNamespace(
        forecasts=forecasts["lstm"],
        metrics=stale_metrics,
        metadata={},
        selected_config=SimpleNamespace(lookback=30),
        backtest=[],
    )

    monkeypatch.setattr(model_selector.lag_regression, "train_formal_lag_regression", lambda *_args: lag)
    monkeypatch.setattr(model_selector.arima_model, "train_formal_arima", lambda *_args: arima)
    monkeypatch.setattr(model_selector.lstm_model, "train_formal_lstm", lambda *_args: lstm)
    monkeypatch.setattr(model_selector, "build_naive_formal_forecasts", lambda *_args: forecasts["naive"])
    monkeypatch.setattr(model_selector, "evaluate_naive", lambda *_args, **_kwargs: stale_metrics)
    monkeypatch.setattr(model_selector, "run_formal_residual_diagnostics", lambda *_args, **_kwargs: {})

    result = model_selector.evaluate_formal_symbol("BPI", df)

    assert result["development_cv_plan"].fold_count == 5
    assert result["mase_denominator"] == pytest.approx(1.0)
    for model in ("lag_reg", "arima", "lstm", "naive"):
        assert result["metrics"][model]["rmse"] == pytest.approx(0.0)
        assert result["metrics"][model]["mae"] == pytest.approx(0.0)
        assert result["metrics"][model]["mase"] == pytest.approx(0.0)
        assert isinstance(result["metrics"][model]["r2"], float)
    assert result["metrics"]["arima"]["ljung_box_pvalue"] == 0.75


def test_formal_orchestration_rejects_a_dirty_git_worktree(tmp_path):
    with patch.object(model_selector, "git_worktree_is_dirty", return_value=True):
        with pytest.raises(RuntimeError, match="clean Git worktree"):
            model_selector.run_formal_evaluation(raw_dir=tmp_path, symbols=["BPI"], run_id="dirty")


def test_formal_finalization_rejects_unconfirmed_arima_diagnostics(tmp_path):
    writer = _complete_writer(tmp_path)
    diagnostics_path = writer.path / "per_company" / "BPI" / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["arima"]["cv_fold_results"][2]["converged"] = None
    diagnostics_path.write_text(json.dumps(diagnostics))
    with pytest.raises(FormalRunIntegrityError, match="fold 3"):
        writer.finalize()
