"""Fast Phase 6 artifact isolation and integrity tests (no model training)."""
from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import model_selector
from services.artifact_runs import FormalRunIntegrityError, FormalRunWriter, RunMode, deployment_current_dir, write_deployment_manifest
from services.time_series_cv import create_formal_evaluation_plan


def _plan_and_forecasts() -> tuple:
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=20), "Close": range(20)})
    plan = create_formal_evaluation_plan(df, "BPI")
    actual = list(range(100, 100 + plan.holdout_count))
    forecasts = {}
    for model in ("lag_reg", "arima", "lstm", "naive"):
        frame = pd.DataFrame({"symbol": "BPI", "model": model, "origin_date": plan.holdout_origin_dates, "target_date": plan.holdout_target_dates, "actual_close": actual, "predicted_close": actual, "error": [0.0] * plan.holdout_count})
        forecasts[model] = frame
    return plan, forecasts


def _complete_writer(tmp_path: Path) -> FormalRunWriter:
    plan, forecasts = _plan_and_forecasts()
    writer = FormalRunWriter(tmp_path, "controlled_run")
    writer.create()
    writer.write_split_manifest({"BPI": plan})
    writer.write_data_manifest({"BPI": {"source_path": "data/raw/BPI.csv", "sha256": "synthetic", "row_count": 20, "first_date": "2025-01-01", "last_date": "2025-01-20"}})
    writer.write_company("BPI", forecasts, {model: {"rmse": 0.0} for model in forecasts}, {})
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
    payload = {"plan": plan, "forecasts": forecasts, "metrics": {model: {"rmse": 0.0} for model in forecasts}, "diagnostics": {}, "development_close": [1.0, 2.0]}
    monkeypatch.setattr(model_selector, "BASE_DIR", tmp_path)
    source = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=2), "Close": [1.0, 2.0]})
    with patch.object(model_selector, "git_worktree_is_dirty", return_value=False), patch.object(model_selector, "validate_ohlcv_csv", return_value=source), patch.object(model_selector, "evaluate_formal_symbol", return_value=payload), patch.object(model_selector, "run_formal_statistical_tests", return_value={"per_company": {}}):
        final = model_selector.run_formal_evaluation(raw_dir=raw_dir, symbols=["BPI"], run_id="formal_only")
    assert final.is_file()
    assert not (tmp_path / "models" / "deployment" / "current").exists()
    assert not (tmp_path / "prediction_cache").exists()
    assert not (tmp_path / "best_models.json").exists()


def test_formal_orchestration_rejects_a_dirty_git_worktree(tmp_path):
    with patch.object(model_selector, "git_worktree_is_dirty", return_value=True):
        with pytest.raises(RuntimeError, match="clean Git worktree"):
            model_selector.run_formal_evaluation(raw_dir=tmp_path, symbols=["BPI"], run_id="dirty")
