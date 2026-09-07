"""Fast Phase 6 artifact isolation and integrity tests (no model training)."""
from __future__ import annotations

import sys
import json
import itertools
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


def _plan_and_forecasts(n: int = 20, symbol: str = "BPI") -> tuple:
    df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=n), "Close": range(n)})
    plan = create_formal_evaluation_plan(df, symbol)
    actual = list(range(100, 100 + plan.holdout_count))
    forecasts = {}
    for model in ("lag_reg", "arima", "lstm", "naive"):
        frame = pd.DataFrame({"symbol": symbol, "model": model, "origin_date": plan.holdout_origin_dates, "target_date": plan.holdout_target_dates, "actual_close": actual, "predicted_close": actual, "error": [0.0] * plan.holdout_count})
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


def _valid_lstm_diagnostics() -> dict:
    configurations = []
    for lookback, hidden, learning_rate, batch in itertools.product(
        (5, 10, 20, 30), (25, 50, 100), (.01, .001), (16, 32)
    ):
        folds = [{
            "fold_number": fold,
            "validation_target_start": f"2025-0{fold}-01",
            "validation_target_end": f"2025-0{fold}-02",
            "validation_target_dates": [f"2025-0{fold}-01", f"2025-0{fold}-02"],
            "seed_results": [
                {"seed": seed, "rmse": .1, "best_epoch": 1}
                for seed in (42, 123, 2026)
            ],
        } for fold in range(1, 6)]
        configurations.append({
            "configuration": {
                "lookback": lookback,
                "hidden_size": hidden,
                "learning_rate": learning_rate,
                "batch_size": batch,
            },
            "status": "complete",
            "mean_validation_rmse": .1,
            "validation_rmse_std": 0.0,
            "fold_results": folds,
        })
    return {"training_metadata": {
        "configuration_count": 48,
        "expected_tuning_fit_count": 720,
        "tuning_seeds": [42, 123, 2026],
        "configuration_results": configurations,
    }}


def _valid_lag_diagnostics() -> dict:
    grid = [float(value) for value in np.logspace(-4, 3, 36)]
    return {"tuning_metadata": {
        "grid": grid,
        "alpha_results": [
            {"alpha": value, "fold_count": 5, "all_folds_converged": index == 20}
            for index, value in enumerate(grid)
        ],
        "selected_index": 20,
        "selected_at_boundary": False,
    }}


def _valid_corporate_action_diagnostics() -> dict:
    return {
        "policy_id": "raw_close_retain_and_flag_v1",
        "primary_rows_excluded": 0,
        "automatic_outlier_or_error_exclusion": False,
        "verified_events": [],
        "sensitivity_analysis": {"status": "not_applicable_no_verified_holdout_events"},
    }


def _valid_diagnostics() -> dict:
    return {
        "lag_reg": _valid_lag_diagnostics(),
        "arima": _valid_arima_diagnostics(),
        "lstm": _valid_lstm_diagnostics(),
        "corporate_actions": _valid_corporate_action_diagnostics(),
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
        _valid_diagnostics(),
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
        "diagnostics": _valid_diagnostics(),
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
        tuning_metadata={"grid": [1e-4, 1.0, 1e3], "alpha_results": [{}, {}, {}], "selected_at_boundary": False},
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


def test_canonical_formal_run_requires_reviewed_corporate_action_registry(tmp_path):
    with pytest.raises(ValueError, match="requires a reviewed"):
        model_selector.run_formal_evaluation(
            raw_dir=tmp_path,
            symbols=list(model_selector.EXPECTED_TICKERS),
            run_id="missing_registry",
        )


def test_formal_finalization_rejects_unconfirmed_arima_diagnostics(tmp_path):
    writer = _complete_writer(tmp_path)
    diagnostics_path = writer.path / "per_company" / "BPI" / "diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    diagnostics["arima"]["cv_fold_results"][2]["converged"] = None
    diagnostics_path.write_text(json.dumps(diagnostics))
    with pytest.raises(FormalRunIntegrityError, match="fold 3"):
        writer.finalize()


def test_formal_frozen_source_preflight_fails_before_creating_run(tmp_path, monkeypatch):
    raw_dir = tmp_path / "frozen"
    raw_dir.mkdir()
    (raw_dir / "BPI.csv").write_text("Date,Close\n2025-01-01,1\n")
    source = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=20),
        "Close": range(20),
    })
    monkeypatch.setattr(model_selector, "BASE_DIR", tmp_path)
    with patch.object(model_selector, "git_worktree_is_dirty", return_value=False), patch.object(
        model_selector, "validate_ohlcv_csv", return_value=source
    ):
        with pytest.raises(ValueError, match="expected frozen cutoff"):
            model_selector.run_formal_evaluation(
                raw_dir=raw_dir,
                symbols=["BPI"],
                run_id="wrong_cutoff",
                expected_data_cutoff="2025-01-19",
                expected_row_count=20,
            )
    assert not (tmp_path / "results" / "formal" / "wrong_cutoff").exists()


def test_formal_run_resumes_only_missing_company_checkpoint(tmp_path, monkeypatch):
    raw_dir = tmp_path / "frozen"
    raw_dir.mkdir()
    for symbol in ("BPI", "ALI"):
        (raw_dir / f"{symbol}.csv").write_text("Date,Close\n2025-01-01,1\n")
    source = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=20),
        "Close": range(20),
    })

    def payload_for(symbol):
        plan, forecasts = _plan_and_forecasts(symbol=symbol)
        return {
            "plan": plan,
            "forecasts": forecasts,
            "metrics": {model: {"rmse": 0.0} for model in forecasts},
            "diagnostics": _valid_diagnostics(),
            "development_close": [float(value) for value in range(17)],
        }

    statistics = {
        "per_company": {
            symbol: {
                "dm_squared_error": {
                    "stage1_vs_naive": [
                        {
                            "model_a": model,
                            "beats_naive_rmse": True,
                            "significantly_beats_naive": False,
                        }
                        for model in ("lag_reg", "arima", "lstm")
                    ]
                }
            }
            for symbol in ("BPI", "ALI")
        }
    }
    monkeypatch.setattr(model_selector, "BASE_DIR", tmp_path)
    common_patches = (
        patch.object(model_selector, "git_worktree_is_dirty", return_value=False),
        patch.object(model_selector, "git_repository_commit", return_value="fixed-commit"),
        patch.object(model_selector, "validate_ohlcv_csv", return_value=source),
        patch.object(model_selector, "run_formal_statistical_tests", return_value=statistics),
    )

    first_calls = []
    def fail_on_ali(symbol, _df):
        first_calls.append(symbol)
        if symbol == "ALI":
            raise RuntimeError("simulated interruption")
        return payload_for(symbol)

    with common_patches[0], common_patches[1], common_patches[2], common_patches[3], patch.object(
        model_selector, "evaluate_formal_symbol", side_effect=fail_on_ali
    ):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            model_selector.run_formal_evaluation(
                raw_dir=raw_dir,
                symbols=["BPI", "ALI"],
                run_id="resumable",
                expected_data_cutoff="2025-01-20",
                expected_row_count=20,
            )
    assert first_calls == ["BPI", "ALI"]
    assert (tmp_path / "results" / "formal" / "resumable" / ".checkpoints" / "BPI" / "complete.json").is_file()

    resumed_calls = []
    def finish_ali(symbol, _df):
        resumed_calls.append(symbol)
        return payload_for(symbol)

    with patch.object(model_selector, "git_worktree_is_dirty", return_value=False), patch.object(
        model_selector, "git_repository_commit", return_value="fixed-commit"
    ), patch.object(model_selector, "validate_ohlcv_csv", return_value=source), patch.object(
        model_selector, "run_formal_statistical_tests", return_value=statistics
    ), patch.object(model_selector, "evaluate_formal_symbol", side_effect=finish_ali):
        finalized = model_selector.run_formal_evaluation(
            raw_dir=raw_dir,
            symbols=["BPI", "ALI"],
            run_id="resumable",
            expected_data_cutoff="2025-01-20",
            expected_row_count=20,
            resume=True,
        )
    assert resumed_calls == ["ALI"]
    assert finalized.is_file()
    assert not (tmp_path / "models" / "deployment").exists()


def test_formal_resume_rejects_changed_frozen_source(tmp_path, monkeypatch):
    raw_dir = tmp_path / "frozen"
    raw_dir.mkdir()
    source_path = raw_dir / "BPI.csv"
    source_path.write_text("original frozen bytes")
    source = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=20),
        "Close": range(20),
    })
    monkeypatch.setattr(model_selector, "BASE_DIR", tmp_path)
    writer = FormalRunWriter(tmp_path, "source_changed")
    original_provenance = model_selector.source_data_manifest(source_path, source)
    writer.create_or_resume({
        "formal_evaluation_version": "1.0",
        "repository_commit": "fixed-commit",
        "symbols": ["BPI"],
        "expected_data_cutoff": "2025-01-20",
        "expected_row_count": 20,
        "source_files": {
            "BPI": {
                "sha256": original_provenance["sha256"],
                "row_count": 20,
                "first_date": "2025-01-01",
                "last_date": "2025-01-20",
            }
        },
    })
    source_path.write_text("changed frozen bytes")

    with patch.object(model_selector, "git_worktree_is_dirty", return_value=False), patch.object(
        model_selector, "git_repository_commit", return_value="fixed-commit"
    ), patch.object(model_selector, "validate_ohlcv_csv", return_value=source), patch.object(
        model_selector, "evaluate_formal_symbol"
    ) as evaluate:
        with pytest.raises(FormalRunIntegrityError, match="resume contract"):
            model_selector.run_formal_evaluation(
                raw_dir=raw_dir,
                symbols=["BPI"],
                run_id="source_changed",
                expected_data_cutoff="2025-01-20",
                expected_row_count=20,
                resume=True,
            )
    evaluate.assert_not_called()
