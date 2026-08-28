from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.evaluation import best_model_consistency_check, common_mase_denominator, moving_block_bootstrap, _stage

def test_common_mase_denominator_excludes_holdout_and_is_shared():
    dev = [10., 12., 11., 14.]
    assert common_mase_denominator(dev) == common_mase_denominator(dev)
    assert common_mase_denominator(dev) == np.mean(np.abs(np.diff(dev)))

def test_stage1_gate_and_direction_are_separate_from_rmse():
    errors = {"naive": np.array([1., -1., 1., -1.]), "lag_reg": np.array([.5, -.5, .5, -.5]), "arima": np.array([2., -2., 2., -2.]), "lstm": np.array([1., -1., 1., -1.])}
    metrics = {key: {"rmse": float(np.sqrt(np.mean(value**2)))} for key, value in errors.items()}
    result = _stage(errors, metrics, 2, .05)
    assert len(result["stage1_vs_naive"]) == 3
    assert result["stage1_vs_naive"][0]["direction"] == "model_a_lower_loss"
    assert result["stage1_vs_naive"][1]["beats_naive_rmse"] is False

def test_moving_block_bootstrap_is_reproducible_and_default_is_5000():
    values = np.arange(20., dtype=float)
    a = moving_block_bootstrap(values, replications=100, seed=42)
    b = moving_block_bootstrap(values, replications=100, seed=42)
    assert a == b
    assert moving_block_bootstrap(values)["bootstrap_replications"] == 5000


def test_rmse_consistency_reports_a_unique_winner_and_eight_of_fifteen_pass():
    result = best_model_consistency_check({
        "arima": [2.0] * 15,
        "lag_reg": [1.0] * 8 + [3.0] * 7,
        "lstm": [3.0] * 8 + [1.0] * 7,
    })
    assert result["dominant_model"] == "lag_reg"
    assert result["dominant_count"] == 8
    assert result["tie"] is False
    assert result["tied_models"] == []
    assert result["pass"] is True


def test_rmse_consistency_reports_two_way_tie_deterministically():
    result = best_model_consistency_check({
        "lstm": [3.0, 3.0, 2.0, 3.0],
        "arima": [3.0, 1.0, 3.0, 1.0],
        "lag_reg": [2.0, 2.0, 1.0, 2.0],
    })
    assert result["dominant_model"] is None
    assert result["dominant_count"] == 2
    assert result["tie"] is True
    assert result["tied_models"] == ["arima", "lag_reg"]
    assert result["pass"] is False


def test_rmse_consistency_reports_three_way_tie_deterministically():
    result = best_model_consistency_check({
        "lstm": [3.0, 3.0, 1.0],
        "arima": [1.0, 3.0, 3.0],
        "lag_reg": [3.0, 1.0, 3.0],
    })
    assert result["dominant_model"] is None
    assert result["dominant_count"] == 1
    assert result["tie"] is True
    assert result["tied_models"] == ["arima", "lag_reg", "lstm"]
    assert result["pass"] is False


def test_rmse_consistency_fails_below_eight_of_fifteen():
    result = best_model_consistency_check({
        "arima": [2.0] * 15,
        "lag_reg": [1.0] * 7 + [3.0] * 8,
        "lstm": [3.0] * 7 + [1.0] * 8,
    })
    assert result["dominant_model"] == "lstm"
    assert result["dominant_count"] == 8
    assert result["pass"] is True
    result = best_model_consistency_check({
        "arima": [2.0] * 15,
        "lag_reg": [1.0] * 7 + [3.0] * 8,
        "lstm": [3.0] * 7 + [1.0] * 7 + [3.0],
    })
    assert result["dominant_count"] == 7
    assert result["pass"] is False
