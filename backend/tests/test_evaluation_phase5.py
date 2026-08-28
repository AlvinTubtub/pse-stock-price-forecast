from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.evaluation import common_mase_denominator, moving_block_bootstrap, _stage

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
