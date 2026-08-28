"""Phase 4 formal LSTM contracts; expensive 48x5 training is mocked here."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.formal_evaluation import FormalHoldoutAlignmentError, validate_formal_holdout_alignment
from services.time_series_cv import create_formal_evaluation_plan


def _load_module():
    path = Path(__file__).resolve().parent.parent / "services" / "forecasting" / "lstm_model.py"
    spec = importlib.util.spec_from_file_location("phase4_lstm_model", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lstm = _load_module()


def _df(n=100):
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    close = 100 + np.arange(n) * .2 + np.sin(np.arange(n) / 3)
    return pd.DataFrame({"Date": dates, "Close": close})


def test_formal_samples_are_univariate_delta_close_with_correct_target():
    df = _df(45)
    samples = lstm._formal_delta_samples(df, 5)
    first = samples.iloc[0]
    assert len(first["sequence"]) == 5
    assert first["target_delta"] == pytest.approx(df["Close"].iloc[6] - df["Close"].iloc[5])
    assert first["origin_date"] == df["Date"].iloc[5]
    assert first["target_date"] == df["Date"].iloc[6]


def test_singleton_minibatch_preserves_lstm_batch_and_target_dimensions():
    """The final one-sample mini-batch must not lose its batch dimension."""
    sequences = np.zeros((17, 30), dtype="float32")
    targets = np.zeros(17, dtype="float32")

    assert tuple(lstm._formal_input_tensor(sequences[0]).shape) == (1, 30, 1)
    assert tuple(lstm._formal_input_tensor(sequences[:16]).shape) == (16, 30, 1)
    assert tuple(lstm._formal_target_tensor(targets[0]).shape) == (1, 1)
    assert tuple(lstm._formal_target_tensor(targets[:16]).shape) == (16, 1)


def test_approved_grid_has_exactly_48_configurations():
    grid = list(itertools.product(lstm.LOOKBACK_GRID, lstm.HIDDEN_UNITS_GRID, lstm.LEARNING_RATE_GRID, lstm.BATCH_SIZE_GRID))
    assert len(grid) == len(set(grid)) == 48
    assert lstm.LOOKBACK_GRID == (5, 10, 20, 30)
    assert lstm.HIDDEN_UNITS_GRID == (25, 50, 100)
    assert lstm.LEARNING_RATE_GRID == (0.01, 0.001)
    assert lstm.BATCH_SIZE_GRID == (16, 32)
    assert lstm.EPOCHS == 200 and lstm.PATIENCE == 10 and lstm.SEED == 42


def test_fold_scaler_and_early_stopping_tail_exclude_validation(monkeypatch):
    samples = lstm._formal_delta_samples(_df(100), 5)
    samples.at[samples.index[-1], "sequence"] = np.full(5, 1_000_000.0)
    scalers = []
    base = lstm.MinMaxScaler
    class RecordingScaler(base):
        def fit(self, values, y=None):
            result = super().fit(values, y); scalers.append(float(self.data_max_[0])); return result
    def fake_train(X_fit, y_fit, X_stop, y_stop, config):
        assert len(X_fit) + len(X_stop) < len(samples)  # no fold validation passed to stopping
        model = lstm._LSTMNet(1, config.hidden_size)
        return 0.0, model.state_dict(), {"epochs_trained": 1, "best_epoch": 1, "early_stopped": False}
    monkeypatch.setattr(lstm, "MinMaxScaler", RecordingScaler)
    monkeypatch.setattr(lstm, "_train_formal_one_config", fake_train)
    score, folds = lstm._evaluate_formal_config(samples, lstm.LSTMConfig(5, 25, .01, 16))
    assert score is not None and len(folds) == 5
    assert all(value < 1_000_000.0 for value in scalers)


def test_formal_output_matches_plan_and_is_oos(monkeypatch):
    df = _df(100)
    plan = create_formal_evaluation_plan(df, "BPI")
    config = lstm.LSTMConfig(5, 25, .01, 16)
    development = df.loc[df["Date"] <= plan.development_end_date]
    scaler = MinMaxScaler().fit(np.diff(development["Close"]).reshape(-1, 1))
    model = lstm._LSTMNet(1, 25)
    for parameter in model.parameters(): parameter.data.zero_()
    monkeypatch.setattr(lstm, "_select_formal_config", lambda _dev: (config, 0.123456, [{"rmse": .123456}] * 5))
    monkeypatch.setattr(lstm, "_fit_final_formal", lambda dev, cfg: (model, scaler, lstm._formal_delta_samples(dev, cfg.lookback), {"best_epoch": 1}))
    formal = lstm.train_formal_lstm(df, plan)
    validated = validate_formal_holdout_alignment({"lstm": formal.forecasts}, plan, required_models=("lstm",))
    assert list(validated["lstm"]["target_date"]) == list(plan.holdout_target_dates)
    assert formal.backtest == list(formal.forecasts["predicted_close"])
    assert formal.metadata["input_design"] == "univariate_delta_close"
    assert formal.artifact["input_size"] == 1


def test_wrong_or_missing_formal_dates_are_not_rescued():
    df = _df(100); plan = create_formal_evaluation_plan(df, "BPI")
    frame = pd.DataFrame({"symbol": "BPI", "model": "lstm", "origin_date": plan.holdout_origin_dates[:-1], "target_date": plan.holdout_target_dates[:-1], "actual_close": 1., "predicted_close": 1., "error": 0.})
    with pytest.raises(FormalHoldoutAlignmentError):
        validate_formal_holdout_alignment({"lstm": frame}, plan, required_models=("lstm",))


def test_versioned_univariate_artifact_predicts_without_legacy_features():
    df = _df(50)
    scaler = MinMaxScaler().fit(np.diff(df["Close"]).reshape(-1, 1))
    model = lstm._LSTMNet(1, 25)
    for parameter in model.parameters(): parameter.data.zero_()
    artifact = {"artifact_version": 2, "input_design": "univariate_delta_close", "state_dict": model.state_dict(), "input_size": 1, "seq_len": 5, "hidden_size": 25, "delta_scaler": scaler}
    expected_delta = scaler.inverse_transform([[0.0]])[0, 0]
    assert lstm.predict_next(artifact, df) == pytest.approx(df["Close"].iloc[-1] + expected_delta)
