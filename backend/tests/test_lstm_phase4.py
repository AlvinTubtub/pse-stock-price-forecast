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
from services.time_series_cv import create_development_cv_date_plan, create_formal_evaluation_plan


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
    assert lstm.TUNING_SEEDS == (42, 123, 2026)
    assert lstm.FINAL_FIT_SEED == 42


def test_fold_scaler_and_early_stopping_tail_exclude_validation(monkeypatch):
    development = _df(100)
    plan = create_formal_evaluation_plan(development, "BPI", holdout_fraction=.01)
    development.loc[development["Date"] == plan.development_end_date, "Close"] += 1_000_000
    cv_plan = create_development_cv_date_plan(plan)
    scalers = []
    base = lstm.MinMaxScaler
    class RecordingScaler(base):
        def fit(self, values, y=None):
            result = super().fit(values, y); scalers.append(float(self.data_max_[0])); return result
    def fake_train(X_fit, y_fit, X_stop, y_stop, config, *, seed):
        assert len(X_fit) >= 2 and len(X_stop) >= 1
        model = lstm._LSTMNet(1, config.hidden_size)
        return 0.0, model.state_dict(), {"epochs_trained": 1, "best_epoch": 1, "early_stopped": False, "seed": seed}
    monkeypatch.setattr(lstm, "MinMaxScaler", RecordingScaler)
    monkeypatch.setattr(lstm, "_train_formal_one_config", fake_train)
    score, score_std, folds = lstm._evaluate_formal_config(development, lstm.LSTMConfig(5, 25, .01, 16), cv_plan)
    assert score is not None and score_std is not None and len(folds) == 5
    assert all(len(fold["seed_results"]) == 3 for fold in folds)
    assert all([row["seed"] for row in fold["seed_results"]] == [42, 123, 2026] for fold in folds)
    assert len(scalers) == 5
    assert all(value < 1_000_000 for value in scalers)


def test_every_lookback_uses_identical_validation_target_ranges(monkeypatch):
    development = _df(100)
    plan = create_formal_evaluation_plan(development, "BPI", holdout_fraction=.01)
    cv_plan = create_development_cv_date_plan(plan)

    def fake_train(X_fit, y_fit, X_stop, y_stop, config, *, seed):
        model = lstm._LSTMNet(1, config.hidden_size)
        for parameter in model.parameters(): parameter.data.zero_()
        return 0.0, model.state_dict(), {"epochs_trained": 1, "best_epoch": 1, "early_stopped": False, "seed": seed}

    monkeypatch.setattr(lstm, "_train_formal_one_config", fake_train)
    validation_dates_by_lookback = []
    for lookback in (5, 10, 20, 30):
        _mean, _std, folds = lstm._evaluate_formal_config(
            development,
            lstm.LSTMConfig(lookback, 25, .01, 16),
            cv_plan,
        )
        validation_dates_by_lookback.append([
            fold["validation_target_dates"]
            for fold in folds
        ])
    assert validation_dates_by_lookback[1:] == [
        validation_dates_by_lookback[0],
        validation_dates_by_lookback[0],
        validation_dates_by_lookback[0],
    ]


def test_configuration_tie_breaking_is_deterministic(monkeypatch):
    monkeypatch.setattr(lstm, "LOOKBACK_GRID", (30, 5))
    monkeypatch.setattr(lstm, "HIDDEN_UNITS_GRID", (50, 25))
    monkeypatch.setattr(lstm, "LEARNING_RATE_GRID", (.01,))
    monkeypatch.setattr(lstm, "BATCH_SIZE_GRID", (32, 16))
    monkeypatch.setattr(
        lstm,
        "_evaluate_formal_config",
        lambda development, config, cv_plan: (1.0, .25, [{}] * cv_plan.fold_count),
    )
    plan = create_formal_evaluation_plan(_df(100), "BPI", holdout_fraction=.01)
    selected, mean_rmse, rmse_std, _folds = lstm._select_formal_config(
        _df(100), create_development_cv_date_plan(plan)
    )
    assert selected == lstm.LSTMConfig(5, 25, .01, 16)
    assert mean_rmse == 1.0 and rmse_std == .25


def test_final_refit_uses_fresh_complete_scaler_and_every_development_sequence(monkeypatch):
    development = _df(100)
    development.loc[development.index[-1], "Close"] += 10_000
    config = lstm.LSTMConfig(5, 25, .01, 16)
    scaler_maxima = []
    base = lstm.MinMaxScaler
    class RecordingScaler(base):
        def fit(self, values, y=None):
            result = super().fit(values, y)
            scaler_maxima.append(float(self.data_max_[0]))
            return result
    def fake_epoch_train(X_fit, y_fit, X_stop, y_stop, cfg, *, seed):
        return 0.0, {"discarded": True}, {"epochs_trained": 4, "best_epoch": 3, "early_stopped": True, "seed": seed}
    seen = {}
    def fake_fixed_train(X_train, y_train, cfg, *, epochs, seed):
        seen.update(count=len(X_train), epochs=epochs, seed=seed)
        return lstm._LSTMNet(1, cfg.hidden_size)
    monkeypatch.setattr(lstm, "MinMaxScaler", RecordingScaler)
    monkeypatch.setattr(lstm, "_train_formal_one_config", fake_epoch_train)
    monkeypatch.setattr(lstm, "_train_fixed_epochs", fake_fixed_train)

    _model, _scaler, samples, metadata = lstm._fit_final_formal(development, config)

    assert len(scaler_maxima) == 2
    assert scaler_maxima[0] < scaler_maxima[1]
    assert seen == {"count": len(samples), "epochs": 3, "seed": 42}
    assert metadata["epoch_selection"]["preliminary_scaler_excludes_stopping_tail"] is True
    assert metadata["final_refit"]["uses_all_development_sequences"] is True
    assert metadata["final_refit"]["uses_stopping_tail"] is False


def test_formal_output_matches_plan_and_is_oos(monkeypatch):
    df = _df(100)
    plan = create_formal_evaluation_plan(df, "BPI")
    config = lstm.LSTMConfig(5, 25, .01, 16)
    development = df.loc[df["Date"] <= plan.development_end_date]
    scaler = MinMaxScaler().fit(np.diff(development["Close"]).reshape(-1, 1))
    model = lstm._LSTMNet(1, 25)
    for parameter in model.parameters(): parameter.data.zero_()
    monkeypatch.setattr(lstm, "_select_formal_config", lambda _dev, _cv: (config, 0.123456, 0.0123, [{"mean_rmse": .123456}] * 5))
    monkeypatch.setattr(lstm, "_fit_final_formal", lambda dev, cfg: (model, scaler, lstm._formal_delta_samples(dev, cfg.lookback), {"epoch_selection": {"best_epoch": 1}, "final_refit": {"uses_all_development_sequences": True}}))
    formal = lstm.train_formal_lstm(df, plan)
    validated = validate_formal_holdout_alignment({"lstm": formal.forecasts}, plan, required_models=("lstm",))
    assert list(validated["lstm"]["target_date"]) == list(plan.holdout_target_dates)
    assert formal.backtest == list(formal.forecasts["predicted_close"])
    assert formal.metadata["input_design"] == "univariate_delta_close"
    assert formal.metadata["tuning_seeds"] == [42, 123, 2026]
    assert formal.metadata["validation_rmse_std"] == pytest.approx(0.0123)
    assert formal.artifact["input_size"] == 1
    assert formal.artifact["training_seed"] == 42


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
