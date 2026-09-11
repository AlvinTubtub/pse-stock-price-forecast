"""Fresh all-model refit, compatibility, and next-session inference tests."""

from datetime import date, timedelta
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from config.model_config import (
    ArimaConfig,
    LagRegressionConfig,
    LstmConfig,
    ModelConfig,
    ModelId,
    RegressionFeatureConfig,
)
from src.data.calendar import PSETradingCalendar
from src.data.validator import OhlcvRecord
from src.features.regression_features import (
    build_regression_dataset,
    build_regression_origin_features,
)
from src.inference import predictor as predictor_module
from src.inference.next_day import (
    predict_next_day_from_artifacts,
    predict_next_day_from_fresh_refit,
)
from src.inference.predictor import (
    load_production_model,
    predict_with_production_model,
)
from src.models.arima import ArimaSpecification
from src.models.lstm import LstmSpecification
from src.training import production_refit as production_module
from src.training.production_refit import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    ModelArtifactCompatibilityError,
    ProductionRefitError,
    ProductionSelections,
    refit_all_principal_models,
    validate_model_artifact_metadata,
)


def synthetic_records(count: int = 60) -> tuple[OhlcvRecord, ...]:
    end = date(2025, 1, 10)  # Friday; the calendar test supplies Monday as closed.
    start = end - timedelta(days=count - 1)
    records = []
    for index in range(count):
        close = 100.0 + 0.08 * index + 1.2 * math.sin(index / 4.0)
        records.append(
            OhlcvRecord(
                trading_date=start + timedelta(days=index),
                open=close - 0.2,
                high=close + 0.7,
                low=close - 0.7,
                close=close,
                volume=10_000.0 + 10.0 * index,
            )
        )
    return tuple(records)


def quick_model_config() -> ModelConfig:
    features = RegressionFeatureConfig(
        return_lags=(1, 2, 3),
        rolling_return_windows=(3, 5),
        volume_windows=(3, 5),
        rsi_period=3,
        ema_fast_period=2,
        ema_slow_period=5,
        macd_signal_period=3,
        bollinger_window=5,
    )
    return ModelConfig(
        evaluation_proportion=0.2,
        lag_regression=LagRegressionConfig(
            alpha_grid=(0.01,),
            cv_splits=2,
            pacf_max_lag=3,
            features=features,
        ),
        arima=ArimaConfig(
            p_values=(0,),
            d_values=(1,),
            q_values=(0,),
            trend_options_by_d=((1, ("t",)),),
            cv_splits=2,
            retry_max_iterations=(200,),
        ),
        lstm=LstmConfig(
            lookback_lengths=(2,),
            hidden_sizes=(3,),
            learning_rates=(0.01,),
            batch_sizes=(8,),
            tuning_seeds=(3, 5, 7),
            cv_splits=2,
            max_epochs=2,
            early_stopping_patience=1,
            stopping_tail_proportion=0.2,
            minimum_stopping_samples=2,
            final_seed=42,
        ),
    )


def test_latest_lir_origin_uses_same_authoritative_feature_implementation() -> None:
    records = synthetic_records()
    config = quick_model_config().lag_regression.features
    dataset = build_regression_dataset(records, config)
    latest_labeled_origin = build_regression_origin_features(records[:-1], config)

    assert latest_labeled_origin.origin_date == dataset.samples[-1].origin_date
    assert latest_labeled_origin.origin_close == dataset.samples[-1].origin_close
    assert latest_labeled_origin.feature_names == dataset.feature_names
    np.testing.assert_allclose(
        latest_labeled_origin.feature_values,
        dataset.samples[-1].feature_values,
    )


def test_fresh_refit_persists_and_predicts_all_three_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    records = synthetic_records()
    config = quick_model_config()
    settings = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(production_module, "SETTINGS", settings)
    monkeypatch.setattr(predictor_module, "SETTINGS", settings)
    models_dir = settings.artifacts_dir / "models" / "ALI"
    models_dir.mkdir(parents=True)
    # Incompatible old-looking files prove --fresh does not attempt to load them.
    for model in (ModelId.LAG_REGRESSION, ModelId.ARIMA, ModelId.LSTM):
        (models_dir / f"{model.value}.metadata.json").write_text(
            "{\"schema_version\": 0}\n",
            encoding="utf-8",
        )
    selections = ProductionSelections(
        lir_alpha=0.01,
        arima=ArimaSpecification((0, 1, 0), "t"),
        lstm=LstmSpecification(2, 3, 0.01, 8),
    )

    result = refit_all_principal_models(
        "ALI",
        selections,
        records,
        model_config=config,
        fresh=True,
    )

    assert result.trained_through == records[-1].trading_date
    assert result.data_row_count == len(records)
    assert {artifact.model_family for artifact in result.artifacts} == {
        ModelId.LAG_REGRESSION,
        ModelId.ARIMA,
        ModelId.LSTM,
    }
    for artifact in result.artifacts:
        assert artifact.model_path.parent == models_dir
        assert artifact.metadata_path.parent == models_dir
        assert artifact.model_path.is_file()
        assert artifact.metadata_path.is_file()
        payload = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
        assert payload["symbol"] == "ALI"
        assert payload["model_family"] == artifact.model_family.value
        assert payload["trained_through"] == records[-1].trading_date.isoformat()
        assert payload["data_row_count"] == len(records)
        assert payload["schema_version"] == MODEL_ARTIFACT_SCHEMA_VERSION
        assert payload["hyperparameters"]
        assert artifact.metadata.created_at.utcoffset() == timedelta(hours=8)
    assert result.artifact_for(ModelId.LSTM).metadata.hyperparameters["seed"] == 42

    calendar = PSETradingCalendar.with_holidays({date(2025, 1, 13)})
    fresh_forecast = predict_next_day_from_fresh_refit(
        result,
        records,
        calendar=calendar,
    )
    loaded_forecast = predict_next_day_from_artifacts(
        "ALI",
        records,
        calendar=calendar,
    )

    assert fresh_forecast.forecast_for == date(2025, 1, 14)
    assert loaded_forecast.forecast_for == fresh_forecast.forecast_for
    assert len(fresh_forecast.predictions) == 3
    assert len(loaded_forecast.predictions) == 3
    assert {prediction.model for prediction in fresh_forecast.predictions} == {
        ModelId.LAG_REGRESSION,
        ModelId.ARIMA,
        ModelId.LSTM,
    }
    for model in ModelId.LAG_REGRESSION, ModelId.ARIMA, ModelId.LSTM:
        fresh = fresh_forecast.prediction_for(model)
        loaded = loaded_forecast.prediction_for(model)
        assert fresh.predicted_close == pytest.approx(loaded.predicted_close)
        assert fresh.predicted_close == pytest.approx(
            fresh.origin_close + fresh.predicted_delta
        )

    with pytest.raises(
        ModelArtifactCompatibilityError,
        match="trained-through date",
    ):
        predict_with_production_model(result.artifact_for(ModelId.ARIMA), records[:-1])

    compatible_payload = result.artifact_for(ModelId.LAG_REGRESSION).metadata.as_dict()
    validate_model_artifact_metadata(
        compatible_payload,
        expected_symbol="ALI",
        expected_model=ModelId.LAG_REGRESSION,
    )
    incompatible_payload = dict(compatible_payload)
    incompatible_payload["schema_version"] = 999
    with pytest.raises(ModelArtifactCompatibilityError, match="schema_version"):
        validate_model_artifact_metadata(
            incompatible_payload,
            expected_symbol="ALI",
            expected_model=ModelId.LAG_REGRESSION,
        )

    arima_artifact = result.artifact_for(ModelId.ARIMA)
    arima_payload = json.loads(
        arima_artifact.metadata_path.read_text(encoding="utf-8")
    )
    arima_payload["hyperparameters"]["order"] = [1, 1, 1]
    arima_artifact.metadata_path.write_text(
        json.dumps(arima_payload),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactCompatibilityError, match="metadata"):
        load_production_model("ALI", ModelId.ARIMA)

    lir_path = result.artifact_for(ModelId.LAG_REGRESSION).model_path
    with lir_path.open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ModelArtifactCompatibilityError, match="checksum"):
        load_production_model("ALI", ModelId.LAG_REGRESSION)


def test_production_refit_rejects_nonfresh_mode() -> None:
    with pytest.raises(ProductionRefitError, match="fresh-only"):
        refit_all_principal_models(
            "ALI",
            ProductionSelections(
                lir_alpha=0.01,
                arima=ArimaSpecification((0, 1, 0), "t"),
                lstm=LstmSpecification(2, 3, 0.01, 8),
            ),
            synthetic_records(),
            model_config=quick_model_config(),
            fresh=False,
        )
