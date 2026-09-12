"""Deterministic principal-model ranking from unified evaluation metrics."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

from config.model_config import ModelId
from src.evaluation.metrics import EvaluationMetrics


PRINCIPAL_MODELS: tuple[ModelId, ...] = (
    ModelId.LAG_REGRESSION,
    ModelId.ARIMA,
    ModelId.LSTM,
)


class ModelSelectionError(ValueError):
    """Raised when principal-model metrics are incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class RankedModel:
    rank: int
    model: ModelId
    criterion: str
    value: float

    def as_dict(self) -> dict[str, int | str | float]:
        return {
            "rank": self.rank,
            "model": self.model.value,
            "criterion": self.criterion,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ModelRanking:
    """Principal-model order; the naive benchmark is deliberately not selectable."""

    criterion: str
    ranked_models: tuple[RankedModel, ...]

    @property
    def best_model(self) -> ModelId:
        return self.ranked_models[0].model

    def as_dict(self) -> dict[str, object]:
        return {
            "criterion": self.criterion,
            "best_model": self.best_model.value,
            "ranked_models": [model.as_dict() for model in self.ranked_models],
        }


def rank_principal_models(
    metrics_by_model: Mapping[ModelId, EvaluationMetrics],
    *,
    criterion: str = "rmse",
) -> ModelRanking:
    """Rank the three principal models by lowest evaluation RMSE by default."""

    if criterion not in {"rmse", "mae", "mase"}:
        raise ModelSelectionError("Ranking criterion must be rmse, mae, or mase")
    missing = [model.value for model in PRINCIPAL_MODELS if model not in metrics_by_model]
    if missing:
        raise ModelSelectionError(f"Missing principal-model metrics: {missing}")
    canonical_order = {model: index for index, model in enumerate(PRINCIPAL_MODELS)}
    ordered = sorted(
        PRINCIPAL_MODELS,
        key=lambda model: (
            getattr(metrics_by_model[model], criterion),
            canonical_order[model],
        ),
    )
    ranked: list[RankedModel] = []
    for index, model in enumerate(ordered, start=1):
        value = float(getattr(metrics_by_model[model], criterion))
        if not math.isfinite(value):
            raise ModelSelectionError(f"Non-finite {criterion} for {model.value}")
        ranked.append(RankedModel(index, model, criterion, value))
    return ModelRanking(criterion=criterion, ranked_models=tuple(ranked))
