"""Model-independent training defaults shared by future model implementations."""

from dataclasses import dataclass
from enum import StrEnum


class ModelId(StrEnum):
    """Canonical model identifiers required by the frontend contract."""

    LAG_REGRESSION = "lag_reg"
    ARIMA = "arima"
    LSTM = "lstm"
    NAIVE = "naive"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings that must be identical across model evaluations."""

    evaluation_proportion: float = 0.15
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.evaluation_proportion < 1.0:
            raise ValueError("evaluation_proportion must be strictly between 0 and 1")


DEFAULT_MODEL_CONFIG = ModelConfig()
