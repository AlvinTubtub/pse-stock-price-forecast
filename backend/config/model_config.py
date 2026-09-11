"""Authoritative model and chronological-tuning configuration."""

from dataclasses import dataclass, field
from enum import StrEnum


class ModelId(StrEnum):
    """Canonical model identifiers required by the frontend contract."""

    LAG_REGRESSION = "lag_reg"
    ARIMA = "arima"
    LSTM = "lstm"
    NAIVE = "naive"


@dataclass(frozen=True, slots=True)
class RegressionFeatureConfig:
    """Causal candidate-feature settings for Lag-Informed Regression."""

    return_lags: tuple[int, ...] = tuple(range(1, 21))
    rolling_return_windows: tuple[int, ...] = (5, 10, 20)
    volume_windows: tuple[int, ...] = (5, 20)
    rsi_period: int = 14
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    macd_signal_period: int = 9
    bollinger_window: int = 20
    bollinger_standard_deviations: float = 2.0
    raw_price_lags: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.return_lags or not self.rolling_return_windows or not self.volume_windows:
            raise ValueError("Return lags and rolling/volume windows cannot be empty")
        positive_integer_groups = (
            self.return_lags,
            self.rolling_return_windows,
            self.volume_windows,
            self.raw_price_lags,
        )
        if any(value <= 0 for group in positive_integer_groups for value in group):
            raise ValueError("Feature lags and windows must be positive integers")
        if len(set(self.return_lags)) != len(self.return_lags):
            raise ValueError("return_lags cannot contain duplicates")
        if len(set(self.raw_price_lags)) != len(self.raw_price_lags):
            raise ValueError("raw_price_lags cannot contain duplicates")
        scalar_periods = (
            self.rsi_period,
            self.ema_fast_period,
            self.ema_slow_period,
            self.macd_signal_period,
            self.bollinger_window,
        )
        if any(period <= 0 for period in scalar_periods):
            raise ValueError("Indicator periods must be positive integers")
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("ema_fast_period must be smaller than ema_slow_period")
        if self.bollinger_standard_deviations <= 0:
            raise ValueError("bollinger_standard_deviations must be positive")


@dataclass(frozen=True, slots=True)
class LagRegressionConfig:
    """Chronological LASSO and fold-local PACF settings."""

    alpha_grid: tuple[float, ...] = (
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    )
    cv_splits: int = 5
    pacf_max_lag: int = 20
    pacf_significance_z: float = 1.96
    max_iterations: int = 100_000
    tolerance: float = 1e-7
    coefficient_zero_tolerance: float = 1e-12
    features: RegressionFeatureConfig = field(default_factory=RegressionFeatureConfig)

    def __post_init__(self) -> None:
        if not self.alpha_grid or any(alpha <= 0 for alpha in self.alpha_grid):
            raise ValueError("alpha_grid must contain positive values")
        if tuple(sorted(set(self.alpha_grid))) != self.alpha_grid:
            raise ValueError("alpha_grid must be unique and strictly increasing")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least 2")
        if self.pacf_max_lag < 1:
            raise ValueError("pacf_max_lag must be positive")
        if self.pacf_significance_z <= 0:
            raise ValueError("pacf_significance_z must be positive")
        if self.max_iterations < 1 or self.tolerance <= 0:
            raise ValueError("LASSO convergence settings must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings that must be identical across model evaluations."""

    evaluation_proportion: float = 0.15
    random_seed: int = 42
    lag_regression: LagRegressionConfig = field(default_factory=LagRegressionConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.evaluation_proportion < 1.0:
            raise ValueError("evaluation_proportion must be strictly between 0 and 1")


DEFAULT_MODEL_CONFIG = ModelConfig()
