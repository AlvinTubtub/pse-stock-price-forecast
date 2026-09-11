"""Authoritative configuration for the ForecastPH backend."""

from .companies import COMPANIES, COMPANY_BY_SYMBOL, Company, get_company
from .model_config import (
    ArimaConfig,
    DEFAULT_MODEL_CONFIG,
    LagRegressionConfig,
    ModelConfig,
    ModelId,
    RegressionFeatureConfig,
)
from .settings import SETTINGS, BackendSettings

__all__ = [
    "BackendSettings",
    "ArimaConfig",
    "COMPANIES",
    "COMPANY_BY_SYMBOL",
    "Company",
    "DEFAULT_MODEL_CONFIG",
    "LagRegressionConfig",
    "ModelConfig",
    "ModelId",
    "RegressionFeatureConfig",
    "SETTINGS",
    "get_company",
]
