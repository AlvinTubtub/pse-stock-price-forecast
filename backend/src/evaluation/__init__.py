"""Unified chronological model evaluation package."""

from .backtest import (
    BacktestAlignmentError,
    BacktestData,
    CanonicalPrediction,
    align_backtest_records,
    naive_prediction_records,
)
from .evaluator import (
    CompanyEvaluation,
    ModelPredictionOutput,
    evaluate_company,
    evaluate_prediction_outputs,
)
from .metrics import (
    EvaluationMetrics,
    MetricError,
    compute_evaluation_metrics,
    compute_mase_denominator,
)
from .model_selection import ModelRanking, rank_principal_models

__all__ = [
    "BacktestAlignmentError",
    "BacktestData",
    "CanonicalPrediction",
    "CompanyEvaluation",
    "EvaluationMetrics",
    "MetricError",
    "ModelPredictionOutput",
    "ModelRanking",
    "align_backtest_records",
    "compute_evaluation_metrics",
    "compute_mase_denominator",
    "evaluate_company",
    "evaluate_prediction_outputs",
    "naive_prediction_records",
    "rank_principal_models",
]
