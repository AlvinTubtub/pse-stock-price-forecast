"""Common chronological development/evaluation planning."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import logging
import math

from config.companies import get_company
from config.model_config import DEFAULT_MODEL_CONFIG, ModelConfig
from src.data.validator import OhlcvRecord
from src.features.targets import NextDayForecastPair, build_next_day_pairs


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompanyEvaluationPlan:
    """One model-agnostic split that all models for a company must share."""

    symbol: str
    development_pairs: tuple[NextDayForecastPair, ...]
    evaluation_pairs: tuple[NextDayForecastPair, ...]
    evaluation_proportion: float

    def __post_init__(self) -> None:
        if not self.development_pairs or not self.evaluation_pairs:
            raise ValueError("Development and evaluation partitions must both be non-empty")
        development_dates = set(self.development_target_dates)
        evaluation_dates = set(self.evaluation_target_dates)
        if development_dates & evaluation_dates:
            raise ValueError("Development and evaluation target dates cannot overlap")
        if self.development_pairs[-1].target_date >= self.evaluation_pairs[0].target_date:
            raise ValueError("Development targets must precede every evaluation target")

    @property
    def development_target_dates(self) -> tuple[date, ...]:
        return tuple(pair.target_date for pair in self.development_pairs)

    @property
    def evaluation_target_dates(self) -> tuple[date, ...]:
        """Canonical target dates to pass unchanged to every future model."""

        return tuple(pair.target_date for pair in self.evaluation_pairs)

    @property
    def all_pairs(self) -> tuple[NextDayForecastPair, ...]:
        return self.development_pairs + self.evaluation_pairs


def split_next_day_pairs(
    symbol: str,
    pairs: Sequence[NextDayForecastPair],
    *,
    evaluation_proportion: float = DEFAULT_MODEL_CONFIG.evaluation_proportion,
) -> CompanyEvaluationPlan:
    """Split ordered pairs by proportion, reserving the newest targets for evaluation."""

    normalized_symbol = get_company(symbol).symbol
    if not 0.0 < evaluation_proportion < 1.0:
        raise ValueError("evaluation_proportion must be strictly between 0 and 1")
    if len(pairs) < 2:
        raise ValueError("At least two next-day pairs are required for a split")
    if any(
        current.target_date >= following.target_date
        for current, following in zip(pairs, pairs[1:])
    ):
        raise ValueError("Next-day pairs must be in strictly increasing target-date order")

    evaluation_count = math.ceil(len(pairs) * evaluation_proportion)
    evaluation_count = min(max(evaluation_count, 1), len(pairs) - 1)
    split_index = len(pairs) - evaluation_count
    plan = CompanyEvaluationPlan(
        symbol=normalized_symbol,
        development_pairs=tuple(pairs[:split_index]),
        evaluation_pairs=tuple(pairs[split_index:]),
        evaluation_proportion=evaluation_proportion,
    )
    LOGGER.info(
        "Built evaluation plan symbol=%s development=%d evaluation=%d evaluation_start=%s",
        plan.symbol,
        len(plan.development_pairs),
        len(plan.evaluation_pairs),
        plan.evaluation_target_dates[0],
    )
    return plan


def build_company_evaluation_plan(
    symbol: str,
    records: Sequence[OhlcvRecord],
    *,
    model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> CompanyEvaluationPlan:
    """Construct the sole chronological evaluation plan for one company."""

    pairs = build_next_day_pairs(records)
    return split_next_day_pairs(
        symbol,
        pairs,
        evaluation_proportion=model_config.evaluation_proportion,
    )
