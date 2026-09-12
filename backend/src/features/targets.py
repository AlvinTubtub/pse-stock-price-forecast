"""Model-independent next-session forecast targets."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import logging

from src.data.validator import OhlcvRecord, require_chronological_records


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NextDayForecastPair:
    """A forecast origin and its immediately following observed session."""

    origin_date: date
    target_date: date
    origin_close: float
    actual_close: float
    target_delta: float


def build_next_day_pairs(
    records: Sequence[OhlcvRecord],
) -> tuple[NextDayForecastPair, ...]:
    """Build Date[t] to Date[t+1] targets without look-ahead features."""

    require_chronological_records(records)
    if len(records) < 2:
        raise ValueError("At least two OHLCV records are required for next-day targets")

    pairs = tuple(
        NextDayForecastPair(
            origin_date=origin.trading_date,
            target_date=target.trading_date,
            origin_close=origin.close,
            actual_close=target.close,
            target_delta=target.close - origin.close,
        )
        for origin, target in zip(records, records[1:])
    )
    LOGGER.info(
        "Built next-day forecast pairs count=%d first_target=%s last_target=%s",
        len(pairs),
        pairs[0].target_date,
        pairs[-1].target_date,
    )
    return pairs
