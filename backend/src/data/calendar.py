"""PSE trading-session calendar helpers."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from config.settings import MANILA_TIMEZONE, as_manila_time
from config.pse_holidays import PSE_CLOSURES


@dataclass(frozen=True, slots=True)
class PSETradingCalendar:
    """Weekend-aware calendar using reviewed PSE closures by default."""

    holidays: frozenset[date] = field(default_factory=lambda: PSE_CLOSURES)

    @classmethod
    def with_holidays(cls, holidays: Iterable[date]) -> "PSETradingCalendar":
        """Add emergency/caller closures without discarding configured dates."""
        return cls(PSE_CLOSURES | frozenset(holidays))

    def is_trading_day(self, candidate: date) -> bool:
        return candidate.weekday() < 5 and candidate not in self.holidays

    def next_trading_day(self, origin: date) -> date:
        """Return the first configured PSE session strictly after origin."""

        candidate = origin + timedelta(days=1)
        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)
        return candidate


def next_pse_trading_day(
    origin: date,
    *,
    holidays: Iterable[date] = (),
) -> date:
    """Convenience wrapper for one-off next-session calculations."""

    return PSETradingCalendar.with_holidays(holidays).next_trading_day(origin)


def timestamp_for_display(value: datetime) -> datetime:
    """Return an aware pipeline timestamp normalized to Asia/Manila."""

    return as_manila_time(value)


DISPLAY_TIMEZONE_NAME = MANILA_TIMEZONE.key
