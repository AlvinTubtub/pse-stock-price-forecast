"""Filesystem and timezone settings for the ForecastPH backend."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


BACKEND_ROOT = Path(__file__).resolve().parents[1]
MANILA_TIMEZONE = ZoneInfo("Asia/Manila")


@dataclass(frozen=True, slots=True)
class BackendSettings:
    """Authoritative backend paths and display timezone."""

    backend_root: Path = BACKEND_ROOT
    raw_data_dir: Path = BACKEND_ROOT / "data" / "raw"
    artifacts_dir: Path = BACKEND_ROOT / "artifacts"
    display_timezone: ZoneInfo = MANILA_TIMEZONE


SETTINGS = BackendSettings()


def manila_now() -> datetime:
    """Return a timezone-aware current timestamp for displayed metadata."""

    return datetime.now(MANILA_TIMEZONE)


def as_manila_time(value: datetime) -> datetime:
    """Convert an aware timestamp to Asia/Manila.

    Naive datetimes are rejected because assuming a timezone would make displayed
    pipeline timestamps ambiguous.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return value.astimezone(MANILA_TIMEZONE)
