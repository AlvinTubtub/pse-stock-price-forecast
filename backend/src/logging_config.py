"""Structured console logging shared by backend command-line entry points."""

from datetime import datetime
import json
import logging
import sys
from typing import Final

from config.settings import MANILA_TIMEZONE


APPLICATION_HANDLER_MARKER: Final[str] = "_forecastph_structured_handler"


class _CurrentStderr:
    """Resolve stderr at write time so test-capture streams cannot become stale."""

    @property
    def encoding(self) -> str | None:
        return getattr(sys.stderr, "encoding", None)

    def write(self, message: str) -> int:
        return sys.stderr.write(message)

    def flush(self) -> None:
        sys.stderr.flush()


class _ApplicationStreamHandler(logging.StreamHandler):
    """ForecastPH-owned handler that follows the process's current stderr."""

    def __init__(self) -> None:
        super().__init__(_CurrentStderr())
        setattr(self, APPLICATION_HANDLER_MARKER, True)


class JsonLogFormatter(logging.Formatter):
    """Emit one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=MANILA_TIMEZONE
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def configure_structured_logging(*, verbose: bool = False) -> None:
    """Idempotently configure one ForecastPH-owned root console handler."""

    root = logging.getLogger()
    application_handlers = [
        handler
        for handler in root.handlers
        if getattr(handler, APPLICATION_HANDLER_MARKER, False)
    ]
    handler: logging.Handler | None = None
    for existing in application_handlers:
        if handler is None and isinstance(existing, _ApplicationStreamHandler):
            handler = existing
            continue
        root.removeHandler(existing)
        existing.close()
    if handler is None:
        handler = _ApplicationStreamHandler()
        root.addHandler(handler)
    handler.setFormatter(JsonLogFormatter())
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
