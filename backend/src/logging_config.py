"""Structured console logging shared by backend command-line entry points."""

from datetime import datetime
import json
import logging

from config.settings import MANILA_TIMEZONE


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
    """Configure root logging once for an orchestration command."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
