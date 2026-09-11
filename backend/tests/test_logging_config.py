"""Regression tests for CLI-owned structured logging handlers."""

import io
import logging
import sys

import pytest

from src.logging_config import (
    APPLICATION_HANDLER_MARKER,
    JsonLogFormatter,
    configure_structured_logging,
)


@pytest.fixture
def restored_root_logger():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(original_level)


def test_repeated_configuration_keeps_one_application_handler_and_callers(
    restored_root_logger,
) -> None:
    root = restored_root_logger
    caller_handler = logging.NullHandler()
    root.addHandler(caller_handler)

    configure_structured_logging(verbose=False)
    configure_structured_logging(verbose=True)

    application_handlers = [
        handler
        for handler in root.handlers
        if getattr(handler, APPLICATION_HANDLER_MARKER, False)
    ]
    assert caller_handler in root.handlers
    assert len(application_handlers) == 1
    assert isinstance(application_handlers[0].formatter, JsonLogFormatter)
    assert root.level == logging.DEBUG
    root.removeHandler(caller_handler)


def test_application_handler_follows_current_stderr_after_old_stream_closes(
    monkeypatch: pytest.MonkeyPatch,
    restored_root_logger,
) -> None:
    first_stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", first_stream)
    configure_structured_logging()
    logging.getLogger("forecastph.test").info("first message")
    assert "first message" in first_stream.getvalue()

    first_stream.close()
    second_stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", second_stream)
    logging.getLogger("forecastph.test").info("second message")

    assert "second message" in second_stream.getvalue()
