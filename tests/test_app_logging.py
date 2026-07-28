from __future__ import annotations

import logging
import re

import pytest

from exam_rec.app_logging import APP_LOGGER_NAME, configure_logging, get_logger


@pytest.fixture
def isolated_app_logger():
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    original_handlers = list(app_logger.handlers)
    original_level = app_logger.level
    original_propagate = app_logger.propagate
    app_logger.handlers = []
    app_logger.setLevel(logging.NOTSET)
    app_logger.propagate = True
    try:
        yield app_logger
    finally:
        for handler in app_logger.handlers:
            handler.close()
        app_logger.handlers = original_handlers
        app_logger.setLevel(original_level)
        app_logger.propagate = original_propagate


def test_configure_logging_emits_one_line_and_is_idempotent(
    monkeypatch,
    capsys,
    isolated_app_logger,
) -> None:
    monkeypatch.delenv("EXAM_REC_LOG_LEVEL", raising=False)

    configure_logging()
    configure_logging()
    get_logger("test").info("Recognition job queued job_id=%s", "abc")

    output = capsys.readouterr().out
    assert len(isolated_app_logger.handlers) == 1
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
        r"INFO exam_rec\.test Recognition job queued job_id=abc\n",
        output,
    )


def test_configure_logging_uses_environment_level(
    monkeypatch,
    isolated_app_logger,
) -> None:
    monkeypatch.setenv("EXAM_REC_LOG_LEVEL", "error")

    configure_logging()

    assert get_logger("test").getEffectiveLevel() == logging.ERROR
    assert isolated_app_logger.handlers[0].level == logging.ERROR


def test_configure_logging_rejects_invalid_level(
    monkeypatch,
    isolated_app_logger,
) -> None:
    monkeypatch.setenv("EXAM_REC_LOG_LEVEL", "verbose")

    with pytest.raises(RuntimeError, match="invalid EXAM_REC_LOG_LEVEL"):
        configure_logging()
