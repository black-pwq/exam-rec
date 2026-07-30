"""Application logging configuration."""

from __future__ import annotations

import logging
import os
import sys


APP_LOGGER_NAME = "exam_rec"
_HANDLER_NAME = "exam_rec.stdout"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging() -> None:
    level_name = os.getenv("EXAM_REC_LOG_LEVEL", "INFO").strip().upper()
    try:
        level = _LOG_LEVELS[level_name]
    except KeyError:
        supported = ", ".join(_LOG_LEVELS)
        raise RuntimeError(
            f"invalid EXAM_REC_LOG_LEVEL: {level_name!r}; "
            f"expected one of: {supported}"
        ) from None

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False

    handler = next(
        (
            candidate
            for candidate in app_logger.handlers
            if candidate.get_name() == _HANDLER_NAME
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        )
        app_logger.addHandler(handler)
    handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    logger_name = (
        name
        if name == APP_LOGGER_NAME or name.startswith(f"{APP_LOGGER_NAME}.")
        else f"{APP_LOGGER_NAME}.{name}"
    )
    return logging.getLogger(logger_name)
