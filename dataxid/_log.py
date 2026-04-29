# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os

from dataxid.exceptions import InvalidRequestError

logger: logging.Logger = logging.getLogger("dataxid")
logger.addHandler(logging.NullHandler())

SENSITIVE_HEADERS = {"authorization", "x-api-key"}

_LOG_FORMAT = "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


class SensitiveHeadersFilter(logging.Filter):
    """Mask sensitive headers (API keys, tokens) in log output."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, dict) and "headers" in record.args:
            headers = record.args["headers"]
            if isinstance(headers, dict):
                record.args["headers"] = {
                    k: "***" if k.lower() in SENSITIVE_HEADERS else v
                    for k, v in headers.items()
                }
        return True


def enable_logging(level: str = "info") -> None:
    """Enable SDK logging at the given level.

    Adds a ``StreamHandler`` to the ``dataxid`` logger (not root).
    Safe to call multiple times; existing handlers are replaced.

    Args:
        level: One of "debug", "info", "warning", "error", "critical".

    Raises:
        InvalidRequestError: If *level* is not a valid Python log level
            name. Inherits from :class:`ValueError` for backward
            compatibility with ``except ValueError``.
    """
    if not isinstance(level, str):
        raise InvalidRequestError(
            f"Invalid log level: {level!r}. "
            f"Use one of: debug, info, warning, error, critical.",
            param="level",
        )
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise InvalidRequestError(
            f"Invalid log level: {level!r}. "
            f"Use one of: debug, info, warning, error, critical.",
            param="level",
        )

    logger.setLevel(numeric)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logger.addHandler(handler)

    if not any(isinstance(f, SensitiveHeadersFilter) for f in logger.filters):
        logger.addFilter(SensitiveHeadersFilter())


def disable_logging() -> None:
    """Disable SDK logging and remove all handlers (except NullHandler)."""
    logger.setLevel(logging.NOTSET)
    logger.handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
    logger.filters.clear()


def setup_logging() -> None:
    """Activate logging from ``DATAXID_LOG`` env var (if set)."""
    env = os.environ.get("DATAXID_LOG")
    if env:
        enable_logging(env)
