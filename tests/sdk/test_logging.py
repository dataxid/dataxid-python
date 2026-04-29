# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the SDK logging surface.

The ``dataxid`` logger is opt-in: importing the package leaves the logger
silent (``NullHandler`` only) so the SDK never spams a host application's
log stream by accident. Tests cover the public ``enable_logging`` /
``disable_logging`` toggle, the ``DATAXID_LOG`` environment hook, and the
``SensitiveHeadersFilter`` that masks API credentials before they reach
any log sink.
"""

import logging
from typing import Any

import pytest

import dataxid
from dataxid._log import SensitiveHeadersFilter, logger, setup_logging


@pytest.fixture(autouse=True)
def _clean_logger() -> None:
    """Reset the package logger before and after each test.

    The ``dataxid`` logger is module-level global state, so without explicit
    isolation a test that calls ``enable_logging`` would leak handlers and
    filters into every test that ran afterwards.
    """
    dataxid.disable_logging()
    logger.handlers = [logging.NullHandler()]
    yield
    dataxid.disable_logging()
    logger.handlers = [logging.NullHandler()]


def _make_record(args: Any) -> logging.LogRecord:
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    record.args = args
    return record


def _stream_handlers() -> list[logging.Handler]:
    return [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.NullHandler)
    ]


class TestEnableLogging:
    """``enable_logging`` attaches a ``StreamHandler`` and credential filter."""

    def test_sets_level(self) -> None:
        dataxid.enable_logging("debug")
        assert logger.level == logging.DEBUG

    def test_level_is_case_insensitive(self) -> None:
        dataxid.enable_logging("WARNING")
        assert logger.level == logging.WARNING

    def test_attaches_sensitive_headers_filter(self) -> None:
        dataxid.enable_logging("info")
        assert any(
            isinstance(f, SensitiveHeadersFilter) for f in logger.filters
        )

    def test_idempotent_does_not_duplicate_handlers_or_filters(self) -> None:
        """Repeated calls update the level but never stack handlers — a host
        application that toggles verbosity at runtime must not see duplicate
        log lines."""
        dataxid.enable_logging("info")
        dataxid.enable_logging("debug")
        dataxid.enable_logging("warning")
        assert len(_stream_handlers()) == 1
        sensitive_filters = [
            f for f in logger.filters if isinstance(f, SensitiveHeadersFilter)
        ]
        assert len(sensitive_filters) == 1
        assert logger.level == logging.WARNING

    def test_invalid_string_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            dataxid.enable_logging("invalid")

    def test_int_level_raises_value_error(self) -> None:
        """The public contract is string-only; passing ``logging.DEBUG`` (an
        int) must raise ``ValueError`` rather than fail with an unrelated
        ``AttributeError`` that callers would have to catch by accident."""
        with pytest.raises(ValueError, match="Invalid log level"):
            dataxid.enable_logging(logging.DEBUG)


class TestDisableLogging:
    """``disable_logging`` returns the logger to its silent default."""

    def test_resets_level_to_notset(self) -> None:
        dataxid.enable_logging("info")
        dataxid.disable_logging()
        assert logger.level == logging.NOTSET

    def test_removes_stream_handlers(self) -> None:
        dataxid.enable_logging("info")
        dataxid.disable_logging()
        assert _stream_handlers() == []

    def test_keeps_null_handler_so_logger_stays_silent(self) -> None:
        dataxid.enable_logging("info")
        dataxid.disable_logging()
        assert any(
            isinstance(h, logging.NullHandler) for h in logger.handlers
        )

    def test_clears_sensitive_headers_filter(self) -> None:
        dataxid.enable_logging("info")
        dataxid.disable_logging()
        assert logger.filters == []

    def test_idempotent_when_logging_was_never_enabled(self) -> None:
        """Tools that always call ``disable_logging`` on shutdown should not
        crash if logging was never turned on in the first place."""
        dataxid.disable_logging()
        dataxid.disable_logging()
        assert logger.level == logging.NOTSET


class TestSensitiveHeadersFilter:
    """``SensitiveHeadersFilter`` redacts API credentials in log records.

    This filter is the only thing standing between an Authorization header
    set by ``DataxidClient`` and a host application's log sink, so it must
    handle every realistic shape of ``record.args`` without crashing.
    """

    @pytest.mark.parametrize(
        "header_name",
        [
            "Authorization", "AUTHORIZATION", "authorization", "auThOrIzAtIon",
            "X-Api-Key", "X-API-KEY", "x-api-key",
        ],
    )
    def test_masks_credential_headers_case_insensitively(
        self, header_name: str
    ) -> None:
        record = _make_record({"headers": {header_name: "secret_value"}})
        SensitiveHeadersFilter().filter(record)
        assert record.args["headers"][header_name] == "***"

    def test_preserves_non_sensitive_headers(self) -> None:
        record = _make_record(
            {"headers": {"Content-Type": "application/json"}}
        )
        SensitiveHeadersFilter().filter(record)
        assert record.args["headers"]["Content-Type"] == "application/json"

    def test_redacts_sensitive_alongside_safe_headers(self) -> None:
        record = _make_record({
            "headers": {
                "Authorization": "Bearer dx_123",
                "Content-Type": "application/json",
                "X-Api-Key": "dx_secret",
            }
        })
        SensitiveHeadersFilter().filter(record)
        masked = record.args["headers"]
        assert masked["Authorization"] == "***"
        assert masked["X-Api-Key"] == "***"
        assert masked["Content-Type"] == "application/json"

    @pytest.mark.parametrize(
        "args",
        [
            None,
            {"headers": None},
            {"headers": "raw-string-not-a-dict"},
            {"headers": 42},
            {"unrelated": "value"},
            (),
        ],
    )
    def test_passes_through_unsupported_args_without_crashing(
        self, args: Any
    ) -> None:
        """The filter must never break logging; unexpected ``args`` shapes
        flow through untouched."""
        record = _make_record(args)
        assert SensitiveHeadersFilter().filter(record) is True


class TestEnvVar:
    """``DATAXID_LOG`` environment variable activates logging on demand."""

    def test_activates_logging_at_requested_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATAXID_LOG", "debug")
        setup_logging()
        assert logger.level == logging.DEBUG

    def test_unset_env_leaves_logger_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DATAXID_LOG", raising=False)
        setup_logging()
        assert logger.level == logging.NOTSET
        assert _stream_handlers() == []


class TestTopLevelImport:
    """``enable_logging`` / ``disable_logging`` are part of the public API."""

    def test_reexported_at_package_root(self) -> None:
        assert dataxid.enable_logging is dataxid._log.enable_logging
        assert dataxid.disable_logging is dataxid._log.disable_logging

    def test_listed_in_dunder_all(self) -> None:
        assert "enable_logging" in dataxid.__all__
        assert "disable_logging" in dataxid.__all__
