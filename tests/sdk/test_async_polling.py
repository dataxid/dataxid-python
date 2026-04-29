# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the SDK's frozen-encoder training poll loop.

The training endpoint is asynchronous: ``POST /train`` returns ``status:
training`` immediately, after which the SDK long-polls ``GET /models/{id}``
until the server reports ``status: ready`` or ``failed``. A legacy synchronous
response (with ``epochs`` already populated) is still accepted for backward
compatibility.

Covered:

* :func:`_poll_training` — interval backoff, timeout, status transitions.
* :func:`_process_async_result` — metrics extraction from the final payload.
* :func:`_process_sync_result` — backward-compat path with epoch history.
* Per-epoch progress logging during long polls.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from dataxid.exceptions import TrainingError, TrainingTimeoutError
from dataxid.training._config import ModelConfig
from dataxid.training._frozen import (
    _POLL_INTERVALS,
    _poll_training,
    _process_async_result,
    _process_sync_result,
)
from dataxid.training._model import Model

_FROZEN_LOGGER = "dataxid.training._frozen"


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _make_model(get_responses: list[dict[str, Any]] | None = None) -> Model:
    """Build a :class:`Model` with a mocked HTTP client.

    ``get_responses`` is consumed in order by successive ``client.get`` calls;
    pass ``None`` (or omit) when the test does not exercise the polling loop.
    """
    client = MagicMock()
    client.get = MagicMock(side_effect=list(get_responses or []))
    client.post = MagicMock(return_value={"data": {"status": "training"}})

    model = Model.__new__(Model)
    model.id = "mdl_test123"
    model._client = client
    model._config = {}
    model.status = "training"
    model.train_losses = []
    model.val_losses = []
    model.stopped_early = False
    return model


@pytest.fixture()
def mock_sleep(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace :func:`time.sleep` inside the polling module with a no-op."""
    fake = MagicMock()
    monkeypatch.setattr("dataxid.training._frozen.time.sleep", fake)
    return fake


@contextmanager
def _capture_logs(logger_name: str) -> Iterator[list[logging.LogRecord]]:
    """Attach a temporary handler to ``logger_name`` and yield its records.

    Implemented directly (rather than via :func:`caplog`) so the assertion is
    not affected by logger state set up elsewhere in the suite (handlers,
    levels, ``disabled`` flags, ``propagate`` flags on parent loggers).
    """
    logger = logging.getLogger(logger_name)
    records: list[logging.LogRecord] = []

    handler = logging.Handler(level=logging.INFO)
    handler.emit = records.append  # type: ignore[method-assign]

    prev_level = logger.level
    prev_disabled = logger.disabled
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.disabled = prev_disabled


# ---------------------------------------------------------------------------
# _poll_training — control flow
# ---------------------------------------------------------------------------


class TestPollTraining:
    """Polling loop status transitions, timeout, and backoff."""

    def test_polls_until_ready(self, mock_sleep: MagicMock) -> None:
        responses = [
            {"data": {"status": "training"}},
            {"data": {"status": "training"}},
            {
                "data": {
                    "status": "ready",
                    "training_result": {
                        "epochs": 10,
                        "train_loss": 0.15,
                        "val_loss": 0.18,
                        "early_stopped": False,
                        "duration_seconds": 5.2,
                    },
                }
            },
        ]
        model = _make_model(responses)
        _poll_training(model, timeout=60, max_epochs=10)

        assert model.status == "ready"
        assert model.stopped_early is False
        assert model._client.get.call_count == 3

    def test_raises_training_error_on_failed(self, mock_sleep: MagicMock) -> None:
        responses = [
            {"data": {"status": "training"}},
            {"data": {"status": "failed", "error": "OOM on GPU worker"}},
        ]
        model = _make_model(responses)

        with pytest.raises(TrainingError, match="OOM on GPU worker"):
            _poll_training(model, timeout=60, max_epochs=10)

    def test_raises_timeout_error(
        self,
        mock_sleep: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When elapsed time exceeds ``timeout`` the loop raises before the
        next GET, surfacing the configured timeout in the error message."""
        ticks = iter([0.0, 0.0, 100.0])
        monkeypatch.setattr(
            "dataxid.training._frozen.time.monotonic", lambda: next(ticks)
        )
        model = _make_model([{"data": {"status": "training"}}])

        with pytest.raises(TrainingTimeoutError, match="10"):
            _poll_training(model, timeout=10, max_epochs=100)

    def test_backoff_uses_documented_intervals(self, mock_sleep: MagicMock) -> None:
        """Sleep durations follow ``_POLL_INTERVALS`` in order."""
        responses = [
            {"data": {"status": "training"}},
            {"data": {"status": "training"}},
            {"data": {"status": "training"}},
            {
                "data": {
                    "status": "ready",
                    "training_result": {"epochs": 5, "train_loss": 0.1},
                }
            },
        ]
        model = _make_model(responses)
        _poll_training(model, timeout=300, max_epochs=5)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls[:3] == _POLL_INTERVALS[:3]


# ---------------------------------------------------------------------------
# _poll_training — log output
# ---------------------------------------------------------------------------


class TestPollTrainingLogs:
    """Per-poll log lines reflect server-reported epoch progress."""

    def test_logs_per_epoch_progress_when_server_reports_it(
        self, mock_sleep: MagicMock
    ) -> None:
        responses = [
            {
                "data": {
                    "status": "training",
                    "current_epoch": 2,
                    "train_loss": 0.35,
                    "val_loss": 0.40,
                }
            },
            {
                "data": {
                    "status": "training",
                    "current_epoch": 4,
                    "train_loss": 0.20,
                    "val_loss": 0.25,
                }
            },
            {
                "data": {
                    "status": "ready",
                    "current_epoch": 5,
                    "train_loss": 0.15,
                    "val_loss": 0.18,
                    "training_result": {
                        "epochs": 5,
                        "train_loss": 0.15,
                        "val_loss": 0.18,
                    },
                }
            },
        ]
        model = _make_model(responses)

        with _capture_logs(_FROZEN_LOGGER) as records:
            _poll_training(model, timeout=60, max_epochs=5)

        rendered = [r.getMessage() for r in records]
        assert any("epoch=2/5" in m for m in rendered)
        assert any("epoch=4/5" in m for m in rendered)

    def test_logs_generic_status_when_server_omits_epoch(
        self, mock_sleep: MagicMock
    ) -> None:
        responses = [
            {"data": {"status": "training"}},
            {
                "data": {
                    "status": "ready",
                    "training_result": {"epochs": 3, "train_loss": 0.2},
                }
            },
        ]
        model = _make_model(responses)

        with _capture_logs(_FROZEN_LOGGER) as records:
            _poll_training(model, timeout=60, max_epochs=3)

        rendered = [r.getMessage() for r in records]
        assert any("Polling training" in m and "status=training" in m for m in rendered)


# ---------------------------------------------------------------------------
# _process_async_result
# ---------------------------------------------------------------------------


class TestProcessAsyncResult:
    """Final-payload metrics extraction for the async path."""

    def test_extracts_metrics(self) -> None:
        model = _make_model()
        result = {
            "epochs": 42,
            "train_loss": 0.123,
            "val_loss": 0.156,
            "early_stopped": True,
            "duration_seconds": 120.5,
        }
        _process_async_result(model, result, max_epochs=100)

        assert model.status == "ready"
        assert model.stopped_early is True
        assert model.train_losses == [0.123]
        assert model.val_losses == [0.156]

    def test_handles_missing_fields(self) -> None:
        """An empty payload still leaves the model in a coherent ``ready``
        state — no losses recorded, ``stopped_early`` defaulting to False."""
        model = _make_model()
        _process_async_result(model, {}, max_epochs=100)

        assert model.status == "ready"
        assert model.stopped_early is False
        assert model.train_losses == []
        assert model.val_losses == []


# ---------------------------------------------------------------------------
# _process_sync_result (backward compat)
# ---------------------------------------------------------------------------


class TestProcessSyncResult:
    """Legacy sync response path: per-epoch history is replayed in order."""

    def test_processes_epoch_history(self) -> None:
        model = _make_model()
        data = {
            "epochs": 3,
            "train_loss": 0.2,
            "val_loss": 0.25,
            "early_stopped": False,
            "epoch_history": [
                {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "learning_rate": 0.001},
                {"epoch": 2, "train_loss": 0.3, "val_loss": 0.4, "learning_rate": 0.001},
                {"epoch": 3, "train_loss": 0.2, "val_loss": 0.25, "learning_rate": 0.0005},
            ],
        }
        _process_sync_result(model, data, max_epochs=10)

        assert model.status == "ready"
        assert model.train_losses == [0.5, 0.3, 0.2]
        assert model.val_losses == [0.6, 0.4, 0.25]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestPollIntervals:
    """``_POLL_INTERVALS`` is monotonically non-decreasing — required for the
    intended exponential-style backoff. The matching default training timeout
    on :class:`ModelConfig` is also pinned to guard against accidental edits."""

    def test_intervals_monotonically_non_decreasing(self) -> None:
        assert _POLL_INTERVALS == sorted(_POLL_INTERVALS)
        assert all(v > 0 for v in _POLL_INTERVALS)

    def test_default_timeout_matches_session_config(self) -> None:
        assert ModelConfig().timeout == 14400.0


# ---------------------------------------------------------------------------
# Backoff and pacing invariants
# ---------------------------------------------------------------------------


class TestPollBackoffInvariants:
    """Pacing contracts the loop must keep regardless of training duration."""

    def test_backoff_clamps_to_last_interval_for_long_polls(
        self, mock_sleep: MagicMock
    ) -> None:
        """Once the index walks past ``_POLL_INTERVALS``, the loop must reuse
        the last value indefinitely — not raise :class:`IndexError`."""
        n_polls = len(_POLL_INTERVALS) + 4
        responses = [{"data": {"status": "training"}}] * n_polls + [
            {
                "data": {
                    "status": "ready",
                    "training_result": {"epochs": 1, "train_loss": 0.1},
                }
            }
        ]
        model = _make_model(responses)
        _poll_training(model, timeout=10000, max_epochs=1)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls[: len(_POLL_INTERVALS)] == _POLL_INTERVALS
        # Every poll past the table reuses the last interval — the plateau.
        # Five extra sleeps: 4 "training" responses + 1 before the final
        # "ready" GET (the loop sleeps before every GET, including the last).
        assert sleep_calls[len(_POLL_INTERVALS) :] == [_POLL_INTERVALS[-1]] * 5

    def test_sleeps_at_least_once_before_first_get(
        self, mock_sleep: MagicMock
    ) -> None:
        """Even when the very first poll returns ``ready``, the loop sleeps
        once so the SDK doesn't spam the server with back-to-back requests
        immediately after ``POST /train``."""
        responses = [
            {
                "data": {
                    "status": "ready",
                    "training_result": {"epochs": 1, "train_loss": 0.1},
                }
            }
        ]
        model = _make_model(responses)
        _poll_training(model, timeout=60, max_epochs=1)

        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0].args[0] == _POLL_INTERVALS[0]


# ---------------------------------------------------------------------------
# None-vs-value semantics in result extraction
# ---------------------------------------------------------------------------


class TestAsyncResultLossSemantics:
    """``None`` (missing) and concrete float values are kept distinct.

    The previous implementation conflated ``None`` and ``0.0`` via
    ``or 0`` + truthy filtering, silently dropping legitimate zero losses.
    These tests pin the corrected ``is not None`` semantics so the regression
    cannot return.
    """

    def test_zero_loss_is_recorded(self) -> None:
        """A reported ``train_loss=0.0`` is a legitimate measurement (perfect
        fit, single-class debug runs) and must reach ``model.train_losses``."""
        model = _make_model()
        _process_async_result(
            model,
            {"epochs": 5, "train_loss": 0.0, "val_loss": 0.0},
            max_epochs=10,
        )
        assert model.train_losses == [0.0]
        assert model.val_losses == [0.0]

    def test_missing_loss_is_skipped(self) -> None:
        """Absent / ``None`` losses leave the lists empty — the model stays
        in a coherent ``ready`` state without spurious zero entries."""
        model = _make_model()
        _process_async_result(
            model,
            {"epochs": 5, "train_loss": None, "val_loss": None},
            max_epochs=10,
        )
        assert model.train_losses == []
        assert model.val_losses == []
        assert model.status == "ready"

    def test_missing_loss_does_not_crash_log_formatting(self) -> None:
        """The completion log must format missing losses safely (``n/a``);
        an unguarded ``%.4f`` against ``None`` would raise :class:`TypeError`."""
        model = _make_model()
        with _capture_logs(_FROZEN_LOGGER) as records:
            _process_async_result(
                model,
                {"epochs": 5, "train_loss": None, "val_loss": None},
                max_epochs=10,
            )
        rendered = [r.getMessage() for r in records]
        completion = next((m for m in rendered if "Training completed" in m), None)
        assert completion is not None
        assert "n/a" in completion
