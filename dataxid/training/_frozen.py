# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Frozen encoder training strategy.

Freeze encoder → encode all batches once → upload → async training → poll.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from dataxid.exceptions import TrainingError, TrainingTimeoutError

if TYPE_CHECKING:
    from dataxid.training._model import Model

logger = logging.getLogger(__name__)

_POLL_INTERVALS = [2, 3, 5, 10, 15, 20, 30]


def train_frozen(
    model: Model,
    batch_size: int,
    max_epochs: int,
    early_stop_patience: int = 4,
    val_split: float = 0.1,
) -> None:
    """Freeze encoder, encode all batches once, train asynchronously."""
    initial_lr = model._config["learning_rate"] or float(
        np.round(0.001 * np.sqrt(batch_size / 32), 5)
    )

    model._encoder.freeze()
    model._encoder.eval_mode()
    logger.info("Encoder frozen (requires_grad=False)")

    logger.info("Encoding all batches...")
    batches = model._encoder._encode_batches(
        batch_size=batch_size, val_split=val_split,
    )
    train_count = sum(1 for b in batches if not b["is_validation"])
    val_count = sum(1 for b in batches if b["is_validation"])
    logger.info("Encoded %d train + %d val batches", train_count, val_count)

    model._client.post(f"/v1/models/{model.id}/init-training", json={
        "max_epochs": max_epochs,
        "patience": early_stop_patience,
        "learning_rate": initial_lr,
        "label_smoothing": model._config["label_smoothing"],
        "embedding_dropout": model._config["embedding_dropout"],
        "time_limit_seconds": model._config["time_limit_seconds"],
        "batches": batches,
    }, idempotent=False)

    logger.info("Starting async training: max_epochs=%d", max_epochs)
    resp = model._client.post(
        f"/v1/models/{model.id}/train", json={}, idempotent=False,
    )

    if resp.get("data", {}).get("epochs") is not None:
        _process_sync_result(model, resp["data"], max_epochs)
        return

    _poll_training(model, timeout=model._config["timeout"], max_epochs=max_epochs)


def _poll_training(model: Model, timeout: float, max_epochs: int) -> None:
    """Poll model status until training completes or fails."""
    started = time.monotonic()
    poll_idx = 0

    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            raise TrainingTimeoutError(
                f"Training did not complete within {timeout:.0f}s",
            )

        interval = _POLL_INTERVALS[min(poll_idx, len(_POLL_INTERVALS) - 1)]
        time.sleep(interval)
        poll_idx += 1

        resp = model._client.get(f"/v1/models/{model.id}")
        data = resp["data"]
        status = data.get("status")
        epoch = data.get("current_epoch", 0)
        t_loss = data.get("train_loss")
        v_loss = data.get("val_loss")

        if epoch > 0 and t_loss is not None:
            val_str = f"{v_loss:.4f}" if v_loss else "n/a"
            logger.info(
                "Training: epoch=%d/%d, train=%.4f, val=%s, elapsed=%.0fs",
                epoch, max_epochs, t_loss, val_str, elapsed,
            )
        else:
            logger.info("Polling training: status=%s, elapsed=%.0fs", status, elapsed)

        if status == "ready":
            result = data.get("training_result") or {}
            _process_async_result(model, result, max_epochs)
            return

        if status == "failed":
            error = data.get("error", "Unknown training error")
            raise TrainingError(f"Training failed: {error}")


def _process_async_result(model: Model, result: dict, max_epochs: int) -> None:  # noqa: ARG001
    """Extract final metrics from training_result (async path)."""
    model.train_losses = []
    model.val_losses = []
    model.stopped_early = result.get("early_stopped", False)

    final_train = result.get("train_loss") or 0
    final_val = result.get("val_loss") or 0

    if final_train:
        model.train_losses.append(final_train)
    if final_val:
        model.val_losses.append(final_val)

    logger.info(
        "Training completed: %d epochs, train_loss=%.4f, val_loss=%.4f, "
        "early_stopped=%s, duration=%.1fs",
        result.get("epochs", 0), final_train, final_val,
        model.stopped_early, result.get("duration_seconds", 0),
    )
    model.status = "ready"


def _process_sync_result(model: Model, data: dict, max_epochs: int) -> None:
    """Process result from synchronous training response."""
    model.train_losses = []
    model.val_losses = []
    model.stopped_early = data.get("early_stopped", False)

    for e in data.get("epoch_history", []):
        t_loss = e.get("train_loss")
        v_loss = e.get("val_loss")
        if t_loss is not None:
            model.train_losses.append(t_loss)
        if v_loss is not None:
            model.val_losses.append(v_loss)

        ckpt = " *" if e.get("is_checkpoint") else ""
        logger.info(
            "Epoch %s/%d: train=%.4f, val=%.4f, lr=%.6f%s",
            e.get("epoch", "?"), max_epochs,
            t_loss or 0, v_loss or 0, e.get("learning_rate", 0), ckpt,
        )

    final_train = data.get("train_loss") or 0
    final_val = data.get("val_loss") or 0
    logger.info(
        "Training completed: %d epochs, train_loss=%.4f, val_loss=%.4f, "
        "early_stopped=%s",
        data.get("epochs", 0), final_train, final_val, model.stopped_early,
    )
    model.status = "ready"
