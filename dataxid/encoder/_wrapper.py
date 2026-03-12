# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Encoder wrapper — manages the encoding lifecycle and wire-format conversion.

Raw data stays local. Only metadata, embeddings, and targets are sent to the API.
"""

import json
import math
import random
from typing import Any

import pandas as pd
import torch

from dataxid.client._serialization import (
    deserialize_encoder_state,
    serialize_embedding,
    serialize_encoder_state,
)
from dataxid.encoder._ports import EncoderPort
from dataxid.exceptions import ModelNotReadyError

_DEFAULT_LR = 1e-3


def _create_backend() -> EncoderPort:
    from dataxid.encoder._builtin import BuiltinEncoder
    return BuiltinEncoder()


def _sanitize_null_bytes(obj: Any) -> Any:
    """Ensure metadata is wire-safe by stripping null bytes from JSON keys/values."""
    s = json.dumps(obj)
    s = s.replace("\\u0000", "")
    return json.loads(s)


class Encoder:
    """Client-side encoder. Raw data never leaves this class."""

    def __init__(
        self,
        embedding_dim: int = 64,
        model_size: str = "medium",
        privacy_enabled: bool = False,
        privacy_noise: float = 0.1,
        device: str = "cpu",
    ):
        self._embedding_dim = embedding_dim
        self._model_size = model_size
        self._privacy_enabled = privacy_enabled
        self._privacy_noise = privacy_noise
        self._device = device
        self._backend: EncoderPort | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._tensors: dict[str, torch.Tensor] | None = None
        self._live_embeddings: dict[int, torch.Tensor] = {}
        self._embed_counter: int = 0
        self.column_stats: dict[str, dict] = {}
        self.features: list[str] = []

    def analyze(
        self,
        df: pd.DataFrame,
        encoding_types: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Analyze raw data and return metadata for model creation."""
        features = list(df.columns)

        self._backend = _create_backend()
        self._backend.analyze(
            df=df,
            features=features,
            embedding_dim=self._embedding_dim,
            model_size=self._model_size,
            device=torch.device(self._device),
            encoding_types=encoding_types,
        )

        self._optimizer = torch.optim.AdamW(
            self._backend.parameters(),
            lr=_DEFAULT_LR,
        )

        empirical_probs = self._backend._compute_priors(df)

        self.features = features
        self.column_stats = self._backend._column_stats()

        return _sanitize_null_bytes({
            "cardinalities": self._backend._vocab_sizes(),
            "features": features,
            "column_stats": self.column_stats,
            "value_mappings": self._backend._value_mappings(),
            "empirical_probs": {
                k: v.tolist() for k, v in empirical_probs.items()
            },
        })

    def prepare(self, df: pd.DataFrame) -> None:
        """Pre-encode data to tensors. Call once, reuse across epochs."""
        self._check_ready()
        self._tensors = self._backend._prepare_tensors(df)

    def encode_batch(
        self,
        indices: list[int] | None = None,
        add_noise: bool = False,
    ) -> tuple[dict[str, Any], dict[str, list[int]], int]:
        """
        Encode a batch → (embedding_payload, targets, embed_id).

        Args:
            indices: Row indices for this batch. None = all rows.
            add_noise: Add Gaussian noise to embedding for privacy.

        Returns:
            (embedding_payload, targets, embed_id)
        """
        self._check_ready()
        if self._tensors is None:
            raise RuntimeError("Call prepare(df) before encode_batch()")

        if indices is not None:
            batch = {k: v[indices] for k, v in self._tensors.items()}
        else:
            batch = self._tensors

        embedding = self._backend._encode_batch(batch)

        if add_noise and self._privacy_enabled:
            embedding = self._add_noise(embedding)

        embed_id = self._embed_counter
        self._embed_counter += 1
        self._live_embeddings[embed_id] = embedding

        targets = {
            sub_col: tensor.squeeze(-1).tolist()
            for sub_col, tensor in batch.items()
        }

        return serialize_embedding(embedding), targets, embed_id

    def _apply_gradient(self, grad: torch.Tensor, embed_id: int | None = None) -> None:
        """Backward pass on the original forward embedding. Accumulates gradient only."""
        self._check_ready()

        if embed_id is None:
            raise RuntimeError("embed_id cannot be None — call encode_batch() first")
        if embed_id not in self._live_embeddings:
            raise RuntimeError(
                f"No live embedding for embed_id={embed_id} — already consumed or never created"
            )
        embedding = self._live_embeddings.pop(embed_id)

        embedding.backward(grad)

    def _discard_embedding(self, embed_id: int) -> None:
        """Discard a stored embedding (val/sentinel batches that skip backward)."""
        self._live_embeddings.pop(embed_id, None)

    def zero_grad(self) -> None:
        """Zero encoder gradients. Called at epoch start and after optimizer step."""
        self._optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        """Step encoder optimizer."""
        self._optimizer.step()

    def train_mode(self) -> None:
        """Set encoder to training mode (dropout active)."""
        self._backend.encoder.train()

    def eval_mode(self) -> None:
        """Set encoder to eval mode (dropout off)."""
        self._backend.encoder.eval()

    def freeze(self) -> None:
        """Freeze encoder parameters (requires_grad=False). No backward pass needed."""
        self._check_ready()
        for p in self._backend.parameters():
            p.requires_grad = False

    def _add_noise(self, embedding: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to embedding for privacy protection."""
        noise = torch.randn_like(embedding) * self._privacy_noise
        return embedding + noise

    def _encode_batches(
        self,
        batch_size: int = 256,
        val_split: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Encode all data into batches (frozen encoder mode)."""
        self._check_ready()
        if self._tensors is None:
            raise RuntimeError("Call prepare(df) before _encode_batches()")

        n_rows = next(iter(self._tensors.values())).shape[0]
        val_size = max(1, round(n_rows * val_split))
        train_size = n_rows - val_size

        all_indices = list(range(n_rows))
        random.shuffle(all_indices)
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]

        batches: list[dict[str, Any]] = []

        n_train_batches = math.ceil(train_size / batch_size)
        with torch.no_grad():
            for i in range(n_train_batches):
                start = i * batch_size
                end = min(start + batch_size, train_size)
                idx = train_indices[start:end]
                embedding, targets, eid = self.encode_batch(idx, add_noise=True)
                self._discard_embedding(eid)
                batches.append({"embedding": embedding, "targets": targets, "is_validation": False})

            val_embedding, val_targets, val_eid = self.encode_batch(val_indices)
            self._discard_embedding(val_eid)
            batches.append({
                "embedding": val_embedding, "targets": val_targets, "is_validation": True,
            })

        return batches

    def _generation_embedding(
        self,
        n_samples: int,
        seed_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Produce embedding for generation. Zero embedding if no seed_data."""
        self._check_ready()

        if seed_data is not None:
            embedding = self._backend.encode(seed_data)
            if self._privacy_enabled:
                embedding = self._add_noise(embedding)
        else:
            embedding = torch.zeros(
                n_samples, self._embedding_dim, device=torch.device(self._device),
            )

        return serialize_embedding(embedding)

    def save_state(self) -> bytes:
        """Serialize encoder weights + optimizer to bytes for checkpoint."""
        self._check_ready()
        return serialize_encoder_state(self._backend, self._optimizer)

    def load_state(self, data: bytes) -> None:
        """Restore encoder weights + optimizer from checkpoint bytes."""
        self._check_ready()
        deserialize_encoder_state(data, self._backend, self._optimizer)

    def _check_ready(self) -> None:
        if self._backend is None:
            raise ModelNotReadyError("Call analyze(df) first.")
