# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Encoder wrapper — manages the encoding lifecycle and API payload preparation.

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
        protect_rare: bool = True,
        device: str = "cpu",
    ):
        self._embedding_dim = embedding_dim
        self._model_size = model_size
        self._privacy_enabled = privacy_enabled
        self._privacy_noise = privacy_noise
        self._protect_rare = protect_rare
        self._device = device
        self._backend: EncoderPort | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._tensors: dict[str, torch.Tensor] | None = None
        self._ctx_tensors: dict[str, torch.Tensor] | None = None
        self._live_embeddings: dict[int, torch.Tensor] = {}
        self._embed_counter: int = 0
        self.column_stats: dict[str, dict] = {}
        self.ctx_column_stats: dict[str, dict] = {}
        self.features: list[str] = []
        self.ctx_features: list[str] = []
        self.has_context: bool = False
        self.is_sequential: bool = False
        self.seq_len_max: int = 1
        self.seq_len_median: int = 1
        self._foreign_key: str | None = None
        self._parent_key: str | None = None
        self._seq_data: dict[str, list] | None = None

    @property
    def protect_rare_enabled(self) -> bool:
        """Whether rare-category protection is active for this encoder."""
        return self._protect_rare

    def analyze(
        self,
        df: pd.DataFrame,
        encoding_types: dict[str, str] | None = None,
        parent: pd.DataFrame | None = None,
        parent_encoding_types: dict[str, str] | None = None,
        foreign_key: str | None = None,
        parent_key: str | None = None,
    ) -> dict[str, Any]:
        """Analyze raw data (and optional parent table) → return metadata for model creation.

        When parent is provided, parent columns are analyzed and compressed
        via FeatureCompressor inside the same encoder — producing a combined embedding.

        When foreign_key is provided and rows are grouped (1:N), sequential mode
        is detected automatically: seq_len_max/median computed, positional cardinalities added.
        """
        from dataxid.encoder._nn import get_positional_cardinalities

        self._foreign_key = foreign_key

        exclude_cols: set[str] = set()
        if foreign_key:
            exclude_cols.add(foreign_key)
        features = [c for c in df.columns if c not in exclude_cols]
        self.features = features

        if parent is not None:
            ctx_exclude: set[str] = set()
            if parent_key:
                ctx_exclude.add(parent_key)
            if foreign_key and foreign_key in parent.columns:
                ctx_exclude.add(foreign_key)
            self.ctx_features = [c for c in parent.columns if c not in ctx_exclude]
            self.has_context = True

        if foreign_key and foreign_key in df.columns:
            seq_lens = df.groupby(foreign_key).size()
            self.is_sequential = True
            self.seq_len_max = int(seq_lens.max())
            self.seq_len_median = int(seq_lens.median())

        self._backend = _create_backend()
        ctx_for_analyze = parent[self.ctx_features] if parent is not None else None
        self._backend.analyze(
            df=df[features],
            features=features,
            embedding_dim=self._embedding_dim,
            model_size=self._model_size,
            device=torch.device(self._device),
            encoding_types=encoding_types,
            parent=ctx_for_analyze,
            parent_encoding_types=parent_encoding_types,
            protect_rare=self._protect_rare,
        )

        self._optimizer = torch.optim.AdamW(
            self._backend.parameters(),
            lr=_DEFAULT_LR,
        )

        self.column_stats = self._backend._column_stats()

        cardinalities = self._backend._vocab_sizes()
        empirical_probs = self._backend._compute_priors(df[features])
        all_empirical_probs = {k: v.tolist() for k, v in empirical_probs.items()}
        all_value_mappings = self._backend._value_mappings()

        if parent is not None:
            ctx_probs = self._backend._compute_ctx_priors(ctx_for_analyze)
            all_empirical_probs.update({k: v.tolist() for k, v in ctx_probs.items()})
            self.ctx_column_stats = {
                k: v for k, v in self.column_stats.items()
                if k in self.ctx_features
            }

        if self.has_context and self._backend.ctx_schema is not None:
            ctx_sub_cols = set(self._backend.ctx_schema.cardinalities.keys())
            cardinalities = {k: v for k, v in cardinalities.items() if k not in ctx_sub_cols}
            all_empirical_probs = {k: v for k, v in all_empirical_probs.items() if k not in ctx_sub_cols}
            all_value_mappings = {k: v for k, v in all_value_mappings.items() if k not in ctx_sub_cols}

        if self.is_sequential:
            pos_cards = get_positional_cardinalities(self.seq_len_max)
            cardinalities = {**pos_cards, **cardinalities}
            for sc, card in pos_cards.items():
                all_empirical_probs[sc] = [1.0 / card] * card

        return _sanitize_null_bytes({
            "cardinalities": cardinalities,
            "features": features,
            "column_stats": self.column_stats,
            "value_mappings": all_value_mappings,
            "empirical_probs": all_empirical_probs,
            "has_context": self.has_context,
            "is_sequential": self.is_sequential,
            "seq_len_max": self.seq_len_max,
            "seq_len_median": self.seq_len_median,
        })

    def prepare(
        self,
        df: pd.DataFrame,
        parent: pd.DataFrame | None = None,
        parent_key: str | None = None,
    ) -> None:
        """Pre-encode data to tensors. Call once, reuse across epochs."""
        self._check_ready()
        self._parent_key = parent_key

        if self.is_sequential and self._foreign_key:
            self._seq_data = self._backend._prepare_sequential_tensors(
                df,
                group_by=self._foreign_key,
                seq_len_max=self.seq_len_max,
                parent=parent,
                parent_key=parent_key,
            )
            self._seq_ctx_embeddings = self._precompute_seq_ctx_embeddings(
                df, parent, parent_key,
            )
            return

        self._tensors = self._backend._prepare_tensors(df[self.features])
        if parent is not None and self.has_context:
            self._ctx_tensors = self._backend._prepare_ctx_tensors(parent)

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

        ctx_batch = None
        if self._ctx_tensors is not None:
            if indices is not None:
                ctx_batch = {k: v[indices] for k, v in self._ctx_tensors.items()}
            else:
                ctx_batch = self._ctx_tensors

        embedding = self._backend._encode_batch(batch, ctx=ctx_batch)

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

        if self.is_sequential:
            return self._encode_batches_sequential(batch_size, val_split)

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

    @torch.no_grad()
    def _precompute_seq_ctx_embeddings(
        self,
        df: pd.DataFrame,
        parent: pd.DataFrame | None,
        parent_key: str | None,
    ) -> torch.Tensor:
        """Pre-compute frozen context embeddings aligned with entity order in _seq_data."""
        first_key = next(iter(self._seq_data))
        n_entities = len(self._seq_data[first_key])
        compressor = self._backend.encoder.ctx_compressor
        ctx_dim = compressor.dim_output if compressor is not None else 0

        if parent is None or compressor is None or not self.ctx_features:
            return torch.zeros(n_entities, ctx_dim, device=torch.device(self._device))

        child_keys = list(df.groupby(self._foreign_key, sort=False).groups.keys())
        if parent_key:
            all_parent_keys = list(parent[parent_key].unique())
            child_set = set(child_keys)
            missing_keys = [k for k in all_parent_keys if k not in child_set]
            entity_keys = child_keys + missing_keys
            ctx_aligned = (
                parent.set_index(parent_key).loc[entity_keys].reset_index(drop=True)
            )
        else:
            ctx_aligned = parent.head(n_entities).reset_index(drop=True)

        ctx_clean = ctx_aligned[self.ctx_features]
        return self._backend.encode_context_only(ctx_clean)

    def _encode_batches_sequential(
        self,
        batch_size: int = 256,
        val_split: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Encode sequential data into batches. No target encoder — context-only embedding."""
        if self._seq_data is None:
            raise RuntimeError("Call prepare(df) before _encode_batches()")

        first_key = next(iter(self._seq_data))
        n_entities = len(self._seq_data[first_key])
        val_size = max(1, round(n_entities * val_split))
        train_size = n_entities - val_size

        all_indices = list(range(n_entities))
        random.shuffle(all_indices)
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:]

        ctx_sub_cols = set(self._backend.ctx_schema.cardinalities.keys()) if self._backend.ctx_schema else set()
        seq_keys = [k for k in self._seq_data if k not in ctx_sub_cols]

        def _make_batch(indices: list[int]) -> dict[str, Any]:
            targets: dict[str, list] = {}
            for k in seq_keys:
                targets[k] = [self._seq_data[k][i] for i in indices]

            embedding = self._seq_ctx_embeddings[indices]
            return {
                "embedding": serialize_embedding(embedding),
                "targets": targets,
                "is_sequential": True,
            }

        batches: list[dict[str, Any]] = []
        n_train_batches = math.ceil(train_size / batch_size)
        for i in range(n_train_batches):
            start = i * batch_size
            end = min(start + batch_size, train_size)
            idx = train_indices[start:end]
            batch = _make_batch(idx)
            batch["is_validation"] = False
            batches.append(batch)

        val_batch = _make_batch(val_indices)
        val_batch["is_validation"] = True
        batches.append(val_batch)

        return batches

    def _generation_embedding(
        self,
        n_samples: int,
        conditions: pd.DataFrame | None = None,
        parent: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Produce the embedding used to drive generation.

        Sequential mode: context-only embedding derived from the parent table.
        Non-sequential mode: full embedding (target + optional context) when
        conditions are provided, otherwise a zero embedding of the right size.
        """
        self._check_ready()

        if self.is_sequential:
            if parent is not None and self.has_context:
                ctx_features_only = parent[self.ctx_features] if self.ctx_features else parent
                embedding = self._backend.encode_context_only(ctx_features_only)
            else:
                ctx_dim = self._backend.encoder.ctx_compressor.dim_output if (  # type: ignore[union-attr]
                    self._backend.encoder.ctx_compressor is not None  # type: ignore[union-attr]
                ) else 0
                n_entities = n_samples
                embedding = torch.zeros(
                    n_entities, ctx_dim, device=torch.device(self._device),
                )
            return serialize_embedding(embedding)

        if conditions is not None:
            ctx_clean = parent[self.ctx_features] if (parent is not None and self.ctx_features) else parent
            embedding = self._backend.encode(conditions, ctx_df=ctx_clean)
            if self._privacy_enabled:
                embedding = self._add_noise(embedding)
        else:
            embed_dim = self._embedding_dim
            embedding = torch.zeros(
                n_samples, embed_dim, device=torch.device(self._device),
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
