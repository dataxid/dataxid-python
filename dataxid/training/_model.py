# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Model class — training and generation API.

Raw data never leaves the SDK. Only embeddings cross the wire.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch

from dataxid.client._http import DataxidClient
from dataxid.encoder._wrapper import Encoder
from dataxid.exceptions import InvalidRequestError
from dataxid.pipeline._decode import compute_fixed_probs, decode_columns
from dataxid.training._config import ModelConfig, _resolve_config
from dataxid.training._frozen import train_frozen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _infer_parent_key(
    foreign_key: str,
    parent: pd.DataFrame,
) -> str:
    """Infer parent_key from foreign_key if the column exists in parent."""
    if foreign_key in parent.columns:
        return foreign_key
    raise InvalidRequestError(
        f"parent_key not provided and column '{foreign_key}' not found in parent. "
        f"Specify parent_key explicitly when FK and PK column names differ.",
        param="parent_key",
    )


def _validate_context_params(
    data: pd.DataFrame,
    parent: pd.DataFrame | None,
    parent_encoding_types: dict[str, str] | None,
    foreign_key: str | None,
    parent_key: str | None,
) -> None:
    if parent_encoding_types is not None and parent is None:
        raise InvalidRequestError(
            "parent_encoding_types was provided without parent.",
            param="parent_encoding_types",
        )

    if parent_key is not None and parent is None:
        raise InvalidRequestError(
            "parent_key was provided without parent.",
            param="parent_key",
        )

    if foreign_key is not None and foreign_key not in data.columns:
        raise InvalidRequestError(
            f"Column '{foreign_key}' not found in data.",
            param="foreign_key",
        )

    if (
        parent_key is not None
        and parent is not None
        and parent_key not in parent.columns
    ):
        raise InvalidRequestError(
            f"Column '{parent_key}' not found in parent.",
            param="parent_key",
        )

    if parent is not None and foreign_key is None:
        if len(parent) != len(data):
            raise InvalidRequestError(
                f"Row count mismatch: parent has {len(parent)} rows, "
                f"data has {len(data)}. In flat context mode rows must be "
                f"aligned 1:1. For 1:N joins use foreign_key.",
                param="parent",
            )


class Model:
    """A Dataxid model. Trains on your data locally, generates synthetic data via API."""

    def __init__(
        self,
        model_id: str,
        client: DataxidClient,
        encoder: Encoder,
        data: pd.DataFrame,
        config: dict[str, Any],
        parent: pd.DataFrame | None = None,
    ):
        self.id = model_id
        self.status: str = "training"
        self._client = client
        self._encoder = encoder
        self._data = data
        self._parent = parent
        self._config = config
        self._n_samples: int | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.stopped_early: bool = False
        self._best_encoder_state: bytes | None = None
        self._encoder_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        n_samples: int | None = None,
        config: dict[str, Any] | ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        parent: pd.DataFrame | None = None,
        parent_encoding_types: dict[str, str] | None = None,
        foreign_key: str | None = None,
        parent_key: str | None = None,
    ) -> Model:
        """
        Create a model: analyze data locally, register via API, and train.

        Args:
            data: Training DataFrame (stays local — only embeddings sent to API)
            n_samples: Number of synthetic rows (stored for generate())
            config: Training config — ModelConfig instance or plain dict
            api_key: Override dataxid.api_key
            base_url: Override dataxid.base_url
            parent: Parent table for context-aware generation
            parent_encoding_types: Encoding overrides for parent columns
            foreign_key: FK column in data linking rows to parent (enables sequential mode)
            parent_key: PK column in parent table (inferred from foreign_key if same name)

        Returns:
            Trained Model ready for generate()
        """
        config = _resolve_config(config)

        if parent is not None and foreign_key is not None and parent_key is None:
            parent_key = _infer_parent_key(foreign_key, parent)

        _validate_context_params(
            data, parent, parent_encoding_types, foreign_key, parent_key,
        )
        http = DataxidClient(api_key=api_key, base_url=base_url)

        embedding_dim = config["embedding_dim"]
        model_size = config["model_size"]
        batch_size = config["batch_size"]
        max_epochs = config["max_epochs"]
        early_stop_patience = config["early_stop_patience"]
        privacy_enabled = config["privacy_enabled"]
        privacy_noise = config["privacy_noise"]
        encoding_types = config["encoding_types"]
        val_split = config["val_split"]

        # --- 1. Analyze locally ---
        logger.info("Analyzing data locally...")
        encoder = Encoder(
            embedding_dim=embedding_dim,
            model_size=model_size,
            privacy_enabled=privacy_enabled,
            privacy_noise=privacy_noise,
        )
        metadata = encoder.analyze(
            data,
            encoding_types=encoding_types,
            parent=parent,
            parent_encoding_types=parent_encoding_types,
            foreign_key=foreign_key,
            parent_key=parent_key,
        )

        if encoder.is_sequential and parent is None:
            raise InvalidRequestError(
                "Sequential mode requires parent. Groups in "
                "foreign_key have multiple rows, which triggers "
                "entity-level training.",
                param="parent",
            )

        encoder.prepare(data, parent=parent, parent_key=parent_key)

        # --- 2. Register model on API ---
        effective_embedding_dim = embedding_dim
        if encoder.is_sequential and encoder._backend.encoder.ctx_compressor is not None:
            effective_embedding_dim = encoder._backend.encoder.ctx_compressor.dim_output

        logger.info("Creating model on API...")
        resp = http.post("/v1/models", json={
            "metadata": metadata,
            "config": {
                "embedding_dim": effective_embedding_dim,
                "model_size": model_size,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
            },
        })
        model_id = resp["data"]["id"]
        logger.info("Model created: %s", model_id)

        model = cls(
            model_id=model_id,
            client=http,
            encoder=encoder,
            data=data,
            config=config,
            parent=parent,
        )
        model._n_samples = n_samples

        # --- 3. Train ---
        train_frozen(
            model,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            val_split=val_split,
        )

        return model

    def generate(
        self,
        n_samples: int | None = None,
        seed_data: pd.DataFrame | None = None,
        parent: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data from the trained model.

        Flat mode: generates n_samples independent rows.
        Sequential mode: generates variable-length sequences per entity,
        then flattens to a single DataFrame with entity key column.

        Args:
            n_samples: Flat: number of rows. Sequential: number of entities.
                       Defaults to len(training_data) or n_entities from training.
            seed_data: DataFrame for conditional generation (flat only).
            parent: Parent table for generation. Falls back to training parent.

        Returns:
            DataFrame with synthetic data
        """
        ctx = parent if parent is not None else self._parent

        if self._encoder.is_sequential:
            return self._generate_sequential(n_samples=n_samples, parent=ctx)

        n = n_samples or getattr(self, "_n_samples", None) or len(self._data)
        logger.info("Generating %d synthetic rows...", n)
        embedding = self._encoder._generation_embedding(
            n_samples=n, seed_data=seed_data, parent=ctx,
        )

        column_stats = self._encoder.column_stats
        fixed_probs = compute_fixed_probs(column_stats) if column_stats else None

        payload: dict[str, Any] = {"embedding": embedding}
        if fixed_probs:
            payload["fixed_probs"] = fixed_probs

        resp = self._client.post(f"/v1/models/{self.id}/generate", json=payload)

        raw_codes = resp["data"]["codes"]
        features = self._encoder.features
        df = decode_columns(raw_codes, features, column_stats)

        fk_col = self._encoder._foreign_key
        if fk_col and ctx is not None and self._encoder._parent_key:
            pk_col = self._encoder._parent_key
            if pk_col in ctx.columns:
                key_values = ctx[pk_col].values
                df.insert(0, fk_col, [
                    key_values[i] if i < len(key_values) else i
                    for i in range(len(df))
                ])

        self.status = "ready"
        logger.info("Generated %d rows, %d columns", len(df), len(df.columns))
        return df

    def _generate_sequential(
        self,
        n_samples: int | None = None,
        parent: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Sequential generation: context embedding → autoregressive decode → flatten."""
        from dataxid.encoder._nn import RIDX_PREFIX, SIDX_PREFIX, SLEN_PREFIX

        if parent is not None and self._encoder.has_context:
            n_entities = len(parent)
        else:
            n_entities = n_samples or getattr(self, "_n_samples", None) or 100

        logger.info("Generating sequences for %d entities (max_len=%d)...",
                     n_entities, self._encoder.seq_len_max)

        embedding = self._encoder._generation_embedding(
            n_samples=n_entities, parent=parent,
        )

        column_stats = self._encoder.column_stats
        fixed_probs = compute_fixed_probs(column_stats) if column_stats else None

        payload: dict[str, Any] = {
            "embedding": embedding,
            "is_sequential": True,
            "seq_len_max": self._encoder.seq_len_max,
        }
        if fixed_probs:
            payload["fixed_probs"] = fixed_probs

        resp = self._client.post(f"/v1/models/{self.id}/generate", json=payload)
        raw_codes = resp["data"]["codes"]

        pos_prefixes = (SIDX_PREFIX, SLEN_PREFIX, RIDX_PREFIX)
        ridx_cols = [k for k in raw_codes if k.startswith(RIDX_PREFIX)]

        if ridx_cols:
            ridx_lists = raw_codes[ridx_cols[0]]
            seq_lens = []
            for entity_ridx in ridx_lists:
                if entity_ridx[0] == 0:
                    seq_lens.append(0)
                    continue
                length = len(entity_ridx)
                for t in range(1, len(entity_ridx)):
                    if entity_ridx[t] == 0:
                        length = t
                        break
                seq_lens.append(length)
        else:
            first_key = next(iter(raw_codes))
            seq_lens = [len(raw_codes[first_key][i]) for i in range(n_entities)]

        data_cols = {k: v for k, v in raw_codes.items()
                     if not k.startswith(pos_prefixes)}

        rows: list[dict[str, int]] = []
        entity_indices: list[int] = []
        for i in range(n_entities):
            for t in range(seq_lens[i]):
                row = {sub_col: vals[i][t] for sub_col, vals in data_cols.items()}
                rows.append(row)
                entity_indices.append(i)

        if not rows:
            return pd.DataFrame(columns=self._encoder.features)

        df_flat = pd.DataFrame(rows)
        features = self._encoder.features
        df = decode_columns(df_flat.to_dict(orient="list"), features, column_stats)

        fk_col = self._encoder._foreign_key
        if fk_col:
            if (parent is not None
                    and self._encoder._parent_key
                    and self._encoder._parent_key in parent.columns):
                key_values = parent[self._encoder._parent_key].values
                df[fk_col] = [
                    key_values[idx] if idx < len(key_values) else idx
                    for idx in entity_indices
                ]
            else:
                df[fk_col] = entity_indices
            cols = [fk_col] + [c for c in df.columns if c != fk_col]
            df = df[cols]

        self.status = "ready"
        logger.info("Generated %d event rows for %d entities", len(df), n_entities)
        return df

    def delete(self) -> None:
        """Delete the model and free server resources."""
        self._client.delete(f"/v1/models/{self.id}")
        self.status = "deleted"
        logger.info("Model %s deleted", self.id)

    def refresh(self) -> dict[str, Any]:
        """Fetch latest model status from API."""
        resp = self._client.get(f"/v1/models/{self.id}")
        self.status = resp["data"]["status"]
        return resp["data"]

