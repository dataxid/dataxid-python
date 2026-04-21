# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Builtin encoder — transforms tabular data into fixed-size embeddings locally.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from dataxid.encoder._nn import (
    _DIGIT_ENCODING_THRESHOLD,
    RIDX_PREFIX,
    SIDX_PREFIX,
    SLEN_PREFIX,
    Encoder,
)
from dataxid.encoder._ports import ColumnSchema, ModelCapacity
from dataxid.exceptions import ModelNotReadyError
from dataxid.pipeline._analyze import compute_stats, unpack_stats
from dataxid.pipeline._encode import encode_columns

logger = logging.getLogger(__name__)

_SMOOTHING = 1.0


class BuiltinEncoder:
    """Privacy-preserving encoder that transforms tabular data into embeddings on your machine."""

    def __init__(self) -> None:
        self.schema: ColumnSchema | None = None
        self.ctx_schema: ColumnSchema | None = None
        self.encoder: nn.Module | None = None

    def analyze(
        self,
        df: pd.DataFrame,
        features: list[str],
        embedding_dim: int,
        model_size: str,
        device: torch.device,
        encoding_types: dict[str, str] | None = None,
        parent: pd.DataFrame | None = None,
        parent_encoding_types: dict[str, str] | None = None,
        protect_rare: bool = True,
    ) -> None:
        column_stats = compute_stats(
            df, features, protect_rare=protect_rare,
            encoding_types=encoding_types,
        )
        cardinalities, encoding_map, value_mappings = unpack_stats(
            column_stats
        )

        self.schema = ColumnSchema(
            features=features,
            cardinalities=cardinalities,
            encoding_map=encoding_map,
            value_mappings=value_mappings,
            column_stats=column_stats,
        )

        ctx_cardinalities: dict[str, int] | None = None
        if parent is not None:
            ctx_features = list(parent.columns)
            ctx_stats = compute_stats(
                parent, ctx_features, protect_rare=protect_rare,
                encoding_types=parent_encoding_types,
            )
            ctx_cards, ctx_enc_map, ctx_val_mappings = unpack_stats(ctx_stats)
            ctx_cardinalities = ctx_cards
            self.ctx_schema = ColumnSchema(
                features=ctx_features,
                cardinalities=ctx_cards,
                encoding_map=ctx_enc_map,
                value_mappings=ctx_val_mappings,
                column_stats=ctx_stats,
            )

        self.encoder = Encoder(
            cardinalities=cardinalities,
            model_size=ModelCapacity(model_size),
            embedding_dim=embedding_dim,
            device=device,
            ctx_cardinalities=ctx_cardinalities,
        )

        n_params = sum(p.numel() for p in self.encoder.parameters())
        ctx_info = f", ctx={len(ctx_cardinalities)} sub-cols" if ctx_cardinalities else ""
        logger.info(
            "Encoder initialized: %d sub-cols%s → %dd embedding (%s params)",
            len(cardinalities),
            ctx_info,
            embedding_dim,
            f"{n_params:,}",
        )

    def _compute_priors(
        self,
        df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        self._check_fitted()
        encoded = encode_columns(
            df, self.schema.features, self.schema.column_stats
        )
        probs_map: dict[str, np.ndarray] = {}
        for sub_col, cardinality in self.schema.cardinalities.items():
            values = encoded[sub_col]
            counts = np.zeros(cardinality)
            for idx in values:
                if 0 <= idx < cardinality:
                    counts[int(idx)] += 1
            alpha = _SMOOTHING
            total = counts.sum() + alpha * cardinality
            probs = (counts + alpha) / max(total, 1e-12)
            probs_map[sub_col] = np.clip(probs, a_min=1e-12, a_max=None)
        return probs_map

    def _compute_ctx_priors(
        self,
        ctx_df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Compute empirical priors for context columns."""
        if self.ctx_schema is None:
            return {}
        encoded = encode_columns(
            ctx_df, self.ctx_schema.features, self.ctx_schema.column_stats
        )
        probs_map: dict[str, np.ndarray] = {}
        for sub_col, cardinality in self.ctx_schema.cardinalities.items():
            values = encoded[sub_col]
            counts = np.zeros(cardinality)
            for idx in values:
                if 0 <= idx < cardinality:
                    counts[int(idx)] += 1
            alpha = _SMOOTHING
            total = counts.sum() + alpha * cardinality
            probs = (counts + alpha) / max(total, 1e-12)
            probs_map[sub_col] = np.clip(probs, a_min=1e-12, a_max=None)
        return probs_map

    def _prepare_tensors(
        self,
        df: pd.DataFrame,
    ) -> dict[str, torch.Tensor]:
        self._check_fitted()
        encoded = encode_columns(
            df, self.schema.features, self.schema.column_stats
        )
        device = self.encoder.device  # type: ignore[union-attr]
        return {
            k: torch.tensor(v, dtype=torch.long).unsqueeze(-1).to(device)
            for k, v in encoded.items()
        }

    def _prepare_ctx_tensors(
        self,
        ctx_df: pd.DataFrame,
    ) -> dict[str, torch.Tensor]:
        """Encode context table to tensor dict."""
        if self.ctx_schema is None:
            raise ModelNotReadyError("No context schema — analyze with parent first.")
        encoded = encode_columns(
            ctx_df, self.ctx_schema.features, self.ctx_schema.column_stats
        )
        device = self.encoder.device  # type: ignore[union-attr]
        return {
            k: torch.tensor(v, dtype=torch.long).unsqueeze(-1).to(device)
            for k, v in encoded.items()
        }

    def _prepare_sequential_tensors(
        self,
        df: pd.DataFrame,
        group_by: str,
        seq_len_max: int,
        parent: pd.DataFrame | None = None,
        parent_key: str | None = None,
    ) -> dict[str, list]:
        """Encode sequential data → dict of list-columns (entity × timesteps).

        Parent entities with zero child rows are included as 0-length sequences
        (single padding row with SLEN=0) so the model learns the "no children"
        pattern.

        Returns nested lists (one list per entity, containing its timesteps)
        rather than dense tensors, so downstream batching can handle variable
        sequence lengths without padding every row to the maximum.
        """
        self._check_fitted()
        key = group_by

        def _pad_group(g: pd.DataFrame) -> pd.DataFrame:
            pad_row = {c: [0] for c in g.columns if c != key}
            pad_row[key] = [g.iloc[0][key]]
            return pd.concat([g, pd.DataFrame(pad_row)], ignore_index=True)

        df_padded = (
            df.groupby(key, sort=False)[df.columns.tolist()]
            .apply(_pad_group)
            .reset_index(drop=True)
        )

        if parent is not None and parent_key:
            child_keys = set(df[key].unique())
            all_parent_keys = set(parent[parent_key].unique())
            missing_keys = all_parent_keys - child_keys
            if missing_keys:
                zero_row = {c: 0 for c in df_padded.columns if c != key}
                zero_rows = [{**zero_row, key: k} for k in missing_keys]
                df_padded = pd.concat([df_padded, pd.DataFrame(zero_rows)], ignore_index=True)

        sidx = df_padded.groupby(key).cumcount(ascending=True)
        slen = df_padded.groupby(key)[key].transform("size") - 1
        ridx = df_padded.groupby(key).cumcount(ascending=False)

        sidx_df = _encode_positional(sidx, seq_len_max, SIDX_PREFIX)
        slen_df = _encode_positional(slen, seq_len_max, SLEN_PREFIX)
        ridx_df = _encode_positional(ridx, seq_len_max, RIDX_PREFIX)

        features = self.schema.features  # type: ignore[union-attr]
        encoded = encode_columns(df_padded[features], features, self.schema.column_stats)

        pos_dfs = pd.concat([sidx_df, slen_df, ridx_df], axis=1)
        all_encoded = pd.concat(
            [df_padded[[key]], pos_dfs, pd.DataFrame(encoded)], axis=1,
        )

        data_cols = [c for c in all_encoded.columns if c != key]
        grouped = all_encoded.groupby(key, sort=False)

        result: dict[str, list] = {}
        for col in data_cols:
            result[col] = grouped[col].apply(list).tolist()

        if parent is not None and self.ctx_schema is not None and parent_key:
            entity_keys = list(grouped.groups.keys())
            ctx_aligned = parent.set_index(parent_key).loc[entity_keys].reset_index(drop=True)
            ctx_encoded = encode_columns(
                ctx_aligned, self.ctx_schema.features, self.ctx_schema.column_stats,
            )
            for k_ctx, v_ctx in ctx_encoded.items():
                result[k_ctx] = v_ctx.tolist() if hasattr(v_ctx, "tolist") else list(v_ctx)

        return result

    def _encode_batch(
        self,
        batch: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._check_fitted()
        return self.encoder(batch, ctx=ctx)  # type: ignore[misc]

    def encode(
        self,
        df: pd.DataFrame,
        ctx_df: pd.DataFrame | None = None,
    ) -> torch.Tensor:
        self._check_fitted()
        encoded = encode_columns(
            df, self.schema.features, self.schema.column_stats
        )
        device = self.encoder.device  # type: ignore[union-attr]
        x = {
            k: torch.tensor(v, dtype=torch.long).unsqueeze(-1).to(device)
            for k, v in encoded.items()
        }
        ctx_tensors = None
        if ctx_df is not None and self.ctx_schema is not None:
            ctx_encoded = encode_columns(
                ctx_df, self.ctx_schema.features, self.ctx_schema.column_stats
            )
            ctx_tensors = {
                k: torch.tensor(v, dtype=torch.long).unsqueeze(-1).to(device)
                for k, v in ctx_encoded.items()
            }
        return self.encoder(x, ctx=ctx_tensors)  # type: ignore[misc]

    def encode_context_only(self, ctx_df: pd.DataFrame) -> torch.Tensor:
        """Encode context table through FeatureCompressor only (no target encoder)."""
        self._check_fitted()
        ctx_tensors = self._prepare_ctx_tensors(ctx_df)
        ctx_embeds = self.encoder.ctx_compressor(ctx_tensors)  # type: ignore[union-attr]
        if ctx_embeds:
            return ctx_embeds[0]
        device = self.encoder.device  # type: ignore[union-attr]
        return torch.zeros(len(ctx_df), 0, device=device)

    def parameters(self):
        self._check_fitted()
        return self.encoder.parameters()  # type: ignore[union-attr]

    def _vocab_sizes(self) -> dict[str, int]:
        self._check_fitted()
        combined = dict(self.schema.cardinalities)  # type: ignore[union-attr]
        if self.ctx_schema is not None:
            combined.update(self.ctx_schema.cardinalities)
        return combined

    def _column_stats(self) -> dict[str, dict]:
        self._check_fitted()
        combined = dict(self.schema.column_stats)  # type: ignore[union-attr]
        if self.ctx_schema is not None:
            combined.update(self.ctx_schema.column_stats)
        return combined

    def _value_mappings(self) -> dict[str, dict]:
        self._check_fitted()
        combined = dict(self.schema.value_mappings)  # type: ignore[union-attr]
        if self.ctx_schema is not None:
            combined.update(self.ctx_schema.value_mappings)
        return combined

    def _check_fitted(self) -> None:
        if self.schema is None or self.encoder is None:
            raise ModelNotReadyError("Call analyze() first.")


def _encode_positional(vals: pd.Series, max_seq_len: int, prefix: str) -> pd.DataFrame:
    """Encode positional values (SIDX/SLEN/RIDX) to sub-column DataFrame."""
    if max_seq_len < _DIGIT_ENCODING_THRESHOLD:
        return pd.DataFrame({f"{prefix}cat": vals.values})
    n_digits = len(str(max_seq_len))
    rows = vals.astype(str).str.pad(width=n_digits, fillchar="0").apply(list).tolist()
    df = pd.DataFrame(rows).astype(int)
    df.columns = [f"{prefix}E{i}" for i in range(n_digits - 1, -1, -1)]
    return df
