# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Encoder neural network — transforms tabular features into a fixed-size embedding.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch import nn

from dataxid.encoder._ports import WIRE_COLUMN_SEP, WIRE_SUB_COLUMN_SEP, ModelCapacity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wire protocol constants — sub-column naming conventions
# ---------------------------------------------------------------------------

POSITIONAL_COLUMN = "__pos__"
SIDX_PREFIX = f"{POSITIONAL_COLUMN}{WIRE_SUB_COLUMN_SEP}sidx_"
RIDX_PREFIX = f"{POSITIONAL_COLUMN}{WIRE_SUB_COLUMN_SEP}ridx_"
SLEN_PREFIX = f"{POSITIONAL_COLUMN}{WIRE_SUB_COLUMN_SEP}slen_"

_DIGIT_ENCODING_THRESHOLD = 100

ModelCapacityOrUnits = ModelCapacity | dict[str, list[int]]


def get_positional_cardinalities(seq_len_max: int) -> dict[str, int]:
    """Positional sub-column cardinalities for sequential mode (SIDX, SLEN, RIDX)."""
    if seq_len_max < _DIGIT_ENCODING_THRESHOLD:
        return {
            f"{SIDX_PREFIX}cat": seq_len_max + 1,
            f"{SLEN_PREFIX}cat": seq_len_max + 1,
            f"{RIDX_PREFIX}cat": seq_len_max + 1,
        }
    digits = [int(d) for d in str(seq_len_max)]
    cards: dict[str, int] = {}
    for i, digit in enumerate(digits):
        e_idx = len(digits) - 1 - i
        card = (digit + 1) if i == 0 else 10
        cards[f"{SIDX_PREFIX}E{e_idx}"] = card
        cards[f"{RIDX_PREFIX}E{e_idx}"] = card
        cards[f"{SLEN_PREFIX}E{e_idx}"] = card
    return cards

_DROPOUT_RATE = 0.25


# ---------------------------------------------------------------------------
# Grouping utility
# ---------------------------------------------------------------------------

def group_sub_columns(
    cardinalities: dict[str, int],
    by: Literal["columns", "tables"] = "columns",
) -> dict[str, list[str]]:
    """Group sub-column keys by their parent column or table prefix."""
    sep = WIRE_COLUMN_SEP if by == "tables" else WIRE_SUB_COLUMN_SEP
    groups: dict[str, list[str]] = {}
    for key in cardinalities:
        parent = key.split(sep)[0]
        groups.setdefault(parent, []).append(key)
    return groups


# ---------------------------------------------------------------------------
# Dimension heuristics
# ---------------------------------------------------------------------------

def _sub_column_dim(
    key: str, model_size: ModelCapacityOrUnits, input_dim: int,
) -> int:
    """Embedding dimension for a single sub-column."""
    if isinstance(model_size, dict):
        return model_size[key]
    factor, exp = model_size.embedding_scale
    output = max(10, int(factor * np.ceil(input_dim ** exp)))
    return min(input_dim, output)


def _column_dim(
    key: str,
    model_size: ModelCapacityOrUnits,
    input_dim: int,
    n_sub_cols: int,
    compress: bool,
) -> int:
    """Compressed dimension for a column (group of sub-columns)."""
    if isinstance(model_size, dict):
        return model_size.get(key, input_dim)
    output = int(model_size.column_base + n_sub_cols)
    should_compress = compress and n_sub_cols > 2
    dim = output if should_compress else input_dim
    return min(input_dim, dim)


def _context_layers(
    key: str, model_size: ModelCapacityOrUnits, input_dim: int,
) -> list[int]:
    """Hidden layer sizes for the flat context compressor."""
    if isinstance(model_size, dict):
        return model_size[key]
    coeff = round(np.log(max(input_dim, np.e)))
    return [u * coeff for u in model_size.context_units]


# ---------------------------------------------------------------------------
# Stage 1 — Sub-column embeddings
# ---------------------------------------------------------------------------

class SubColumnEmbedding(nn.Module):
    """Learnable embedding per sub-column (one nn.Embedding each)."""

    def __init__(
        self,
        model_size: ModelCapacityOrUnits,
        cardinalities: dict[str, int],
        device: torch.device,
    ):
        super().__init__()
        self.cardinalities = cardinalities
        self.device = device
        self.dims: list[int] = []
        self.layers = nn.ModuleDict()

        has_ridx = any(k.startswith(RIDX_PREFIX) for k in cardinalities)
        last_slen = next(
            (k for k in reversed(cardinalities)
             if k.startswith(SLEN_PREFIX) and has_ridx),
            None,
        )

        for sub_col, card in cardinalities.items():
            tag = self._tag(sub_col)
            dim = _sub_column_dim(key=tag, model_size=model_size, input_dim=card)
            emb = nn.Embedding(num_embeddings=card, embedding_dim=dim, device=device)
            if sub_col == last_slen:
                emb.weight.requires_grad = False
            self.layers[tag] = emb
            self.dims.append(dim)

    def __bool__(self) -> bool:
        return bool(self.layers)

    @staticmethod
    def _tag(sub_col: str) -> str:
        return f"embedder@{sub_col}"

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for sub_col in self.cardinalities:
            t = torch.as_tensor(x[sub_col], device=self.device)
            if t.is_nested:
                t = torch.nested.to_padded_tensor(t, 0)
            t = self.layers[self._tag(sub_col)](t)
            out[sub_col] = torch.squeeze(t, -2)
        return out


# ---------------------------------------------------------------------------
# Stage 2 — Column-level compression
# ---------------------------------------------------------------------------

class ColumnEmbedding(nn.Module):
    """Compress sub-column embeddings into a single vector per column."""

    def __init__(
        self,
        model_size: ModelCapacityOrUnits,
        cardinalities: dict[str, int],
        sub_col_dims: list[int],
        device: torch.device,
    ):
        super().__init__()
        self.cardinalities = cardinalities
        self.device = device
        self.column_groups = group_sub_columns(cardinalities, by="columns")
        self.dims: list[int] = []
        self.layers = nn.ModuleDict()

        do_compress = len(cardinalities) > 50
        offset = 0
        for col, subs in self.column_groups.items():
            n = len(subs)
            in_dim = sum(sub_col_dims[offset: offset + n])
            out_dim = _column_dim(
                key=self._tag(col),
                model_size=model_size,
                input_dim=in_dim,
                n_sub_cols=n,
                compress=do_compress,
            )
            layer = (
                nn.Linear(in_dim, out_dim, device=device)
                if out_dim < in_dim
                else nn.Identity()
            )
            self.layers[self._tag(col)] = layer
            self.dims.append(out_dim)
            offset += n

    @staticmethod
    def _tag(col: str) -> str:
        return f"column_embedder@{col}"

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for col, subs in self.column_groups.items():
            cat = torch.cat([x[s] for s in subs], dim=-1)
            out[col] = self.layers[self._tag(col)](cat)
        return out


# ---------------------------------------------------------------------------
# Optional — Flat context compressor
# ---------------------------------------------------------------------------

class FeatureCompressor(nn.Module):
    """Compress flat features into a fixed-size vector."""

    def __init__(
        self,
        model_size: ModelCapacityOrUnits,
        cardinalities: dict[str, int],
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(p=_DROPOUT_RATE)

        self.sub_col_embedding = SubColumnEmbedding(
            model_size=model_size, cardinalities=cardinalities, device=device,
        )

        self.compressor_layers = nn.ModuleDict()
        self.dim_output = 0
        if self.sub_col_embedding:
            total_dim = sum(self.sub_col_embedding.dims)
            hidden = _context_layers(
                key="flat_context@", model_size=model_size, input_dim=total_dim,
            )
            layers = nn.ModuleList()
            dims = [total_dim] + hidden
            for d_in, d_out in zip(dims[:-1], dims[1:], strict=False):
                layers.append(nn.Linear(d_in, d_out, device=device))
            self.compressor_layers["flat_context@"] = layers
            self.dim_output = hidden[-1]

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        embeddings = self.sub_col_embedding(x)
        result: list[torch.Tensor] = []
        if embeddings:
            t = torch.cat(list(embeddings.values()), dim=-1)
            for layer in self.compressor_layers["flat_context@"]:
                t = layer(t)
                t = self.dropout(t)
                result = [t]
        return result


# ---------------------------------------------------------------------------
# Encoder — the main module
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encode tabular features into a fixed-size embedding."""

    def __init__(
        self,
        cardinalities: dict[str, int],
        model_size: ModelCapacityOrUnits,
        embedding_dim: int,
        device: torch.device,
        ctx_cardinalities: dict[str, int] | None = None,
    ):
        super().__init__()
        self.cardinalities = cardinalities
        self.ctx_cardinalities = ctx_cardinalities or {}
        self.embedding_dim = embedding_dim
        self.device = device
        self.has_context = len(self.ctx_cardinalities) > 0

        self.sub_col_embedding = SubColumnEmbedding(
            model_size=model_size, cardinalities=cardinalities, device=device,
        )

        self.col_embedding = ColumnEmbedding(
            model_size=model_size,
            cardinalities=cardinalities,
            sub_col_dims=self.sub_col_embedding.dims,
            device=device,
        )

        self.ctx_compressor: FeatureCompressor | None = None
        ctx_dim = 0
        if self.has_context:
            self.ctx_compressor = FeatureCompressor(
                model_size=model_size,
                cardinalities=self.ctx_cardinalities,
                device=device,
            )
            ctx_dim = self.ctx_compressor.dim_output

        total_dim = sum(self.col_embedding.dims) + ctx_dim
        self.compressor = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2, device=device),
            nn.ReLU(),
            nn.Dropout(_DROPOUT_RATE),
            nn.Linear(embedding_dim * 2, embedding_dim, device=device),
            nn.ReLU(),
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
        ctx: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sub_embeds = self.sub_col_embedding(x)
        col_embeds = self.col_embedding(sub_embeds)
        combined = torch.cat(list(col_embeds.values()), dim=-1)

        if self.has_context and self.ctx_compressor is not None:
            if ctx is not None:
                ctx_embeds = self.ctx_compressor(ctx)
                if ctx_embeds:
                    combined = torch.cat([combined, ctx_embeds[0]], dim=-1)
            else:
                ctx_dim = self.ctx_compressor.dim_output
                zeros = torch.zeros(
                    combined.shape[0], ctx_dim, device=combined.device,
                )
                combined = torch.cat([combined, zeros], dim=-1)

        return self.compressor(combined)
