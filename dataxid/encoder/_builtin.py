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

from dataxid.encoder._nn import Encoder
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
        self.encoder: nn.Module | None = None

    def analyze(
        self,
        df: pd.DataFrame,
        features: list[str],
        embedding_dim: int,
        model_size: str,
        device: torch.device,
        encoding_types: dict[str, str] | None = None,
    ) -> None:
        column_stats = compute_stats(
            df, features, value_protection=True,
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

        self.encoder = Encoder(
            cardinalities=cardinalities,
            model_size=ModelCapacity(model_size),
            embedding_dim=embedding_dim,
            device=device,
        )

        n_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            "Encoder initialized: %d sub-cols → %dd embedding (%s params)",
            len(cardinalities),
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

    def _encode_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        self._check_fitted()
        return self.encoder(batch)  # type: ignore[misc]

    def encode(
        self,
        df: pd.DataFrame,
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
        return self.encoder(x)  # type: ignore[misc]

    def parameters(self):
        self._check_fitted()
        return self.encoder.parameters()  # type: ignore[union-attr]

    def _vocab_sizes(self) -> dict[str, int]:
        self._check_fitted()
        return dict(self.schema.cardinalities)  # type: ignore[union-attr]

    def _column_stats(self) -> dict[str, dict]:
        self._check_fitted()
        return dict(self.schema.column_stats)  # type: ignore[union-attr]

    def _value_mappings(self) -> dict[str, dict]:
        self._check_fitted()
        return dict(self.schema.value_mappings)  # type: ignore[union-attr]

    def _check_fitted(self) -> None:
        if self.schema is None or self.encoder is None:
            raise ModelNotReadyError("Call analyze() first.")
