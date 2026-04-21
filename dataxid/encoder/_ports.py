# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Encoder port — defines the contract between the Encoder wrapper and its backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Shared types — used by both adapters and the Encoder wrapper
# ---------------------------------------------------------------------------

# Wire protocol — sub-column key format: {WIRE_PREFIX}:/{feature}__{sub_col}
WIRE_PREFIX = "feat"
WIRE_TABLE_SEP = ":"
WIRE_COLUMN_SEP = "/"
WIRE_SUB_COLUMN_SEP = "__"


def wire_key(feature: str, sub_col: str) -> str:
    """Build a wire protocol sub-column key: ``feat:/{feature}__{sub_col}``."""
    return f"{WIRE_PREFIX}{WIRE_TABLE_SEP}{WIRE_COLUMN_SEP}{feature}{WIRE_SUB_COLUMN_SEP}{sub_col}"

class ModelCapacity(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @property
    def embedding_scale(self) -> tuple[int, float]:
        return {"small": (2, 0.15), "medium": (3, 0.25), "large": (4, 0.33)}[self.value]

    @property
    def column_base(self) -> int:
        return {"small": 4, "medium": 10, "large": 16}[self.value]

    @property
    def context_units(self) -> list[int]:
        return {"small": [2], "medium": [8], "large": [32]}[self.value]


class EncodingType(str, Enum):
    auto = "AUTO"
    categorical = "TABULAR_CATEGORICAL"
    numeric_auto = "TABULAR_NUMERIC_AUTO"
    numeric_discrete = "TABULAR_NUMERIC_DISCRETE"
    numeric_binned = "TABULAR_NUMERIC_BINNED"
    numeric_digit = "TABULAR_NUMERIC_DIGIT"
    character = "TABULAR_CHARACTER"
    datetime = "TABULAR_DATETIME"
    datetime_relative = "TABULAR_DATETIME_RELATIVE"
    lat_long = "TABULAR_LAT_LONG"


@dataclass
class ColumnSchema:
    """Encoding configuration produced by analyze(), consumed by encode/prepare."""
    features: list[str]
    cardinalities: dict[str, int] = field(default_factory=dict)
    encoding_map: dict[str, str] = field(default_factory=dict)
    value_mappings: dict[str, dict] = field(default_factory=dict)
    column_stats: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EncoderPort — the contract between Encoder and its backend
# ---------------------------------------------------------------------------

@runtime_checkable
class EncoderPort(Protocol):
    """What the Encoder wrapper needs from its backend."""

    schema: ColumnSchema | None
    encoder: nn.Module | None

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
        """Analyze raw data → populate schema and build encoder network.

        When ``protect_rare`` is True, rare categorical values are replaced
        with the ``<protected>`` sentinel before the vocabulary is built.
        """
        ...

    def _compute_priors(
        self,
        df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Compute empirical probability distribution per sub-column."""
        ...

    def _prepare_tensors(
        self,
        df: pd.DataFrame,
    ) -> dict[str, torch.Tensor]:
        """Encode raw data to tensor dict (call once, reuse across epochs)."""
        ...

    def _encode_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass: tensor dict → embedding (batch_size, embedding_dim)."""
        ...

    def encode(
        self,
        df: pd.DataFrame,
    ) -> torch.Tensor:
        """Encode raw data directly to embedding (used for generation)."""
        ...

    def parameters(self):
        """Encoder network parameters (for optimizer setup)."""
        ...

    def _vocab_sizes(self) -> dict[str, int]:
        """Sub-column cardinalities."""
        ...

    def _column_stats(self) -> dict[str, dict]:
        """Per-column statistics."""
        ...

    def _value_mappings(self) -> dict[str, dict]:
        """Value mappings."""
        ...

