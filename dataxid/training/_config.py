# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
ModelConfig — typed training configuration for Model.create().

Replaces the untyped config dict with IDE-discoverable, validated fields.
Backward compatible: Model.create(config={...}) still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """
    Training configuration for Model.create().

    Example::

        import dataxid

        model = dataxid.Model.create(
            data=df,
            config=dataxid.ModelConfig(
                embedding_dim=128,
                model_size="large",
                max_epochs=50,
            ),
        )
    """

    embedding_dim: int = 64
    model_size: Literal["small", "medium", "large"] = "medium"
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 4
    val_split: float = 0.1
    privacy_enabled: bool = False
    privacy_noise: float = 0.1
    encoding_types: dict[str, str] | None = None
    accumulation_steps: int = 1
    learning_rate: float | None = None
    label_smoothing: float = 0.0
    embedding_dropout: float = 0.5
    time_limit_seconds: float = 0.0
    seed: int | None = None
    timeout: float = 14400.0

    def to_dict(self) -> dict:
        """Convert to plain dict (internal use)."""
        return dict(self.__dict__)


def _resolve_config(config: dict | ModelConfig | None) -> dict:
    """Normalize config input → plain dict with all defaults filled.

    All missing keys are populated from ModelConfig defaults, ensuring
    config["key"] access is always safe after this call.
    """
    if config is None:
        return ModelConfig().to_dict()
    if isinstance(config, ModelConfig):
        return config.to_dict()
    if isinstance(config, dict):
        return ModelConfig(**config).to_dict()
    raise TypeError(
        f"config must be a ModelConfig instance or dict, got {type(config).__name__}"
    )
