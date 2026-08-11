# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Training — Model lifecycle, config, and training strategies.
"""

from dataxid.training._config import ModelConfig as _ModelConfig
from dataxid.training._model import Model

class ModelConfig(_ModelConfig):
    def __init__(self, *args, encoder_mode="tabular", **kwargs):
        if encoder_mode not in ("tabular", "sequential"):
            raise ValueError(f"encoder_mode must be 'tabular' or 'sequential', got {encoder_mode}")
        self.encoder_mode = encoder_mode
        super().__init__(*args, **kwargs)

__all__ = ["Model", "ModelConfig"]
