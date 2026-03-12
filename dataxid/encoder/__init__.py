# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Client-side encoder — neural network, port protocol, and backend adapters.

Transforms tabular data into fixed-size embeddings locally.
Raw data never leaves this package.
"""

from dataxid.encoder._wrapper import Encoder

__all__ = ["Encoder"]
