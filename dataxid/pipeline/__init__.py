# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Data pipeline — analysis, encoding, type transforms, and privacy helpers.

Converts raw tabular data into integer-coded sub-columns and produces
the wire metadata (cardinalities, column_stats, value_mappings) consumed
by the encoder and the Dataxid API.
"""

from dataxid.pipeline._analyze import compute_stats, detect_encoding, unpack_stats
from dataxid.pipeline._decode import compute_fixed_probs, decode_columns
from dataxid.pipeline._encode import encode_columns

__all__ = [
    "compute_stats",
    "unpack_stats",
    "detect_encoding",
    "encode_columns",
    "decode_columns",
    "compute_fixed_probs",
]
