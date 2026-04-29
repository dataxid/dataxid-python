# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the client-side decode pipeline (``dataxid.pipeline._decode``).

The decode stage maps integer codes produced by the server back into real
feature values using per-column statistics held only on the client side.

Coverage focuses on two contracts that are easy to break and costly to ship:

1. ``rare_strategy`` controls whether the ``<protected>`` sentinel is replaced
   by a frequent draw (``"sample"``, default) or kept literally (``"mask"``).
2. Numeric decode paths must never let the sentinel leak into a numeric dtype,
   regardless of the strategy — otherwise the sentinel would be coerced to NA
   and silently mask a type-pollution bug.

Coverage of other encoding types (continuous, datetime, lat/long, character)
lives in the pipeline-level test suite; this module is intentionally scoped to
the client-visible ``rare_strategy`` knob and the ``decode_columns`` dispatcher.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from dataxid.encoder._ports import EncodingType, wire_key
from dataxid.pipeline._decode import (
    _decode_categorical,
    _decode_numeric_discrete,
    decode_columns,
)
from dataxid.pipeline._transform import NULL_TOKEN, RARE_TOKEN


@pytest.fixture(autouse=True)
def _isolate_numpy_rng() -> Iterator[None]:
    """Pin ``numpy.random`` to a fixed seed per test and restore on exit.

    The decode pipeline still uses the legacy global RNG, so two tests run
    back-to-back can otherwise observe each other's draws. Snapshotting the
    state and resetting it makes assertions on stochastic output reproducible
    without leaking determinism across the rest of the suite.
    """
    state = np.random.get_state()
    np.random.seed(0)
    try:
        yield
    finally:
        np.random.set_state(state)


@pytest.fixture
def categorical_stats() -> dict:
    """Vocabulary with both sentinels and three frequent categories.

    Layout: ``0=<<NULL>>``, ``1=<protected>``, ``2=A``, ``3=B``, ``4=C``.
    """
    return {
        "codes": {NULL_TOKEN: 0, RARE_TOKEN: 1, "A": 2, "B": 3, "C": 4},
        "encoding_type": EncodingType.categorical,
        "no_of_rare_categories": 3,
    }


@pytest.fixture
def numeric_discrete_stats() -> dict:
    """Numeric discrete column without ``NULL_TOKEN`` in its vocabulary.

    Layout: ``0=<protected>``, ``1=1``, ``2=2``, ``3=3``. Because no NULL
    sentinel is registered, rare codes are resolved by sampling a frequent
    value rather than collapsing to NA — the path most production columns hit.
    """
    return {
        "codes": {RARE_TOKEN: 0, "1": 1, "2": 2, "3": 3},
        "encoding_type": EncodingType.numeric_discrete,
        "min_decimal": 0,
    }


class TestDecodeCategoricalSample:
    """Default mode — rare codes are replaced by frequent draws."""

    def test_no_rare_passthrough(self, categorical_stats: dict) -> None:
        codes = np.array([2, 3, 4, 2])
        result = _decode_categorical(codes, categorical_stats)
        assert list(result) == ["A", "B", "C", "A"]

    def test_rare_replaced_not_literal(self, categorical_stats: dict) -> None:
        codes = np.array([1, 1, 1, 1, 1])
        result = _decode_categorical(codes, categorical_stats)
        assert RARE_TOKEN not in set(result.dropna())
        assert set(result.dropna()).issubset({"A", "B", "C"})

    def test_null_still_becomes_na(self, categorical_stats: dict) -> None:
        codes = np.array([0, 2])
        result = _decode_categorical(codes, categorical_stats)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == "A"


class TestDecodeCategoricalMask:
    """Mask mode — rare codes surface as the literal ``<protected>`` token."""

    def test_mask_preserves_rare_null_and_frequent(
        self, categorical_stats: dict
    ) -> None:
        codes = np.array([0, 1, 2, 1, 3])
        result = _decode_categorical(
            codes, categorical_stats, rare_strategy="mask"
        )
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == RARE_TOKEN
        assert result.iloc[2] == "A"
        assert result.iloc[3] == RARE_TOKEN
        assert result.iloc[4] == "B"


class TestDecodeCategoricalEdgeCases:
    """Boundary conditions where the rare-replacement path has no fallback."""

    def test_sample_falls_back_to_na_when_no_frequent_categories(
        self,
    ) -> None:
        """If the vocabulary contains only sentinels, sampled rare codes
        cannot be replaced by a real value and must collapse to NA rather
        than echo the literal ``<protected>`` token."""
        stats = {
            "codes": {NULL_TOKEN: 0, RARE_TOKEN: 1},
            "encoding_type": EncodingType.categorical,
            "no_of_rare_categories": 1,
        }
        codes = np.array([1, 1, 0])
        result = _decode_categorical(codes, stats)
        assert result.isna().all()


class TestDecodeNumericDiscrete:
    """Numeric columns must never leak ``<protected>`` into numeric output."""

    def test_rare_never_appears_in_output(
        self, numeric_discrete_stats: dict
    ) -> None:
        codes = np.array([0, 1, 0, 2])
        result = _decode_numeric_discrete(codes, numeric_discrete_stats)
        assert result.dtype.name == "Int64"
        for v in result.dropna():
            assert v in {1, 2, 3}

    def test_frequent_passthrough(
        self, numeric_discrete_stats: dict
    ) -> None:
        codes = np.array([1, 2, 3, 1])
        result = _decode_numeric_discrete(codes, numeric_discrete_stats)
        assert list(result) == [1, 2, 3, 1]

    def test_rare_collapses_to_na_when_null_token_in_vocabulary(
        self,
    ) -> None:
        """When ``NULL_TOKEN`` is part of the vocabulary, the discrete
        decoder treats rare codes as missing instead of resampling — the
        opposite branch of the resampling test above."""
        stats = {
            "codes": {NULL_TOKEN: 0, RARE_TOKEN: 1, "1": 2, "2": 3},
            "encoding_type": EncodingType.numeric_discrete,
            "min_decimal": 0,
        }
        codes = np.array([0, 1, 2, 1, 3])
        result = _decode_numeric_discrete(codes, stats)
        assert result.dtype.name == "Int64"
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 2


class TestDecodeColumnsModeForwarding:
    def _make_raw(self, feature: str, codes: list[int]) -> dict:
        return {wire_key(feature, "cat"): codes}

    def test_returns_only_requested_columns_in_order(
        self, categorical_stats: dict, numeric_discrete_stats: dict
    ) -> None:
        """Dispatcher contract: output frame holds exactly the requested
        feature names, in the requested order, regardless of the input
        ``raw_codes`` ordering."""
        raw = {
            wire_key("num", "cat"): [1, 2, 3],
            wire_key("cat", "cat"): [2, 3, 4],
        }
        stats = {"cat": categorical_stats, "num": numeric_discrete_stats}
        df = decode_columns(raw, ["cat", "num"], stats)
        assert list(df.columns) == ["cat", "num"]
        assert len(df) == 3

    def test_mask_mode_keeps_rare(self, categorical_stats: dict) -> None:
        raw = self._make_raw("cat", [1, 2, 1])
        df = decode_columns(
            raw,
            ["cat"],
            {"cat": categorical_stats},
            rare_strategy="mask",
        )
        assert list(df["cat"]) == [RARE_TOKEN, "A", RARE_TOKEN]

    def test_mask_mode_does_not_leak_into_numeric(
        self, numeric_discrete_stats: dict
    ) -> None:
        """Mask mode must not bypass the numeric dtype guard."""
        raw = {wire_key("num", "cat"): [0, 1, 2, 3]}
        df = decode_columns(
            raw,
            ["num"],
            {"num": numeric_discrete_stats},
            rare_strategy="mask",
        )
        assert df["num"].dtype.name == "Int64"
        for v in df["num"].dropna():
            assert v in {1, 2, 3}
