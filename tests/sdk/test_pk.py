# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :mod:`dataxid._pk`, the primary-key generators used when
synthesizing tables.

Three generators are covered:

* :func:`generate_dxid` — ``dxid_`` + 22 char base62 string (128 bits entropy)
* :func:`generate_uuid` — RFC 4122 v4 UUID string
* :func:`generate_primary_keys` — dispatch helper for ``dxid`` / ``int`` / ``uuid``

Properties asserted: format invariants (prefix, length, charset), uniqueness
under stress, and seed-determinism for the generators that draw from Python's
global ``random`` state. The shared ``_isolate_python_rng`` fixture in
``conftest.py`` keeps each test's seeding from leaking.
"""

import random
import string
import uuid

import pytest

from dataxid._pk import (
    PK_TYPES,
    PkType,
    generate_dxid,
    generate_primary_keys,
    generate_uuid,
)

BASE62_ALPHABET = set(string.digits + string.ascii_uppercase + string.ascii_lowercase)
DXID_PREFIX = "dxid_"
DXID_BODY_LEN = 22
DXID_TOTAL_LEN = len(DXID_PREFIX) + DXID_BODY_LEN


class TestPkTypeConstant:
    """``PK_TYPES`` is the wire-level enum of supported primary-key formats."""

    def test_contains_expected_values(self) -> None:
        assert set(PK_TYPES) == {"dxid", "int", "uuid"}


class TestGenerateDxid:
    """``generate_dxid`` — dxid format invariants and seed determinism."""

    def test_format_invariants(self) -> None:
        """Single source of truth for the dxid wire format: returns a string,
        starts with ``dxid_``, body is exactly 22 base62 characters."""
        result = generate_dxid()
        assert isinstance(result, str)
        assert result.startswith(DXID_PREFIX)
        assert len(result) == DXID_TOTAL_LEN

        body = result[len(DXID_PREFIX):]
        assert len(body) == DXID_BODY_LEN
        assert set(body).issubset(BASE62_ALPHABET)

    def test_many_are_unique(self) -> None:
        """Stress test: 1000 dxids in a row collide with negligible probability
        (128-bit entropy → birthday bound ~10^19). Any collision indicates a
        regression in the entropy source."""
        ids = {generate_dxid() for _ in range(1000)}
        assert len(ids) == 1000

    def test_respects_random_seed(self) -> None:
        random.seed(42)
        a = [generate_dxid() for _ in range(5)]
        random.seed(42)
        b = [generate_dxid() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self) -> None:
        random.seed(1)
        a = generate_dxid()
        random.seed(2)
        b = generate_dxid()
        assert a != b


class TestGenerateUuidQuirks:
    """``generate_uuid`` deliberately draws from a separate entropy source.

    Pinned here because callers must not rely on ``random.seed`` for UUID
    determinism — that's a soft contract the module docstring promises and
    that this test guards against accidental changes to.
    """

    def test_does_not_respect_random_seed(self) -> None:
        random.seed(99)
        a = generate_uuid()
        random.seed(99)
        b = generate_uuid()
        assert a != b


class TestGenerateUuid:
    """``generate_uuid`` — UUID v4 format invariants."""

    def test_returns_uuid_v4_string(self) -> None:
        result = generate_uuid()
        assert isinstance(result, str)
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_many_are_unique(self) -> None:
        ids = {generate_uuid() for _ in range(1000)}
        assert len(ids) == 1000


class TestGeneratePrimaryKeys:
    """``generate_primary_keys`` — dispatch helper for the three pk types."""

    def test_dxid_returns_list_of_dxids(self) -> None:
        keys = generate_primary_keys("dxid", 5)
        assert len(keys) == 5
        assert all(isinstance(k, str) and k.startswith(DXID_PREFIX) for k in keys)

    def test_int_returns_1_based_sequence(self) -> None:
        assert generate_primary_keys("int", 4) == [1, 2, 3, 4]

    def test_int_zero_returns_empty(self) -> None:
        assert generate_primary_keys("int", 0) == []

    def test_uuid_returns_list_of_uuids(self) -> None:
        keys = generate_primary_keys("uuid", 3)
        assert len(keys) == 3
        for k in keys:
            assert isinstance(k, str)
            uuid.UUID(k)

    def test_dxid_respects_random_seed(self) -> None:
        random.seed(123)
        a = generate_primary_keys("dxid", 10)
        random.seed(123)
        b = generate_primary_keys("dxid", 10)
        assert a == b

    def test_dxid_produces_unique_keys_within_batch(self) -> None:
        keys = generate_primary_keys("dxid", 500)
        assert len(set(keys)) == 500

    @pytest.mark.parametrize("pk_type", PK_TYPES)
    def test_length_matches_request(self, pk_type: PkType) -> None:
        assert len(generate_primary_keys(pk_type, 7)) == 7

    @pytest.mark.parametrize("pk_type", PK_TYPES)
    def test_zero_n_returns_empty_list(self, pk_type: PkType) -> None:
        """All three pk types treat ``n=0`` as a valid no-op rather than an error."""
        assert generate_primary_keys(pk_type, 0) == []

    def test_uuid_produces_unique_keys_within_batch(self) -> None:
        """Symmetric to the dxid uniqueness check: UUID v4's 122-bit entropy
        means a 500-key batch should never collide."""
        keys = generate_primary_keys("uuid", 500)
        assert len(set(keys)) == 500

    def test_int_sequence_is_non_stateful(self) -> None:
        """Each call restarts at 1 — there is no hidden global counter that
        would cause cross-call key collisions in callers building multiple tables."""
        first = generate_primary_keys("int", 3)
        second = generate_primary_keys("int", 3)
        assert first == second == [1, 2, 3]

    def test_invalid_pk_type_raises_assertion(self) -> None:
        """``Literal`` types are enforced statically; this guards the runtime
        ``assert_never`` fallback against accidental removal of the dispatch arm."""
        with pytest.raises(AssertionError):
            generate_primary_keys("snowflake", 1)  # type: ignore[arg-type]
