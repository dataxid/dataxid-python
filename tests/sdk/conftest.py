# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for SDK tests.

Provides small, deterministic DataFrames covering the four data shapes
the SDK supports — flat, flat-with-context, sequential, sequential-with-context —
plus pre-built ``Encoder`` instances at each pipeline stage.

All fixtures are function-scoped (the pytest default). DataFrames are
mutable and ``Encoder`` is stateful, so a fresh instance per test prevents
cross-test contamination at the cost of negligible setup time.
"""

import random
from collections.abc import Iterator

import pandas as pd
import pytest
import torch

from dataxid.encoder import Encoder


@pytest.fixture(autouse=True)
def _isolate_torch_rng() -> Iterator[None]:
    """Pin ``torch``'s global RNG to a fixed seed for each test.

    ``Encoder`` initializes its model weights from the global generator,
    so without isolation any test that constructs an encoder leaks state
    into later tests, making reproductions of a single failure depend on
    the order other tests ran in.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(0)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


@pytest.fixture(autouse=True)
def _isolate_python_rng() -> Iterator[None]:
    """Snapshot and restore Python's ``random`` global state per test.

    Several SDK paths (primary-key generation, sample shuffling) draw from
    ``random``. Tests that call ``random.seed(...)`` would otherwise leak
    the seeded sequence into every later test, making failures order-dependent.
    """
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Flat table — 8 rows with one numeric and one categorical column."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
    })


@pytest.fixture
def ctx_df() -> pd.DataFrame:
    """Parent table for ``sample_df``, aligned 1:1 (one parent row per child row)."""
    return pd.DataFrame({
        "region": ["North", "South", "North", "East", "South", "North", "East", "South"],
        "population": [100, 200, 100, 300, 200, 100, 300, 200],
    })


@pytest.fixture
def seq_df() -> pd.DataFrame:
    """
    Sequential (1:N) child table — 3 entities with variable-length sequences (3, 2, 3 rows).

    Foreign key ``account_id`` references the parent's primary key ``id``
    (see :func:`seq_ctx_df`). The intentionally different column names
    exercise the SDK's ``parent_key`` / ``foreign_key`` parameters.
    """
    return pd.DataFrame({
        "account_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0, 50.0, 75.0, 60.0],
        "type": ["credit", "debit", "credit", "debit", "credit", "credit", "debit", "credit"],
    })


@pytest.fixture
def seq_ctx_df() -> pd.DataFrame:
    """Parent table for ``seq_df`` — one row per entity, primary key ``id``."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "region": ["North", "South", "East"],
    })


@pytest.fixture
def encoder() -> Encoder:
    """Fresh ``Encoder`` with a minimal CPU config — fast enough for CI."""
    return Encoder(embedding_dim=16, model_size="small", device="cpu")


@pytest.fixture
def analyzed_encoder(encoder: Encoder, sample_df: pd.DataFrame) -> Encoder:
    """Encoder after :meth:`Encoder.analyze` — ready for ``prepare`` / ``encode``."""
    encoder.analyze(sample_df)
    return encoder


@pytest.fixture
def prepared_encoder(analyzed_encoder: Encoder, sample_df: pd.DataFrame) -> Encoder:
    """
    Encoder after ``analyze`` + ``prepare`` — ready for ``encode_batch`` / ``backward``.

    ``sample_df`` is reused intentionally: ``prepare`` must run on the same
    data that ``analyze`` observed.
    """
    analyzed_encoder.prepare(sample_df)
    return analyzed_encoder
