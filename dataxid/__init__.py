# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Dataxid Python SDK.

Privacy-preserving synthetic data generation.
Raw data never leaves your environment — only embeddings cross the API boundary.

Quick start::

    import dataxid
    import pandas as pd

    dataxid.api_key = "dx_test_..."
    df = pd.read_csv("data.csv")

    synthetic = dataxid.synthesize(data=df, n_samples=1000)
"""

from __future__ import annotations

__version__ = "0.1.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from dataxid._log import disable_logging, enable_logging
from dataxid._log import logger as _logger
from dataxid._log import setup_logging as _setup_logging
from dataxid._table import Table
from dataxid.exceptions import (
    APIError,
    AuthenticationError,
    ConflictError,
    DataxidError,
    InvalidRequestError,
    ModelNotReadyError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    TrainingError,
    TrainingTimeoutError,
)
from dataxid.training._config import ModelConfig
from dataxid.training._model import Model

api_key: str | None = None
base_url: str = "https://api.dataxid.com"


_setup_logging()


def synthesize(
    data: pd.DataFrame,
    n_samples: int = 100,
    config: dict | ModelConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    parent: pd.DataFrame | None = None,
    parent_encoding_types: dict[str, str] | None = None,
    foreign_key: str | None = None,
    parent_key: str | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic data in one call.

    This is the simplest way to use Dataxid. Equivalent to::

        model = dataxid.Model.create(data=df, n_samples=n)
        synthetic = model.generate()
        model.delete()

    Raw data never leaves your machine. Only embeddings (64 floats/row)
    are sent to the API.

    Args:
        data: Training DataFrame
        n_samples: Number of synthetic rows to generate
        config: Training config (model_size, embedding_dim, max_epochs, etc.)
        api_key: Override dataxid.api_key
        base_url: Override dataxid.base_url
        parent: Parent table for context-aware generation (1:1 aligned or joined via foreign_key)
        parent_encoding_types: Encoding overrides for parent columns
        foreign_key: FK column in data linking rows to parent (enables sequential mode)
        parent_key: PK column in parent table (inferred from foreign_key if same name)

    Returns:
        DataFrame with synthetic data (target columns only)
    """
    model = Model.create(
        data=data,
        n_samples=n_samples,
        config=config,
        api_key=api_key,
        base_url=base_url,
        parent=parent,
        parent_encoding_types=parent_encoding_types,
        foreign_key=foreign_key,
        parent_key=parent_key,
    )
    try:
        return model.generate(n_samples=n_samples)
    except Exception:
        _logger.warning(
            "generate() failed — deleting model %s and re-raising.", model.id
        )
        raise
    finally:
        model.delete()


def synthesize_tables(
    tables: dict[str, Table],
    config: dict | ModelConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Synthesize multiple related tables with referential integrity.

    Trains and generates each table in dependency order (parents first).
    Primary keys are excluded from training and auto-assigned after generation.
    Foreign keys in child tables reference the synthetic parent's primary keys.

    Example::

        from dataxid import Table

        accounts = Table(accounts_df, primary_key="account_id")
        transactions = Table(transactions_df,
                             foreign_keys={"account_id": accounts})

        syn = dataxid.synthesize_tables({
            "accounts": accounts,
            "transactions": transactions,
        })
        syn["accounts"]       # synthetic accounts with auto-assigned PK
        syn["transactions"]   # synthetic transactions with valid FK references

    Args:
        tables: Mapping of table name to Table definition.
        config: Training config applied to all tables.
        api_key: Override dataxid.api_key.
        base_url: Override dataxid.base_url.

    Returns:
        Dict mapping table name to synthetic DataFrame.
    """
    from dataxid._table import (
        _assign_primary_keys,
        _remap_foreign_keys,
        _resolve_fk_targets,
        _topological_sort,
        _validate_tables,
    )

    _validate_tables(tables)
    order = _topological_sort(tables)
    assert order is not None  # _validate_tables guarantees no cycles
    fk_targets = _resolve_fk_targets(tables)

    results: dict[str, pd.DataFrame] = {}

    for name in order:
        tbl = tables[name]
        pk_col = tbl.primary_key
        resolved_fks = fk_targets[name]

        training_data = tbl.data
        if pk_col is not None:
            training_data = training_data.drop(columns=[pk_col])

        ctx_fk = _pick_context_fk(tbl)
        if ctx_fk:
            parent_name = resolved_fks[ctx_fk]
            parent_tbl = tables[parent_name]
            parent_syn = results[parent_name]

            real_parent = parent_tbl.data
            model = Model.create(
                data=training_data,
                config=config,
                api_key=api_key,
                base_url=base_url,
                parent=real_parent,
                parent_key=parent_tbl.primary_key,
                foreign_key=ctx_fk,
            )
            try:
                syn_df = model.generate(parent=parent_syn)
            except Exception:
                _logger.warning(
                    "generate() failed for table '%s' — deleting model %s and re-raising.",
                    name, model.id,
                )
                raise
            finally:
                model.delete()
        else:
            syn_df = synthesize(
                data=training_data,
                n_samples=len(tbl.data),
                config=config,
                api_key=api_key,
                base_url=base_url,
            )

        for fk_c, par_name in resolved_fks.items():
            if fk_c == ctx_fk:
                continue
            par_syn = results[par_name]
            par_pk = tables[par_name].primary_key
            if par_pk is not None:
                syn_df = _remap_foreign_keys(syn_df, fk_c, par_syn, par_pk)

        if pk_col is not None:
            syn_df = _assign_primary_keys(syn_df, pk_col)

        results[name] = syn_df

    return results


def _pick_context_fk(tbl: Table) -> str | None:
    """Determine which FK column to use as sequential context.

    Returns None if the table has no foreign_keys or sequential=False.
    For single FK → that FK. For multiple FKs → sequence_by (validated by Table).
    """
    if not tbl.foreign_keys or not tbl.sequential:
        return None
    if tbl.sequence_by:
        return tbl.sequence_by
    if len(tbl.foreign_keys) == 1:
        return next(iter(tbl.foreign_keys))
    return None


__all__ = [
    "enable_logging",
    "disable_logging",
    "api_key",
    "base_url",
    "synthesize",
    "synthesize_tables",
    "Table",
    "Model",
    "ModelConfig",
    "DataxidError",
    "AuthenticationError",
    "InvalidRequestError",
    "ModelNotReadyError",
    "NotFoundError",
    "RateLimitError",
    "QuotaExceededError",
    "ConflictError",
    "TrainingError",
    "TrainingTimeoutError",
    "APIError",
]
