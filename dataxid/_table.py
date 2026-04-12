# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-table synthesis — table definitions and orchestration helpers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from dataxid.exceptions import InvalidRequestError

# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class Table:
    """
    A table definition for multi-table synthesis.

    Example::

        from dataxid import Table

        Table(districts_df, primary_key="district_id")
        Table(accounts_df, primary_key="account_id",
              references={"district_id": "districts"})
        Table(transactions_df,
              references={"account_id": "accounts"},
              context_key="account_id")

    Args:
        data: Training DataFrame for this table.
        primary_key: Column name to use as primary key. Excluded from training,
            auto-assigned after generation (1-based auto-increment).
        references: Foreign key mapping ``{fk_column: referenced_table_name}``.
            Ensures referential integrity: FK values in the generated table
            are remapped to valid synthetic parent PKs.
        context_key: FK column to use as sequential training context. Must be
            a key in ``references``. When set, the table is generated
            conditioned on the parent (one group of rows per parent entity).
            When ``None`` (default), the table is generated independently
            and FK columns are only remapped for integrity.
    """

    data: pd.DataFrame
    primary_key: str | None = None
    references: dict[str, str] = field(default_factory=dict)
    context_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise InvalidRequestError(
                f"Table.data must be a DataFrame, got {type(self.data).__name__}.",
                param="data",
            )
        if self.primary_key is not None and self.primary_key not in self.data.columns:
            raise InvalidRequestError(
                f"primary_key '{self.primary_key}' not found in DataFrame columns: "
                f"{list(self.data.columns)}.",
                param="primary_key",
            )
        for fk_col in self.references:
            if fk_col not in self.data.columns:
                raise InvalidRequestError(
                    f"Foreign key column '{fk_col}' not found in DataFrame columns: "
                    f"{list(self.data.columns)}.",
                    param="references",
                )
        if self.context_key is not None and self.context_key not in self.references:
            raise InvalidRequestError(
                f"context_key '{self.context_key}' must be one of the references keys: "
                f"{list(self.references.keys())}.",
                param="context_key",
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_tables(tables: dict[str, Table]) -> None:
    """Validate table definitions: FK targets exist, referenced tables have PK, no cycles."""
    if not tables:
        raise InvalidRequestError("tables must contain at least one table.", param="tables")

    for name, tbl in tables.items():
        for fk_col, ref_table in tbl.references.items():
            if ref_table not in tables:
                raise InvalidRequestError(
                    f"Table '{name}' references '{ref_table}' via column '{fk_col}', "
                    f"but '{ref_table}' is not in tables.",
                    param="references",
                )
            if tables[ref_table].primary_key is None:
                raise InvalidRequestError(
                    f"Table '{name}' references '{ref_table}', but '{ref_table}' "
                    f"has no primary_key defined.",
                    param="primary_key",
                )

    order = _topological_sort(tables)
    if order is None:
        raise InvalidRequestError(
            "Circular dependency detected among table references.",
            param="references",
        )


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topological_sort(tables: dict[str, Table]) -> list[str] | None:
    """Return generation order (parents first) or None if cycle detected."""
    in_degree: dict[str, int] = {name: 0 for name in tables}
    dependents: dict[str, list[str]] = {name: [] for name in tables}

    for name, tbl in tables.items():
        parents = set(tbl.references.values())
        in_degree[name] = len(parents)
        for parent in parents:
            dependents[parent].append(name)

    queue: deque[str] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    order: list[str] = []

    while queue:
        current = queue.popleft()
        order.append(current)
        for child in dependents[current]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(tables):
        return None
    return order


# ---------------------------------------------------------------------------
# PK / FK helpers
# ---------------------------------------------------------------------------

def _assign_primary_keys(df: pd.DataFrame, pk_column: str) -> pd.DataFrame:
    """Insert 1-based auto-increment primary key as first column."""
    result = df.copy()
    result.insert(0, pk_column, range(1, len(result) + 1))
    return result


def _remap_foreign_keys(
    child_df: pd.DataFrame,
    fk_column: str,
    parent_df: pd.DataFrame,
    parent_pk: str,
) -> pd.DataFrame:
    """Ensure all FK values in child exist in the synthetic parent's PK domain."""
    valid_pks = set(parent_df[parent_pk])
    result = child_df.copy()

    orphan_mask = ~result[fk_column].isin(valid_pks)
    if orphan_mask.any():
        pk_values = parent_df[parent_pk].values
        result.loc[orphan_mask, fk_column] = [
            pk_values[i % len(pk_values)]
            for i in range(orphan_mask.sum())
        ]

    return result
