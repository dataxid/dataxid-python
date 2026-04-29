# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-table synthesis — table definitions and orchestration helpers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from dataxid._pk import PK_TYPES, PkType, generate_primary_keys
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

        districts = Table(districts_df, primary_key="district_id")
        accounts = Table(accounts_df, primary_key="account_id",
                         foreign_keys={"district_id": districts})
        transactions = Table(transactions_df,
                             foreign_keys={"account_id": accounts})

    Args:
        data: Training DataFrame for this table.
        primary_key: Column name to use as primary key. Excluded from training,
            auto-assigned after generation according to ``pk_type``.
        foreign_keys: Foreign key mapping ``{fk_column: parent_Table}``.
            When ``sequential`` is True (default), the child table is generated
            conditioned on the parent — preserving correlations. FK values in
            the generated table are remapped to valid synthetic parent PKs.
        sequential: When True (default) and ``foreign_keys`` is set, use
            sequential generation conditioned on the parent. When False,
            the table is generated independently and FK columns are only
            remapped for referential integrity.
        sequence_by: FK column to use as the primary sequential context when
            the table has multiple foreign keys. Must be a key in
            ``foreign_keys``. Required when ``len(foreign_keys) > 1`` and
            ``sequential`` is True; ignored otherwise.
        pk_type: Primary-key format used when assigning synthetic keys.
            - ``"dxid"`` (default): ``dxid_`` prefix + 22 char base62 body.
              Globally unique, URL/DB safe, 128-bit entropy.
            - ``"int"``: 1-based auto-increment integers.
            - ``"uuid"``: standard UUID v4 strings.
            Ignored when ``primary_key`` is ``None``.
    """

    data: pd.DataFrame
    primary_key: str | None = None
    foreign_keys: dict[str, Table] = field(default_factory=dict)
    sequential: bool = True
    sequence_by: str | None = None
    pk_type: PkType = "dxid"

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise InvalidRequestError(
                f"Table.data must be a DataFrame, got {type(self.data).__name__}.",
                param="data",
            )
        if self.pk_type not in PK_TYPES:
            raise InvalidRequestError(
                f"pk_type must be one of {PK_TYPES}, got {self.pk_type!r}.",
                param="pk_type",
            )
        if self.primary_key is not None and self.primary_key not in self.data.columns:
            raise InvalidRequestError(
                f"primary_key '{self.primary_key}' not found in DataFrame columns: "
                f"{list(self.data.columns)}.",
                param="primary_key",
            )
        for fk_col, parent_table in self.foreign_keys.items():
            if fk_col not in self.data.columns:
                raise InvalidRequestError(
                    f"Foreign key column '{fk_col}' not found in DataFrame columns: "
                    f"{list(self.data.columns)}.",
                    param="foreign_keys",
                )
            if fk_col == self.primary_key:
                raise InvalidRequestError(
                    f"Foreign key '{fk_col}' cannot also be the primary_key "
                    f"of the same table.",
                    param="foreign_keys",
                )
            if not isinstance(parent_table, Table):
                raise InvalidRequestError(
                    f"foreign_keys values must be Table instances, got "
                    f"{type(parent_table).__name__} for key '{fk_col}'.",
                    param="foreign_keys",
                )
            if parent_table is self:
                raise InvalidRequestError(
                    f"Foreign key '{fk_col}' references the same Table "
                    f"instance — self-references are not supported.",
                    param="foreign_keys",
                )
            if parent_table.primary_key is None:
                raise InvalidRequestError(
                    f"Referenced table for '{fk_col}' must have a primary_key defined.",
                    param="foreign_keys",
                )
        if self.sequence_by is not None:
            if not self.sequential:
                raise InvalidRequestError(
                    "sequence_by and sequential=False are mutually exclusive.",
                    param="sequence_by",
                )
            if self.sequence_by not in self.foreign_keys:
                raise InvalidRequestError(
                    f"sequence_by '{self.sequence_by}' must be one of the foreign_keys: "
                    f"{list(self.foreign_keys.keys())}.",
                    param="sequence_by",
                )
        if (
            self.sequential
            and len(self.foreign_keys) > 1
            and self.sequence_by is None
        ):
            raise InvalidRequestError(
                f"Table has {len(self.foreign_keys)} foreign keys. Use sequence_by "
                f"to specify which relationship to use for sequential generation. "
                f"Options: {list(self.foreign_keys.keys())}.",
                param="sequence_by",
            )


# ---------------------------------------------------------------------------
# Internal helpers — resolve Table object references to table names
# ---------------------------------------------------------------------------

def _build_table_identity(tables: dict[str, Table]) -> dict[int, str]:
    """Map Table object id → table name for FK resolution."""
    return {id(tbl): name for name, tbl in tables.items()}


def _resolve_fk_targets(
    tables: dict[str, Table],
) -> dict[str, dict[str, str]]:
    """Resolve foreign_keys {fk_col: Table} → {fk_col: parent_name} for each table."""
    identity = _build_table_identity(tables)
    result: dict[str, dict[str, str]] = {}
    for name, tbl in tables.items():
        resolved: dict[str, str] = {}
        for fk_col, parent_table in tbl.foreign_keys.items():
            parent_name = identity.get(id(parent_table))
            if parent_name is None:
                raise InvalidRequestError(
                    f"Table '{name}' references a Table object via '{fk_col}' "
                    f"that is not in the tables dict.",
                    param="foreign_keys",
                )
            resolved[fk_col] = parent_name
        result[name] = resolved
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_tables(tables: dict[str, Table]) -> None:
    """Validate table definitions: FK targets exist, referenced tables have PK, no cycles."""
    if not isinstance(tables, dict):
        raise InvalidRequestError(
            f"tables must be a dict mapping name to Table, got "
            f"{type(tables).__name__}.",
            param="tables",
        )
    if not tables:
        raise InvalidRequestError("tables must contain at least one table.", param="tables")

    for name, tbl in tables.items():
        if not isinstance(name, str):
            raise InvalidRequestError(
                f"tables keys must be strings, got {type(name).__name__} ({name!r}).",
                param="tables",
            )
        if not isinstance(tbl, Table):
            raise InvalidRequestError(
                f"tables[{name!r}] must be a Table instance, got "
                f"{type(tbl).__name__}.",
                param="tables",
            )

    _resolve_fk_targets(tables)

    order = _topological_sort(tables)
    if order is None:
        raise InvalidRequestError(
            "Circular dependency detected among table foreign_keys.",
            param="foreign_keys",
        )


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topological_sort(tables: dict[str, Table]) -> list[str] | None:
    """Return generation order (parents first) or None if cycle detected."""
    resolved = _resolve_fk_targets(tables)

    in_degree: dict[str, int] = {name: 0 for name in tables}
    dependents: dict[str, list[str]] = {name: [] for name in tables}

    for name in tables:
        parents = set(resolved[name].values())
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

def _assign_primary_keys(
    df: pd.DataFrame, pk_column: str, pk_type: PkType = "dxid",
) -> pd.DataFrame:
    """Insert a freshly generated primary key column at position 0."""
    result = df.copy()
    result.insert(0, pk_column, generate_primary_keys(pk_type, len(result)))
    return result


def _remap_foreign_keys(
    child_df: pd.DataFrame,
    fk_column: str,
    parent_df: pd.DataFrame,
    parent_pk: str,
) -> pd.DataFrame:
    """Ensure all FK values in child exist in the synthetic parent's PK domain.

    When the parent's synthetic PK dtype differs from the child's FK dtype
    (e.g. parent was regenerated as ``dxid`` strings while the child still
    holds the original integer keys), the FK column is first cast to the
    parent PK dtype so the remap stays type-consistent.
    """
    valid_pks = set(parent_df[parent_pk])
    result = child_df.copy()

    parent_dtype = parent_df[parent_pk].dtype
    if result[fk_column].dtype != parent_dtype:
        result[fk_column] = result[fk_column].astype(parent_dtype)

    orphan_mask = ~result[fk_column].isin(valid_pks)
    if orphan_mask.any():
        pk_values = parent_df[parent_pk].values
        result.loc[orphan_mask, fk_column] = [
            pk_values[i % len(pk_values)]
            for i in range(orphan_mask.sum())
        ]

    return result
