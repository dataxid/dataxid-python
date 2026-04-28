# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :func:`dataxid.synthesize_tables` and the supporting machinery in
:mod:`dataxid._table`.

The orchestrator coordinates multi-table synthesis by:

* validating the :class:`~dataxid.Table` graph (FK targets, cycles, sequencing
  constraints),
* topologically sorting tables so parents are generated before children,
* dispatching root tables to :func:`dataxid.synthesize` and child tables to
  :meth:`dataxid.Model.create` / ``Model.generate``,
* assigning synthetic primary keys (:func:`_assign_primary_keys`) and remapping
  foreign keys to maintain referential integrity (:func:`_remap_foreign_keys`).

All API entry points are mocked; no network calls or real training occurs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import dataxid
from dataxid import Distribution, ModelConfig, Table
from dataxid._table import (
    _assign_primary_keys,
    _remap_foreign_keys,
    _topological_sort,
    _validate_tables,
)
from dataxid.exceptions import InvalidRequestError

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def accounts_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": [101, 102, 103],
            "district": ["A", "B", "A"],
            "balance": [1000, 2000, 1500],
        }
    )


@pytest.fixture()
def transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": [101, 101, 102, 103, 103],
            "amount": [100.0, 200.0, 300.0, 50.0, 75.0],
            "type": ["credit", "debit", "credit", "credit", "debit"],
        }
    )


@pytest.fixture()
def _api_key() -> Iterator[None]:
    """Set a dummy API key for orchestrator tests; restore after."""
    original = dataxid.api_key
    dataxid.api_key = "dx_test_synthesize_tables"
    try:
        yield
    finally:
        dataxid.api_key = original


def _make_fake_df(
    data: pd.DataFrame,
    n: int,
    parent: pd.DataFrame | None = None,
    parent_key: str | None = None,
    foreign_key: str | None = None,
) -> pd.DataFrame:
    """Build a synthetic DataFrame mirroring ``data``'s column structure.

    Numeric columns are filled with ``range(n)``; categorical columns cycle
    through observed values. When ``parent`` and FK info are supplied, the
    FK column is filled by cycling the parent's PK values, guaranteeing
    referential integrity in tests that rely on it.
    """
    result: dict[str, list[Any]] = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            result[col] = list(range(n))
        else:
            vals = data[col].unique().tolist()
            result[col] = [vals[i % len(vals)] for i in range(n)]

    if foreign_key and parent is not None and parent_key:
        pk_values = parent[parent_key].values
        result[foreign_key] = [pk_values[i % len(pk_values)] for i in range(n)]

    return pd.DataFrame(result)


def _mock_synthesize(
    data: pd.DataFrame,
    n_samples: int | None = None,
    parent: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Stand-in for :func:`dataxid.synthesize` (root / flat tables)."""
    n = n_samples or (len(parent) if parent is not None else len(data))
    return _make_fake_df(
        data, n, parent, kwargs.get("parent_key"), kwargs.get("foreign_key")
    )


def _mock_model_create(
    data: pd.DataFrame,
    parent: pd.DataFrame | None = None,
    foreign_key: str | None = None,
    parent_key: str | None = None,
    **kwargs: Any,
) -> MagicMock:
    """Stand-in for :meth:`dataxid.Model.create` (sequential child tables)."""
    mock_model = MagicMock()
    mock_model.id = "mdl_test_mock"

    captured = {
        "data": data,
        "parent": parent,
        "foreign_key": foreign_key,
        "parent_key": parent_key,
    }

    def _generate(parent: pd.DataFrame | None = None, **kw: Any) -> pd.DataFrame:
        cd = parent if parent is not None else captured["parent"]
        n = len(cd) if cd is not None else len(captured["data"])
        return _make_fake_df(
            captured["data"], n, cd, captured["parent_key"], captured["foreign_key"]
        )

    mock_model.generate = _generate
    mock_model.delete = MagicMock()
    return mock_model


@contextmanager
def _mock_synthesize_stack(
    synthesize_side_effect: Any = _mock_synthesize,
    create_side_effect: Any = _mock_model_create,
) -> Iterator[None]:
    """Patch both :func:`dataxid.synthesize` and :meth:`Model.create` together."""
    with (
        patch("dataxid.synthesize", side_effect=synthesize_side_effect),
        patch("dataxid.Model.create", side_effect=create_side_effect),
    ):
        yield


def _make_capturing_create(
    create_calls: list[dict[str, Any]] | None = None,
    generate_calls: list[dict[str, Any]] | None = None,
) -> Any:
    """Factory: a ``Model.create`` side_effect that records call kwargs.

    Pass ``create_calls`` to capture ``Model.create`` invocations and / or
    ``generate_calls`` to capture the resulting model's ``generate`` calls.
    Either list may be ``None`` to skip recording.
    """

    def _side_effect(
        data: pd.DataFrame,
        parent: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> MagicMock:
        if create_calls is not None:
            create_calls.append({"data": data, "parent": parent, **kwargs})
        mock_model = _mock_model_create(data, parent=parent, **kwargs)

        if generate_calls is not None:
            original_generate = mock_model.generate

            def _capture_generate(**kw: Any) -> pd.DataFrame:
                generate_calls.append(kw)
                return original_generate(**kw)

            mock_model.generate = _capture_generate
        return mock_model

    return _side_effect


def _make_capturing_synthesize(
    synth_calls: list[dict[str, Any]],
) -> Any:
    """Factory: a ``synthesize`` side_effect that records call kwargs."""

    def _side_effect(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        synth_calls.append(kwargs)
        return _mock_synthesize(data, **kwargs)

    return _side_effect


# ---------------------------------------------------------------------------
# Table dataclass validation
# ---------------------------------------------------------------------------


class TestTableValidation:
    """Field-level invariants of the :class:`~dataxid.Table` dataclass."""

    def test_valid_table(self, accounts_df: pd.DataFrame) -> None:
        tbl = Table(accounts_df, primary_key="account_id")
        assert tbl.primary_key == "account_id"

    def test_data_must_be_dataframe(self) -> None:
        with pytest.raises(InvalidRequestError, match="must be a DataFrame"):
            Table("not_a_dataframe")

    def test_primary_key_must_exist(self, accounts_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="not found"):
            Table(accounts_df, primary_key="nonexistent")

    def test_fk_column_must_exist(self, accounts_df: pd.DataFrame) -> None:
        parent = Table(accounts_df, primary_key="account_id")
        with pytest.raises(InvalidRequestError, match="Foreign key column"):
            Table(accounts_df, foreign_keys={"nonexistent": parent})

    def test_fk_value_must_be_table(self, accounts_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="must be Table instances"):
            Table(accounts_df, foreign_keys={"account_id": "accounts"})

    def test_fk_parent_must_have_pk(self, accounts_df: pd.DataFrame) -> None:
        parent_no_pk = Table(accounts_df)
        with pytest.raises(InvalidRequestError, match="must have a primary_key"):
            Table(accounts_df, foreign_keys={"account_id": parent_no_pk})

    def test_foreign_keys_default_empty(self, accounts_df: pd.DataFrame) -> None:
        tbl = Table(accounts_df)
        assert tbl.foreign_keys == {}

    def test_positional_data(self, accounts_df: pd.DataFrame) -> None:
        tbl = Table(accounts_df)
        assert len(tbl.data) == 3

    def test_sequential_default_true(self, accounts_df: pd.DataFrame) -> None:
        tbl = Table(accounts_df)
        assert tbl.sequential is True

    def test_sequence_by_must_be_in_foreign_keys(
        self, accounts_df: pd.DataFrame
    ) -> None:
        parent = Table(accounts_df, primary_key="account_id")
        with pytest.raises(InvalidRequestError, match="must be one of the foreign_keys"):
            Table(
                accounts_df,
                foreign_keys={"account_id": parent},
                sequence_by="nonexistent",
            )

    def test_sequence_by_and_sequential_false_mutually_exclusive(
        self, accounts_df: pd.DataFrame
    ) -> None:
        parent = Table(accounts_df, primary_key="account_id")
        with pytest.raises(InvalidRequestError, match="mutually exclusive"):
            Table(
                accounts_df,
                foreign_keys={"account_id": parent},
                sequential=False,
                sequence_by="account_id",
            )

    def test_multi_fk_requires_sequence_by(self) -> None:
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "product_id": [10, 20, 30],
                "amount": [100, 200, 300],
            }
        )
        customers = Table(
            pd.DataFrame({"customer_id": [1, 2, 3]}), primary_key="customer_id"
        )
        products = Table(
            pd.DataFrame({"product_id": [10, 20, 30]}), primary_key="product_id"
        )
        with pytest.raises(InvalidRequestError, match="sequence_by"):
            Table(df, foreign_keys={"customer_id": customers, "product_id": products})

    def test_multi_fk_with_sequence_by_ok(self) -> None:
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "product_id": [10, 20, 30],
                "amount": [100, 200, 300],
            }
        )
        customers = Table(
            pd.DataFrame({"customer_id": [1, 2, 3]}), primary_key="customer_id"
        )
        products = Table(
            pd.DataFrame({"product_id": [10, 20, 30]}), primary_key="product_id"
        )
        tbl = Table(
            df,
            foreign_keys={"customer_id": customers, "product_id": products},
            sequence_by="customer_id",
        )
        assert tbl.sequence_by == "customer_id"


class TestTablePkType:
    """``Table.pk_type`` — accepted values and storage semantics."""

    def test_default_is_dxid(self, accounts_df: pd.DataFrame) -> None:
        tbl = Table(accounts_df, primary_key="account_id")
        assert tbl.pk_type == "dxid"

    @pytest.mark.parametrize("pk_type", ["dxid", "int", "uuid"])
    def test_accepts_valid_types(
        self, accounts_df: pd.DataFrame, pk_type: str
    ) -> None:
        tbl = Table(accounts_df, primary_key="account_id", pk_type=pk_type)
        assert tbl.pk_type == pk_type

    def test_invalid_pk_type_rejected(self, accounts_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="pk_type must be one of"):
            Table(accounts_df, primary_key="account_id", pk_type="snowflake")

    def test_pk_type_valid_without_primary_key(
        self, accounts_df: pd.DataFrame
    ) -> None:
        """``pk_type`` is stored even when ``primary_key`` is None — unused
        but must not raise."""
        tbl = Table(accounts_df, pk_type="int")
        assert tbl.pk_type == "int"


# ---------------------------------------------------------------------------
# Internal helpers — _validate_tables / _topological_sort
# ---------------------------------------------------------------------------


class TestValidateTables:
    """Cross-table validation (FK targets, cycles)."""

    def test_empty_tables(self) -> None:
        with pytest.raises(InvalidRequestError, match="at least one"):
            _validate_tables({})

    def test_fk_target_not_in_tables(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        orphan_parent = Table(accounts_df, primary_key="account_id")
        child = Table(transactions_df, foreign_keys={"account_id": orphan_parent})
        with pytest.raises(InvalidRequestError, match="not in the tables dict"):
            _validate_tables({"transactions": child})

    def test_valid_two_tables(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        _validate_tables(
            {
                "accounts": acct,
                "transactions": Table(
                    transactions_df, foreign_keys={"account_id": acct}
                ),
            }
        )

    def test_circular_dependency(self, accounts_df: pd.DataFrame) -> None:
        a = Table(accounts_df, primary_key="account_id")
        b = Table(accounts_df, primary_key="account_id")
        # Bypass __post_init__ to wire a cycle that the dataclass would reject.
        object.__setattr__(a, "foreign_keys", {"district": b})
        object.__setattr__(b, "foreign_keys", {"district": a})
        with pytest.raises(InvalidRequestError, match="Circular dependency"):
            _validate_tables({"a": a, "b": b})


class TestTopologicalSort:
    """Parents before children; cycles return ``None``."""

    def test_single_table(self, accounts_df: pd.DataFrame) -> None:
        tables = {"accounts": Table(accounts_df, primary_key="account_id")}
        assert _topological_sort(tables) == ["accounts"]

    def test_parent_before_child(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        tables = {
            "transactions": Table(
                transactions_df, foreign_keys={"account_id": acct}
            ),
            "accounts": acct,
        }
        order = _topological_sort(tables)
        assert order is not None
        assert order.index("accounts") < order.index("transactions")

    def test_cycle_returns_none(self, accounts_df: pd.DataFrame) -> None:
        a = Table(accounts_df, primary_key="account_id")
        b = Table(accounts_df, primary_key="account_id")
        object.__setattr__(a, "foreign_keys", {"district": b})
        object.__setattr__(b, "foreign_keys", {"district": a})
        assert _topological_sort({"a": a, "b": b}) is None


# ---------------------------------------------------------------------------
# Internal helpers — _assign_primary_keys / _remap_foreign_keys
# ---------------------------------------------------------------------------


class TestAssignPrimaryKeys:
    """Synthetic PK assignment after generation."""

    def test_inserts_first_column(self) -> None:
        df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
        result = _assign_primary_keys(df, "id")
        assert list(result.columns) == ["id", "name"]

    def test_default_pk_type_is_dxid(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = _assign_primary_keys(df, "pk")
        assert all(isinstance(v, str) and v.startswith("dxid_") for v in result["pk"])

    def test_int_pk_type_one_based_increment(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = _assign_primary_keys(df, "pk", pk_type="int")
        assert list(result["pk"]) == [1, 2, 3]

    def test_uuid_pk_type_returns_uuid_strings(self) -> None:
        import uuid as _uuid

        df = pd.DataFrame({"x": [1, 2]})
        result = _assign_primary_keys(df, "pk", pk_type="uuid")
        for v in result["pk"]:
            assert isinstance(v, str)
            _uuid.UUID(v)

    def test_does_not_mutate_original(self) -> None:
        df = pd.DataFrame({"x": [1]})
        _assign_primary_keys(df, "pk")
        assert "pk" not in df.columns


class TestRemapForeignKeys:
    """FK remap preserves referential integrity to a synthetic parent."""

    def test_no_orphans_no_change(self) -> None:
        parent = pd.DataFrame({"pk": [1, 2, 3]})
        child = pd.DataFrame({"fk": [1, 2, 3, 1]})
        result = _remap_foreign_keys(child, "fk", parent, "pk")
        assert list(result["fk"]) == [1, 2, 3, 1]

    def test_orphans_remapped(self) -> None:
        parent = pd.DataFrame({"pk": [1, 2]})
        child = pd.DataFrame({"fk": [1, 999, 888]})
        result = _remap_foreign_keys(child, "fk", parent, "pk")
        assert set(result["fk"]).issubset({1, 2})

    def test_does_not_mutate_original(self) -> None:
        parent = pd.DataFrame({"pk": [1]})
        child = pd.DataFrame({"fk": [999]})
        _remap_foreign_keys(child, "fk", parent, "pk")
        assert child["fk"].iloc[0] == 999


# ---------------------------------------------------------------------------
# synthesize_tables — full orchestration (2-table)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestSynthesizeTables:
    """End-to-end orchestration over a parent → child pair."""

    def test_return_shape_contract(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """Returns ``dict[name -> DataFrame]`` keyed by the input tables."""
        acct = Table(accounts_df, primary_key="account_id")
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"accounts", "transactions"}
        assert all(isinstance(df, pd.DataFrame) for df in result.values())

    def test_pk_auto_assigned(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id", pk_type="int")
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )
        syn_accounts = result["accounts"]
        assert "account_id" in syn_accounts.columns
        assert syn_accounts["account_id"].is_unique
        assert list(syn_accounts["account_id"]) == [1, 2, 3]

    def test_pk_excluded_from_training(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """The PK column must be stripped before sending data for training,
        otherwise the model would learn meaningless ID statistics."""
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []

        def _capture(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
            synth_calls.append({"columns": list(data.columns), "kwargs": kwargs})
            return _mock_synthesize(data, **kwargs)

        with _mock_synthesize_stack(synthesize_side_effect=_capture):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert "account_id" not in synth_calls[0]["columns"]

    def test_referential_integrity(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )
        syn_acct_ids = set(result["accounts"]["account_id"])
        syn_trans_ids = set(result["transactions"]["account_id"])
        orphans = syn_trans_ids - syn_acct_ids
        assert len(orphans) == 0, f"Orphan FK values: {orphans}"

    def test_single_root_table(self, accounts_df: pd.DataFrame) -> None:
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {"accounts": Table(accounts_df, primary_key="account_id")}
            )
        assert "accounts" in result
        assert "account_id" in result["accounts"].columns

    def test_table_without_pk(self, accounts_df: pd.DataFrame) -> None:
        df = accounts_df.drop(columns=["account_id"])
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables({"data": Table(df)})
        assert "data" in result
        assert len(result["data"]) == len(df)

    def test_child_uses_real_parent_for_training(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """Training context uses the real parent rows; only generation
        switches to the synthetic parent."""
        acct = Table(accounts_df, primary_key="account_id")
        create_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(create_calls=create_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert len(create_calls) == 1
        ctx_used = create_calls[0]["parent"]
        assert list(ctx_used["account_id"]) == [101, 102, 103]

    def test_child_generates_with_synthetic_parent(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id", pk_type="int")
        generate_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(generate_calls=generate_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert len(generate_calls) == 1
        syn_parent = generate_calls[0]["parent"]
        assert list(syn_parent["account_id"]) == [1, 2, 3]

    def test_non_sequential_uses_flat_plus_remap(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """``foreign_keys`` + ``sequential=False`` → flat ``synthesize`` for
        both tables, FK column remapped post-hoc for referential integrity."""
        acct = Table(accounts_df, primary_key="account_id")
        synthesize_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synthesize_calls)
        ):
            result = dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df,
                        foreign_keys={"account_id": acct},
                        sequential=False,
                    ),
                }
            )

        assert len(synthesize_calls) == 2
        syn_acct_ids = set(result["accounts"]["account_id"])
        syn_trans_ids = set(result["transactions"]["account_id"])
        assert syn_trans_ids.issubset(syn_acct_ids)


# ---------------------------------------------------------------------------
# sequence_by — N-parent disambiguation
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestSequenceBy:
    """Multi-FK child uses ``sequence_by`` to pick its sequential parent."""

    def test_sequence_by_selects_correct_parent(self) -> None:
        df = pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "product_id": [10, 20, 30],
                "amount": [100, 200, 300],
            }
        )
        customers = Table(
            pd.DataFrame({"customer_id": [1, 2, 3], "name": ["A", "B", "C"]}),
            primary_key="customer_id",
        )
        products = Table(
            pd.DataFrame(
                {"product_id": [10, 20, 30], "category": ["X", "Y", "Z"]}
            ),
            primary_key="product_id",
        )
        create_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(create_calls=create_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "customers": customers,
                    "products": products,
                    "orders": Table(
                        df,
                        foreign_keys={
                            "customer_id": customers,
                            "product_id": products,
                        },
                        sequence_by="product_id",
                    ),
                }
            )

        assert len(create_calls) == 1
        assert create_calls[0]["foreign_key"] == "product_id"


# ---------------------------------------------------------------------------
# 3-table chain (grandparent -> parent -> child)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestThreeTableChain:
    """Topological execution and integrity over a 3-level FK chain."""

    @pytest.fixture()
    def districts_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"district_id": [1, 2, 3], "region": ["north", "south", "east"]}
        )

    @pytest.fixture()
    def chain_accounts_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "account_id": [101, 102, 103, 104],
                "district_id": [1, 1, 2, 3],
                "frequency": ["monthly", "weekly", "monthly", "weekly"],
            }
        )

    @pytest.fixture()
    def chain_transactions_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "account_id": [101, 101, 102, 103, 104],
                "amount": [100.0, 200.0, 300.0, 50.0, 75.0],
                "type": ["credit", "debit", "credit", "credit", "debit"],
            }
        )

    def _build_tables(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> tuple[Table, Table, Table]:
        districts = Table(districts_df, primary_key="district_id")
        accounts = Table(
            chain_accounts_df,
            primary_key="account_id",
            foreign_keys={"district_id": districts},
        )
        transactions = Table(
            chain_transactions_df, foreign_keys={"account_id": accounts}
        )
        return districts, accounts, transactions

    def test_returns_all_three_tables(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> None:
        dist, acct, tx = self._build_tables(
            districts_df, chain_accounts_df, chain_transactions_df
        )
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {"districts": dist, "accounts": acct, "transactions": tx}
            )
        assert set(result.keys()) == {"districts", "accounts", "transactions"}

    def test_root_uses_synthesize_children_use_create(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> None:
        """Only the root (no FK) goes through ``synthesize``; the two
        downstream tables go through ``Model.create``."""
        dist, acct, tx = self._build_tables(
            districts_df, chain_accounts_df, chain_transactions_df
        )
        synthesize_calls: list[dict[str, Any]] = []
        create_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synthesize_calls),
            create_side_effect=_make_capturing_create(create_calls=create_calls),
        ):
            dataxid.synthesize_tables(
                {"districts": dist, "accounts": acct, "transactions": tx}
            )

        assert len(synthesize_calls) == 1
        assert len(create_calls) == 2

    def test_pks_auto_assigned_all_levels(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> None:
        districts = Table(districts_df, primary_key="district_id", pk_type="int")
        accounts = Table(
            chain_accounts_df,
            primary_key="account_id",
            pk_type="int",
            foreign_keys={"district_id": districts},
        )
        transactions = Table(
            chain_transactions_df, foreign_keys={"account_id": accounts}
        )
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {
                    "districts": districts,
                    "accounts": accounts,
                    "transactions": transactions,
                }
            )
        assert list(result["districts"]["district_id"]) == [1, 2, 3]
        assert result["accounts"]["account_id"].is_unique
        assert result["accounts"]["account_id"].iloc[0] == 1

    def test_referential_integrity_all_levels(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> None:
        dist, acct, tx = self._build_tables(
            districts_df, chain_accounts_df, chain_transactions_df
        )
        with _mock_synthesize_stack():
            result = dataxid.synthesize_tables(
                {"districts": dist, "accounts": acct, "transactions": tx}
            )
        district_ids = set(result["districts"]["district_id"])
        acct_district_ids = set(result["accounts"]["district_id"])
        assert acct_district_ids.issubset(district_ids), (
            f"Orphan district_id in accounts: {acct_district_ids - district_ids}"
        )

        acct_ids = set(result["accounts"]["account_id"])
        trans_acct_ids = set(result["transactions"]["account_id"])
        assert trans_acct_ids.issubset(acct_ids), (
            f"Orphan account_id in transactions: {trans_acct_ids - acct_ids}"
        )

    def test_topological_order(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> None:
        dist, acct, tx = self._build_tables(
            districts_df, chain_accounts_df, chain_transactions_df
        )
        order = _topological_sort(
            {"transactions": tx, "districts": dist, "accounts": acct}
        )
        assert order is not None
        assert order.index("districts") < order.index("accounts")
        assert order.index("accounts") < order.index("transactions")

    def test_chain_context_trains_with_real_generates_with_synthetic(
        self,
        districts_df: pd.DataFrame,
        chain_accounts_df: pd.DataFrame,
        chain_transactions_df: pd.DataFrame,
    ) -> None:
        dist, acct, tx = self._build_tables(
            districts_df, chain_accounts_df, chain_transactions_df
        )
        create_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(create_calls=create_calls)
        ):
            dataxid.synthesize_tables(
                {"districts": dist, "accounts": acct, "transactions": tx}
            )

        assert len(create_calls) == 2
        tx_call = create_calls[1]
        parent = tx_call["parent"]
        parent_key = tx_call.get("parent_key")
        assert parent_key is not None
        assert list(parent[parent_key]) == [101, 102, 103, 104]


# ---------------------------------------------------------------------------
# synthesize_tables — distribution forwarding
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestSynthesizeTablesDistribution:
    """``distribution={...}`` is forwarded to the right backend per table."""

    def test_distribution_passed_to_root_table(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        dist = Distribution(column="district", probabilities={"A": 0.5, "B": 0.5})

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                distribution={"accounts": dist},
            )

        assert len(synth_calls) == 1
        assert synth_calls[0].get("distribution") is dist

    def test_distribution_passed_to_child_table(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        generate_calls: list[dict[str, Any]] = []
        dist = Distribution(
            column="type", probabilities={"credit": 0.5, "debit": 0.5}
        )

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(generate_calls=generate_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                distribution={"transactions": dist},
            )

        assert len(generate_calls) == 1
        assert generate_calls[0].get("distribution") is dist

    def test_distribution_none_by_default(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        generate_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls),
            create_side_effect=_make_capturing_create(generate_calls=generate_calls),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert synth_calls[0].get("distribution") is None
        assert generate_calls[0].get("distribution") is None

    def test_distribution_isolation_per_table(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """A distribution targeting ``transactions`` must not leak into the
        ``accounts`` call (and vice versa)."""
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        generate_calls: list[dict[str, Any]] = []
        dist = Distribution(
            column="type", probabilities={"credit": 0.5, "debit": 0.5}
        )

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls),
            create_side_effect=_make_capturing_create(generate_calls=generate_calls),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                distribution={"transactions": dist},
            )

        assert synth_calls[0].get("distribution") is None
        assert generate_calls[0].get("distribution") is dist


# ---------------------------------------------------------------------------
# synthesize_tables — conditions forwarding
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestSynthesizeTablesConditions:
    """``conditions={...}`` is forwarded to the right backend per table."""

    def test_conditions_passed_to_root_table(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        conds = pd.DataFrame({"district": ["A", "B", "A"]})

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                conditions={"accounts": conds},
            )

        assert len(synth_calls) == 1
        assert synth_calls[0].get("conditions") is conds

    def test_conditions_passed_to_child_table(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        generate_calls: list[dict[str, Any]] = []
        conds = pd.DataFrame(
            {"type": ["credit", "debit", "credit", "debit", "credit"]}
        )

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(generate_calls=generate_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                conditions={"transactions": conds},
            )

        assert len(generate_calls) == 1
        assert generate_calls[0].get("conditions") is conds

    def test_conditions_none_by_default(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        generate_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls),
            create_side_effect=_make_capturing_create(generate_calls=generate_calls),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert synth_calls[0].get("conditions") is None
        assert generate_calls[0].get("conditions") is None

    def test_conditions_isolation_per_table(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        generate_calls: list[dict[str, Any]] = []
        conds = pd.DataFrame(
            {"type": ["credit", "debit", "credit", "debit", "credit"]}
        )

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls),
            create_side_effect=_make_capturing_create(generate_calls=generate_calls),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                conditions={"transactions": conds},
            )

        assert synth_calls[0].get("conditions") is None
        assert generate_calls[0].get("conditions") is conds

    def test_conditions_long_format_forwarded_to_child(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """Long-format conditions (rows include the FK column) are passed
        through unchanged so the model can match them against per-parent
        sequences."""
        acct = Table(accounts_df, primary_key="account_id")
        generate_calls: list[dict[str, Any]] = []
        conds = pd.DataFrame(
            {
                "account_id": [101, 101, 102, 103],
                "type": ["credit", "debit", "credit", "debit"],
            }
        )

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(generate_calls=generate_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                conditions={"transactions": conds},
            )

        assert len(generate_calls) == 1
        passed_conds = generate_calls[0].get("conditions")
        assert passed_conds is conds
        assert "account_id" in passed_conds.columns


# ---------------------------------------------------------------------------
# Internal helper invariants (probed)
# ---------------------------------------------------------------------------


class TestAssignPrimaryKeysDeterminism:
    """Pin down ``_assign_primary_keys`` invariants the orchestrator relies on."""

    def test_int_pk_is_stateless_across_calls(self) -> None:
        """Every ``int`` assignment starts at 1 — without this, the orchestrator
        could not assign monotonic PKs to multiple sibling tables."""
        df = pd.DataFrame({"x": [10, 20, 30]})
        first = _assign_primary_keys(df, "pk", pk_type="int")
        second = _assign_primary_keys(df, "pk", pk_type="int")
        assert list(first["pk"]) == [1, 2, 3]
        assert list(second["pk"]) == [1, 2, 3]

    def test_int_pk_resets_per_table_in_orchestrator(
        self,
        accounts_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
    ) -> None:
        """Two PK-bearing tables in one ``synthesize_tables`` call each get
        their own ``[1..n]`` sequence — IDs are not globally monotonic."""
        # Give transactions its own PK so both tables go through assignment.
        tx_df = transactions_df.copy()
        tx_df.insert(0, "tx_id", range(len(tx_df)))

        acct = Table(accounts_df, primary_key="account_id", pk_type="int")
        tx = Table(
            tx_df,
            primary_key="tx_id",
            pk_type="int",
            foreign_keys={"account_id": acct},
        )

        original = dataxid.api_key
        dataxid.api_key = "dx_test_isolated"
        try:
            with _mock_synthesize_stack():
                result = dataxid.synthesize_tables({"accounts": acct, "transactions": tx})
        finally:
            dataxid.api_key = original

        assert list(result["accounts"]["account_id"]) == [1, 2, 3]
        assert result["transactions"]["tx_id"].iloc[0] == 1


class TestRemapForeignKeysDeterminism:
    """Orphan FKs are reassigned by cycling parent PKs, deterministically."""

    def test_orphans_cycle_through_parent_pks(self) -> None:
        parent = pd.DataFrame({"pk": [1, 2]})
        child = pd.DataFrame({"fk": [1, 999, 888, 777, 666]})
        first = _remap_foreign_keys(child, "fk", parent, "pk")
        second = _remap_foreign_keys(child, "fk", parent, "pk")
        assert list(first["fk"]) == list(second["fk"])
        # First value already valid; orphans cycle [1, 2, 1, 2].
        assert list(first["fk"]) == [1, 1, 2, 1, 2]


# ---------------------------------------------------------------------------
# Top-level forwarding invariants
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestSynthesizeTablesForwarding:
    """``seed`` / ``config`` / ``api_key`` reach the per-table backends."""

    def test_seed_forwarded_to_root_synthesize(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """A user-supplied ``seed`` must reach ``synthesize`` for the root
        table — otherwise ``synthesize_tables`` could not produce reproducible
        outputs."""
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                seed=42,
            )

        assert synth_calls[0].get("seed") == 42

    def test_seed_forwarded_to_child_generate(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """The same ``seed`` must reach the child's ``model.generate`` call."""
        acct = Table(accounts_df, primary_key="account_id")
        generate_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            create_side_effect=_make_capturing_create(generate_calls=generate_calls)
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                seed=42,
            )

        assert generate_calls[0].get("seed") == 42

    def test_config_forwarded_to_create_and_synthesize(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """A ``ModelConfig`` instance reaches both backends so per-table
        training stays consistent with the user's intent."""
        cfg = ModelConfig(max_epochs=5)
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        create_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls),
            create_side_effect=_make_capturing_create(create_calls=create_calls),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                config=cfg,
            )

        assert synth_calls[0].get("config") is cfg
        assert create_calls[0].get("config") is cfg

    def test_api_key_override_forwarded(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """An explicit ``api_key=`` overrides ``dataxid.api_key`` for every
        downstream call."""
        acct = Table(accounts_df, primary_key="account_id")
        synth_calls: list[dict[str, Any]] = []
        create_calls: list[dict[str, Any]] = []

        with _mock_synthesize_stack(
            synthesize_side_effect=_make_capturing_synthesize(synth_calls),
            create_side_effect=_make_capturing_create(create_calls=create_calls),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                },
                api_key="sk_explicit_override",
            )

        assert synth_calls[0].get("api_key") == "sk_explicit_override"
        assert create_calls[0].get("api_key") == "sk_explicit_override"


# ---------------------------------------------------------------------------
# Resource cleanup invariants
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_api_key")
class TestModelCleanup:
    """Child models are deleted after each child table is generated."""

    def test_child_model_deleted_on_success(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """``Model.delete()`` runs after a successful ``generate`` so the
        server-side model is not orphaned."""
        acct = Table(accounts_df, primary_key="account_id")
        created_models: list[MagicMock] = []

        def _track_create(*args: Any, **kwargs: Any) -> MagicMock:
            m = _mock_model_create(*args, **kwargs)
            created_models.append(m)
            return m

        with _mock_synthesize_stack(create_side_effect=_track_create):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert len(created_models) == 1
        created_models[0].delete.assert_called_once()

    def test_child_model_deleted_on_generate_failure(
        self, accounts_df: pd.DataFrame, transactions_df: pd.DataFrame
    ) -> None:
        """If ``generate`` raises, the model is still deleted before the
        exception propagates — no leaks on the failure path."""
        acct = Table(accounts_df, primary_key="account_id")
        created_models: list[MagicMock] = []

        def _track_create(*args: Any, **kwargs: Any) -> MagicMock:
            m = _mock_model_create(*args, **kwargs)

            def _boom(**_: Any) -> None:
                raise RuntimeError("simulated generate failure")

            m.generate = _boom
            created_models.append(m)
            return m

        with (
            _mock_synthesize_stack(create_side_effect=_track_create),
            pytest.raises(RuntimeError, match="simulated generate failure"),
        ):
            dataxid.synthesize_tables(
                {
                    "accounts": acct,
                    "transactions": Table(
                        transactions_df, foreign_keys={"account_id": acct}
                    ),
                }
            )

        assert len(created_models) == 1
        created_models[0].delete.assert_called_once()
