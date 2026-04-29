# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Boundary validation tests for the public SDK surface.

Public entry points (``synthesize``, ``synthesize_tables``, ``Model.create``,
``Model.generate``, ``Model.impute``, ``Table.__post_init__``,
``DataxidClient.__init__``) reject malformed inputs *before* any side effect
runs — local analysis, HTTP calls, or training. Each test asserts:

* the failure surfaces as :class:`InvalidRequestError`,
* the ``param`` attribute names the offending field, and
* no HTTP request and no training pass were started.
"""

from collections.abc import Iterator
from unittest.mock import patch

import httpx
import pandas as pd
import pytest

import dataxid
from dataxid import Synthetic, Table
from dataxid.client._http import DataxidClient
from dataxid.exceptions import InvalidRequestError
from dataxid.training._model import Model


@pytest.fixture(autouse=True)
def _set_api_key() -> Iterator[None]:
    original = dataxid.api_key
    dataxid.api_key = "dx_test_boundary"
    yield
    dataxid.api_key = original


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "city": ["A", "B", "A", "C"],
    })


@pytest.fixture
def parent_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "region": ["N", "S", "E", "W"],
    })


@pytest.fixture
def no_side_effects() -> Iterator[tuple]:
    """Patch HTTP transport and training loop; assert neither was invoked.

    Boundary validation must reject bad inputs before any of these run, so
    a positive ``call_count`` is itself a regression.
    """
    with patch.object(DataxidClient, "_request") as mock_req, \
         patch("dataxid.training._model.train_frozen") as mock_train:
        yield mock_req, mock_train
        assert mock_req.call_count == 0, (
            "HTTP request was made before validation rejected the input"
        )
        assert mock_train.call_count == 0, (
            "Training started before validation rejected the input"
        )


# ---------------------------------------------------------------------------
# synthesize()
# ---------------------------------------------------------------------------

class TestSynthesizeBoundary:
    """``synthesize()`` rejects malformed scalar tunables up-front."""

    def test_seed_non_int_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="seed") as exc_info:
            dataxid.synthesize(data=sample_df, seed="42")  # type: ignore[arg-type]
        assert exc_info.value.param == "seed"

    def test_seed_bool_rejected(self, sample_df: pd.DataFrame) -> None:
        """``True`` is an ``int`` subclass; reject it explicitly so callers
        cannot accidentally seed the RNG with a boolean."""
        with pytest.raises(InvalidRequestError, match="seed") as exc_info:
            dataxid.synthesize(data=sample_df, seed=True)  # type: ignore[arg-type]
        assert exc_info.value.param == "seed"

    def test_diversity_zero_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="diversity") as exc_info:
            dataxid.synthesize(data=sample_df, diversity=0)
        assert exc_info.value.param == "diversity"

    def test_diversity_negative_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="diversity") as exc_info:
            dataxid.synthesize(data=sample_df, diversity=-0.5)
        assert exc_info.value.param == "diversity"

    def test_rare_cutoff_above_one_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="rare_cutoff") as exc_info:
            dataxid.synthesize(data=sample_df, rare_cutoff=1.5)
        assert exc_info.value.param == "rare_cutoff"

    def test_rare_cutoff_zero_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="rare_cutoff") as exc_info:
            dataxid.synthesize(data=sample_df, rare_cutoff=0)
        assert exc_info.value.param == "rare_cutoff"

    def test_rare_strategy_unknown_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="rare_strategy") as exc_info:
            dataxid.synthesize(
                data=sample_df, rare_strategy="masc",  # type: ignore[arg-type]
            )
        assert exc_info.value.param == "rare_strategy"


# ---------------------------------------------------------------------------
# synthesize_tables()
# ---------------------------------------------------------------------------

class TestSynthesizeTablesBoundary:
    """``synthesize_tables()`` rejects mis-shaped ``tables`` and per-table dicts."""

    def test_tables_not_a_dict_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="tables") as exc_info:
            dataxid.synthesize_tables(tables=[Table(sample_df)])  # type: ignore[arg-type]
        assert exc_info.value.param == "tables"

    def test_tables_value_not_a_table_rejected(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match=r"tables\['users'\] must be a Table"
        ) as exc_info:
            dataxid.synthesize_tables(
                tables={"users": sample_df},  # type: ignore[dict-item]
            )
        assert exc_info.value.param == "tables"

    def test_tables_empty_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="at least one table") as exc_info:
            dataxid.synthesize_tables(tables={})
        assert exc_info.value.param == "tables"

    def test_seed_non_int_rejected(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="seed") as exc_info:
            dataxid.synthesize_tables(
                tables={"t": Table(sample_df)}, seed="42",  # type: ignore[arg-type]
            )
        assert exc_info.value.param == "seed"

    def test_synthetic_unknown_table_rejected(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="synthetic refers to unknown table"
        ) as exc_info:
            dataxid.synthesize_tables(
                tables={"users": Table(sample_df)},
                synthetic={"accountss": Synthetic(diversity=0.9)},
            )
        assert exc_info.value.param == "synthetic"

    def test_synthetic_value_wrong_type_rejected(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="must be a Synthetic instance"
        ) as exc_info:
            dataxid.synthesize_tables(
                tables={"users": Table(sample_df)},
                synthetic={"users": {"diversity": 0.9}},  # type: ignore[dict-item]
            )
        assert exc_info.value.param == "synthetic"

    def test_distribution_value_wrong_type_rejected(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="must be a Distribution instance"
        ) as exc_info:
            dataxid.synthesize_tables(
                tables={"users": Table(sample_df)},
                distribution={"users": {"column": "city"}},  # type: ignore[dict-item]
            )
        assert exc_info.value.param == "distribution"

    def test_bias_value_wrong_type_rejected(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="must be a Bias instance"
        ) as exc_info:
            dataxid.synthesize_tables(
                tables={"users": Table(sample_df)},
                bias={"users": {"target": "city"}},  # type: ignore[dict-item]
            )
        assert exc_info.value.param == "bias"

    def test_conditions_value_wrong_type_rejected(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="must be a pandas DataFrame"
        ) as exc_info:
            dataxid.synthesize_tables(
                tables={"users": Table(sample_df)},
                conditions={"users": [{"city": "A"}]},  # type: ignore[dict-item]
            )
        assert exc_info.value.param == "conditions"


# ---------------------------------------------------------------------------
# Model.create()
# ---------------------------------------------------------------------------

class TestModelCreateBoundary:
    """``Model.create()`` rejects bad inputs before any side effect runs."""

    def test_data_not_dataframe_rejected(
        self, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="data") as exc_info:
            Model.create(data=[{"age": 25}])  # type: ignore[arg-type]
        assert exc_info.value.param == "data"

    def test_data_none_rejected(self, no_side_effects: tuple) -> None:
        with pytest.raises(InvalidRequestError, match="data") as exc_info:
            Model.create(data=None)  # type: ignore[arg-type]
        assert exc_info.value.param == "data"

    def test_n_samples_zero_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="n_samples") as exc_info:
            Model.create(data=sample_df, n_samples=0)
        assert exc_info.value.param == "n_samples"

    def test_n_samples_negative_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="n_samples") as exc_info:
            Model.create(data=sample_df, n_samples=-5)
        assert exc_info.value.param == "n_samples"

    def test_n_samples_bool_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="n_samples") as exc_info:
            Model.create(data=sample_df, n_samples=True)  # type: ignore[arg-type]
        assert exc_info.value.param == "n_samples"

    def test_parent_not_dataframe_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="parent") as exc_info:
            Model.create(data=sample_df, parent="not a df")  # type: ignore[arg-type]
        assert exc_info.value.param == "parent"

    def test_parent_encoding_types_not_dict_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="parent_encoding_types"
        ) as exc_info:
            Model.create(
                data=sample_df,
                parent_encoding_types=["region"],  # type: ignore[arg-type]
            )
        assert exc_info.value.param == "parent_encoding_types"

    def test_parent_encoding_types_unknown_value_rejected(
        self,
        sample_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        no_side_effects: tuple,
    ) -> None:
        """Bad value should fail at the boundary, not deep inside ``analyze()``."""
        with pytest.raises(InvalidRequestError, match="must be one of") as exc_info:
            Model.create(
                data=sample_df,
                parent=parent_df,
                parent_encoding_types={"region": "FANCY_TYPE"},
            )
        assert exc_info.value.param == "parent_encoding_types"

    def test_parent_encoding_types_non_string_key_rejected(
        self,
        sample_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        no_side_effects: tuple,
    ) -> None:
        with pytest.raises(
            InvalidRequestError, match="keys must be strings"
        ) as exc_info:
            Model.create(
                data=sample_df,
                parent=parent_df,
                parent_encoding_types={1: "AUTO"},  # type: ignore[dict-item]
            )
        assert exc_info.value.param == "parent_encoding_types"

    def test_parent_encoding_types_context_error_takes_priority(
        self,
        sample_df: pd.DataFrame,
        no_side_effects: tuple,
    ) -> None:
        """When ``parent`` is missing, the context error wins over content errors.

        Even with an invalid value, the user's real problem is the missing
        ``parent`` argument; surfacing the content failure first would send
        them down a debug detour.
        """
        with pytest.raises(InvalidRequestError, match="without parent") as exc_info:
            Model.create(
                data=sample_df,
                parent_encoding_types={"col": "FANCY_TYPE"},
            )
        assert exc_info.value.param == "parent_encoding_types"

    def test_foreign_key_non_string_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="foreign_key") as exc_info:
            Model.create(data=sample_df, foreign_key=42)  # type: ignore[arg-type]
        assert exc_info.value.param == "foreign_key"

    def test_parent_key_non_string_rejected(
        self, sample_df: pd.DataFrame, no_side_effects: tuple
    ) -> None:
        with pytest.raises(InvalidRequestError, match="parent_key") as exc_info:
            Model.create(data=sample_df, parent_key=42)  # type: ignore[arg-type]
        assert exc_info.value.param == "parent_key"


# ---------------------------------------------------------------------------
# Model.generate() / Model.impute()
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_model(
    sample_df: pd.DataFrame,
) -> Iterator[Model]:
    """Build a Model without exercising real network or training.

    Stubs ``DataxidClient._request`` and ``train_frozen`` for the duration
    of construction, so the resulting Model is suitable for boundary tests
    that should reject inputs before any further I/O.
    """
    def _fake_train(model: Model, **_kwargs: object) -> None:
        model.status = "ready"

    create_resp = {
        "data": {"id": "mdl_test", "status": "training", "config": {}},
    }
    with patch.object(
        DataxidClient, "_request", return_value=create_resp,
    ), patch("dataxid.training._model.train_frozen", _fake_train):
        model = Model.create(
            data=sample_df,
            config={"embedding_dim": 16, "model_size": "small", "max_epochs": 1},
        )
        yield model


class TestModelGenerateBoundary:
    """``Model.generate()`` rejects malformed kwargs before sending a request."""

    def test_n_samples_zero_rejected(self, trained_model: Model) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="n_samples") as exc_info:
                trained_model.generate(n_samples=0)
            assert mock_req.call_count == 0
        assert exc_info.value.param == "n_samples"

    def test_n_samples_negative_rejected(self, trained_model: Model) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="n_samples") as exc_info:
                trained_model.generate(n_samples=-1)
            assert mock_req.call_count == 0
        assert exc_info.value.param == "n_samples"

    def test_conditions_non_dataframe_rejected(self, trained_model: Model) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="conditions") as exc_info:
                trained_model.generate(
                    conditions={"city": ["A"]},  # type: ignore[arg-type]
                )
            assert mock_req.call_count == 0
        assert exc_info.value.param == "conditions"

    def test_parent_non_dataframe_rejected(self, trained_model: Model) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="parent") as exc_info:
                trained_model.generate(parent="oops")  # type: ignore[arg-type]
            assert mock_req.call_count == 0
        assert exc_info.value.param == "parent"


class TestModelImputeBoundary:
    """``Model.impute()`` rejects malformed kwargs before sending a request."""

    def test_X_non_dataframe_rejected(self, trained_model: Model) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="X must be") as exc_info:
                trained_model.impute(X={"a": [1]})  # type: ignore[arg-type]
            assert mock_req.call_count == 0
        assert exc_info.value.param == "X"

    def test_parent_non_dataframe_rejected(
        self, trained_model: Model, sample_df: pd.DataFrame
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="parent") as exc_info:
                trained_model.impute(X=sample_df, parent="oops")  # type: ignore[arg-type]
            assert mock_req.call_count == 0
        assert exc_info.value.param == "parent"

    def test_trials_zero_rejected(
        self, trained_model: Model, sample_df: pd.DataFrame
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="trials") as exc_info:
                trained_model.impute(X=sample_df, trials=0)
            assert mock_req.call_count == 0
        assert exc_info.value.param == "trials"

    def test_trials_negative_rejected(
        self, trained_model: Model, sample_df: pd.DataFrame
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="trials") as exc_info:
                trained_model.impute(X=sample_df, trials=-1)
            assert mock_req.call_count == 0
        assert exc_info.value.param == "trials"

    def test_trials_non_int_rejected(
        self, trained_model: Model, sample_df: pd.DataFrame
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            with pytest.raises(InvalidRequestError, match="trials") as exc_info:
                trained_model.impute(X=sample_df, trials=1.5)  # type: ignore[arg-type]
            assert mock_req.call_count == 0
        assert exc_info.value.param == "trials"


# ---------------------------------------------------------------------------
# Table — extra invariants beyond the existing test_synthesize_tables coverage
# ---------------------------------------------------------------------------

class TestTableExtraValidation:
    """``Table.__post_init__`` rejects nonsensical FK topologies."""

    def test_fk_equal_to_primary_key_rejected(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "region": ["N", "S"]})
        parent = Table(
            pd.DataFrame({"id": [1, 2]}),
            primary_key="id",
        )
        with pytest.raises(
            InvalidRequestError, match="cannot also be the primary_key"
        ) as exc_info:
            Table(df, primary_key="id", foreign_keys={"id": parent})
        assert exc_info.value.param == "foreign_keys"

    def test_self_reference_rejected(self) -> None:
        """A Table whose foreign_keys point back at itself would silently
        create a cycle. ``__post_init__`` rejects it explicitly so callers
        see a targeted error instead of a topological-sort cycle message."""
        df = pd.DataFrame({"id": [1, 2], "parent_id": [1, 1]})
        tbl = Table(df, primary_key="id")
        tbl.foreign_keys = {"parent_id": tbl}
        with pytest.raises(
            InvalidRequestError, match="self-references are not supported"
        ) as exc_info:
            tbl.__post_init__()
        assert exc_info.value.param == "foreign_keys"


# ---------------------------------------------------------------------------
# DataxidClient
# ---------------------------------------------------------------------------

class TestDataxidClientBoundary:
    """``DataxidClient.__init__`` enforces the secure defaults of the SDK."""

    def test_api_key_non_string_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="api_key") as exc_info:
            DataxidClient(api_key=42)  # type: ignore[arg-type]
        assert exc_info.value.param == "api_key"

    def test_base_url_non_string_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="base_url") as exc_info:
            DataxidClient(base_url=42)  # type: ignore[arg-type]
        assert exc_info.value.param == "base_url"

    def test_base_url_empty_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="non-empty") as exc_info:
            DataxidClient(base_url="   ")
        assert exc_info.value.param == "base_url"

    def test_base_url_https_accepted(self) -> None:
        DataxidClient(base_url="https://api.dataxid.com")

    def test_base_url_http_localhost_accepted(self) -> None:
        DataxidClient(base_url="http://localhost:8000")

    def test_base_url_http_127_0_0_1_accepted(self) -> None:
        DataxidClient(base_url="http://127.0.0.1:8000/v1")

    def test_base_url_http_non_localhost_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="plaintext") as exc_info:
            DataxidClient(base_url="http://api.dataxid.com")
        assert exc_info.value.param == "base_url"

    def test_base_url_no_scheme_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="must start with") as exc_info:
            DataxidClient(base_url="api.dataxid.com")
        assert exc_info.value.param == "base_url"

    def test_base_url_unknown_scheme_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="scheme must be") as exc_info:
            DataxidClient(base_url="ftp://api.dataxid.com")
        assert exc_info.value.param == "base_url"

    def test_timeout_wrong_type_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="timeout") as exc_info:
            DataxidClient(timeout="30")  # type: ignore[arg-type]
        assert exc_info.value.param == "timeout"

    def test_timeout_bool_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="timeout") as exc_info:
            DataxidClient(timeout=True)  # type: ignore[arg-type]
        assert exc_info.value.param == "timeout"

    def test_timeout_httpx_object_accepted(self) -> None:
        DataxidClient(timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0))

    def test_global_base_url_revalidated_on_property_access(self) -> None:
        """Setting ``dataxid.base_url`` to an insecure value must surface as
        an error when the client is actually used, not silently leak the API
        key over plaintext HTTP."""
        original = dataxid.base_url
        try:
            dataxid.base_url = "http://api.dataxid.com"
            client = DataxidClient()
            with pytest.raises(InvalidRequestError, match="plaintext") as exc_info:
                _ = client.base_url
            assert exc_info.value.param == "base_url"
        finally:
            dataxid.base_url = original
