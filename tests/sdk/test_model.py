# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end integration tests for the :class:`dataxid.Model` orchestration layer.

These tests stub out the HTTP transport (``DataxidClient._request``) and the
local frozen-training loop, then exercise the public model lifecycle:

* :meth:`Model.create` — analyze + ``POST /v1/models`` + frozen training
* :meth:`Model.generate` — local encode + ``POST /v1/models/{id}/generate``
* :meth:`Model.delete` — ``DELETE /v1/models/{id}``
* :meth:`Model.refresh` — ``GET /v1/models/{id}``
* :meth:`Model.impute` — generate-driven imputation
* :func:`dataxid.synthesize` — one-shot create+generate+delete convenience

The goal is to pin the *contract* between the SDK orchestration layer and the
HTTP client (request shape, sequencing, error propagation) without exercising
real network I/O.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import dataxid
from dataxid.client._http import DataxidClient
from dataxid.client._serialization import deserialize_embedding
from dataxid.encoder._wrapper import Encoder
from dataxid.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
)
from dataxid.training._config import Distribution, ModelConfig, Synthetic
from dataxid.training._model import Model

MODEL_ID = "mdl_test_abc123"
EMBEDDING_DIM = 16
N_ROWS = 8

SEQ_N_ENTITIES = 3
SEQ_LEN_MAX = 3

_DEFAULT_CONFIG: dict[str, Any] = {
    "embedding_dim": EMBEDDING_DIM,
    "model_size": "small",
    "max_epochs": 1,
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
    })


@pytest.fixture
def ctx_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["North", "South", "North", "East", "South", "North", "East", "South"],
        "population": [100, 200, 100, 300, 200, 100, 300, 200],
    })


@pytest.fixture
def seq_df() -> pd.DataFrame:
    return pd.DataFrame({
        "account_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0, 50.0, 75.0, 60.0],
        "type": ["credit", "debit", "credit", "debit", "credit", "credit", "debit", "credit"],
    })


@pytest.fixture
def seq_ctx_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3],
        "region": ["North", "South", "East"],
    })


@pytest.fixture(autouse=True)
def _set_api_key() -> Iterator[None]:
    """Ensure every test runs with a deterministic API key, restored after."""
    original = dataxid.api_key
    dataxid.api_key = "dx_test_integration"
    yield
    dataxid.api_key = original


@pytest.fixture
def mock_train_frozen() -> Iterator[None]:
    """Replace the local training loop with a deterministic 'ready' stub."""
    def _fake_train(model: Model, **_kwargs: Any) -> None:
        model.train_losses = [0.5]
        model.val_losses = [0.4]
        model.stopped_early = False
        model.status = "ready"

    with patch("dataxid.training._model.train_frozen", _fake_train):
        yield


def _make_create_response() -> dict[str, Any]:
    return {
        "data": {
            "id": MODEL_ID,
            "status": "training",
            "config": {"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
        }
    }


def _make_generate_response(n_samples: int = N_ROWS) -> dict[str, Any]:
    return {
        "data": {
            "n_rows": n_samples,
            "n_cols": 2,
            "columns": ["feat:/age__cat", "feat:/city__cat"],
            "codes": {
                "feat:/age__cat": list(range(n_samples)),
                "feat:/city__cat": [0] * n_samples,
            },
        }
    }


def _make_sequential_generate_response() -> dict[str, Any]:
    """Mock sequential codes: ``{sub_col: [[entity0_vals], [entity1_vals], ...]}``."""
    from dataxid.encoder._nn import RIDX_PREFIX, SIDX_PREFIX, SLEN_PREFIX
    return {
        "data": {
            "n_rows": SEQ_N_ENTITIES,
            "n_cols": 4,
            "columns": [
                f"{SIDX_PREFIX}cat", f"{SLEN_PREFIX}cat", f"{RIDX_PREFIX}cat",
                "feat:/amount__cat", "feat:/type__cat",
            ],
            "codes": {
                f"{SIDX_PREFIX}cat": [[0, 1, 2], [0, 1, 0], [0, 1, 2]],
                f"{SLEN_PREFIX}cat": [[3, 3, 3], [2, 2, 0], [3, 3, 3]],
                f"{RIDX_PREFIX}cat": [[3, 2, 1], [2, 1, 0], [3, 2, 1]],
                "feat:/amount__cat": [[1, 2, 1], [3, 2, 0], [0, 1, 0]],
                "feat:/type__cat": [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
            },
        }
    }


def _flat_decode(
    raw_codes: dict[str, Any],
    features: Any,
    column_stats: Any,
    **_kwargs: Any,
) -> pd.DataFrame:
    """Decode stub for flat (non-sequential) generate responses."""
    n = len(next(iter(raw_codes.values())))
    return pd.DataFrame({"age": list(range(n)), "city": ["A"] * n})


def _seq_decode(
    raw_codes: dict[str, Any],
    features: Any,
    column_stats: Any,
    **_kwargs: Any,
) -> pd.DataFrame:
    """Decode stub for sequential generate responses."""
    n = len(next(iter(raw_codes.values())))
    return pd.DataFrame({"amount": list(range(n)), "type": ["credit"] * n})


class _Driver:
    """Single source of truth for the create-then-do-something setup pattern.

    Centralises the boilerplate of patching ``DataxidClient._request`` and
    ``decode_columns`` for a sequence of HTTP responses, so individual tests
    only express the *intent* (which generate / delete / impute payloads they
    expect) and the assertions.
    """

    def __init__(self, mock_req: Any, decode_stub: Any = _flat_decode) -> None:
        self.mock_req = mock_req
        self.decode_stub = decode_stub

    def queue(self, *responses: Any) -> None:
        """Set the side-effect queue for ``DataxidClient._request``."""
        self.mock_req.side_effect = list(responses)

    def call_args(self, index: int) -> Any:
        return self.mock_req.call_args_list[index]

    def json_at(self, index: int) -> dict[str, Any]:
        return self.mock_req.call_args_list[index][1]["json"]

    def last_json(self) -> dict[str, Any]:
        return self.mock_req.call_args_list[-1][1]["json"]


class TestModelCreate:
    """Model.create() returns a ready model and POSTs the expected payload."""

    def test_returns_ready_model_with_id(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        assert model.id == MODEL_ID
        assert model.status == "ready"

    def test_first_call_posts_to_models_endpoint(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        first_call = mock_req.call_args_list[0]
        assert first_call[0][0] == "POST"
        assert first_call[0][1] == "/v1/models"

    def test_metadata_payload_contains_features(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        body = mock_req.call_args_list[0][1]["json"]
        assert "metadata" in body
        assert body["metadata"]["features"] == ["age", "city"]

    def test_accepts_modelconfig_instance(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        cfg = ModelConfig(embedding_dim=EMBEDDING_DIM, model_size="small", max_epochs=1)
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=cfg)
        assert model.id == MODEL_ID

    def test_modelconfig_and_dict_produce_identical_payload(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        """Dict and ModelConfig inputs must serialize to the same wire format."""
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_dict:
            Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        dict_body = mock_dict.call_args_list[0][1]["json"]

        cfg = ModelConfig(embedding_dim=EMBEDDING_DIM, model_size="small", max_epochs=1)
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_cfg:
            Model.create(data=sample_df, config=cfg)
        cfg_body = mock_cfg.call_args_list[0][1]["json"]

        assert dict_body["config"] == cfg_body["config"]


class TestGenerate:
    """Model.generate() returns decoded rows and forwards the right payload."""

    def test_returns_dataframe_of_expected_shape(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            result = model.generate()
        assert len(result) == N_ROWS
        assert list(result.columns) == ["age", "city"]
        assert model.status == "ready"

    def test_default_generation_sends_zero_embedding(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate()
        body = d.json_at(1)
        assert isinstance(body["embedding"], dict)
        assert body["embedding"]["shape"] == [N_ROWS, EMBEDDING_DIM]
        emb = deserialize_embedding(body["embedding"])
        assert torch.all(emb == 0), "Default generation must use zero embedding"

    def test_conditional_generation_keeps_zero_embedding(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        """Conditioning is via fixed_values, not the embedding tensor."""
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(conditions=sample_df)
        body = d.json_at(1)
        assert body["embedding"]["shape"] == [N_ROWS, EMBEDDING_DIM]
        emb = deserialize_embedding(body["embedding"])
        assert torch.all(emb == 0)

    def test_seed_forwarded_when_set(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(seed=42)
        assert d.json_at(1)["seed"] == 42

    def test_seed_omitted_when_none(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate()
        assert "seed" not in d.json_at(1)

    def test_distribution_dict_rejected(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        """Plain dict for ``distribution=`` is a typed-API mistake; raise early."""
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            mock_req.side_effect = [_make_create_response()]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            with pytest.raises(TypeError, match="Distribution instance"):
                model.generate(distribution={"column": "city", "probabilities": {"A": 1.0}})

    def test_distribution_sets_fixed_probs_and_column_order(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dist = Distribution(column="city", probabilities={"A": 0.8, "B": 0.2})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(distribution=dist)
        body = d.json_at(1)
        assert "fixed_probs" in body
        assert body["column_order"][0] == "feat:/city"

    def test_no_distribution_omits_column_order(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate()
        assert "column_order" not in d.json_at(1)

    def test_conditions_send_fixed_values_and_column_order(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        conds = pd.DataFrame({"city": ["A", "B"]})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode), \
             patch.object(Encoder, "_generation_embedding",
                          return_value=torch.zeros(2, EMBEDDING_DIM)):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(2))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(conditions=conds)
        body = d.json_at(1)
        assert isinstance(body["fixed_values"], dict)
        assert body["column_order"][0] == "feat:/city"

    def test_conditions_drive_n_samples(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        conds = pd.DataFrame({"city": ["A", "B", "C"]})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode), \
             patch.object(Encoder, "_generation_embedding",
                          return_value=torch.zeros(3, EMBEDDING_DIM)) as mock_emb:
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(3))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(conditions=conds)
        mock_emb.assert_called_once()
        assert mock_emb.call_args[1]["n_samples"] == 3

    def test_conditions_conflicting_n_samples_raises(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        conds = pd.DataFrame({"city": ["A", "B"]})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(2)]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            with pytest.raises(ValueError, match="conflicts with conditions length"):
                model.generate(conditions=conds, n_samples=99)

    def test_conditions_with_matching_n_samples(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        conds = pd.DataFrame({"city": ["A", "B"]})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode), \
             patch.object(Encoder, "_generation_embedding",
                          return_value=torch.zeros(2, EMBEDDING_DIM)):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(2))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            result = model.generate(conditions=conds, n_samples=2)
        assert len(result) == 2

    def test_conditions_combined_with_distribution(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        conds = pd.DataFrame({"city": ["A", "B"]})
        dist = Distribution(column="city", probabilities={"A": 0.5, "B": 0.5})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode), \
             patch.object(Encoder, "_generation_embedding",
                          return_value=torch.zeros(2, EMBEDDING_DIM)):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(2))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(conditions=conds, distribution=dist)
        body = d.json_at(1)
        assert "fixed_values" in body
        assert body["column_order"][0] == "feat:/city"

    def test_no_conditions_omits_fixed_values(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate()
        assert "fixed_values" not in d.json_at(1)


class TestDelete:
    """Model.delete() issues DELETE and updates local status."""

    def test_calls_delete_with_model_id(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            mock_req.side_effect = [_make_create_response(), None]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.delete()
        delete_call = mock_req.call_args_list[1]
        assert delete_call[0][0] == "DELETE"
        assert MODEL_ID in delete_call[0][1]
        assert model.status == "deleted"


class TestRefresh:
    """Model.refresh() issues GET and updates local status from API."""

    def test_get_request_updates_status(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                {"data": {"id": MODEL_ID, "status": "ready", "epoch": 1}},
            ]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            data = model.refresh()
        refresh_call = mock_req.call_args_list[1]
        assert refresh_call[0][0] == "GET"
        assert model.status == "ready"
        assert data["id"] == MODEL_ID


class TestSynthesize:
    """``dataxid.synthesize()`` chains create → generate → delete."""

    def test_returns_dataframe_of_requested_size(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            mock_req.side_effect = [
                _make_create_response(),
                _make_generate_response(N_ROWS),
                None,  # delete
            ]
            result = dataxid.synthesize(
                data=sample_df, n_samples=N_ROWS, config=_DEFAULT_CONFIG,
            )
        assert len(result) == N_ROWS

    def test_delete_called_after_generate(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            mock_req.side_effect = [
                _make_create_response(),
                _make_generate_response(N_ROWS),
                None,
            ]
            dataxid.synthesize(
                data=sample_df, n_samples=N_ROWS, config=_DEFAULT_CONFIG,
            )
        assert mock_req.call_args_list[-1][0][0] == "DELETE"

    def test_synthesize_with_parent(
        self,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        """Parent table is forwarded through the convenience wrapper."""
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            mock_req.side_effect = [
                _make_create_response(),
                _make_generate_response(N_ROWS),
                None,
            ]
            result = dataxid.synthesize(
                data=sample_df, n_samples=N_ROWS, parent=ctx_df, config=_DEFAULT_CONFIG,
            )
        assert len(result) == N_ROWS

    def test_delete_called_when_generate_fails(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        """Synthesize must clean up the temporary model even on failure."""
        with patch.object(DataxidClient, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                Exception("generate failed"),
                None,
            ]
            with pytest.raises(Exception, match="generate failed"):
                dataxid.synthesize(
                    data=sample_df, n_samples=N_ROWS, config=_DEFAULT_CONFIG,
                )
        delete_calls = [c for c in mock_req.call_args_list if c[0][0] == "DELETE"]
        assert len(delete_calls) == 1


class TestStatusTransitions:
    """The full create → generate → delete lifecycle keeps status coherent."""

    def test_full_lifecycle(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            mock_req.side_effect = [
                _make_create_response(),
                _make_generate_response(N_ROWS),
                None,  # delete
            ]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            assert model.status == "ready"
            model.generate()
            assert model.status == "ready"
            model.delete()
            assert model.status == "deleted"


class TestModelCreateWithContext:
    """Flat multitable: ``parent=`` carries context but does not enable sequential mode."""

    def test_returns_model_with_parent(
        self,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, parent=ctx_df, config=_DEFAULT_CONFIG)
        assert model._parent is not None

    def test_metadata_marks_has_context(
        self,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            Model.create(data=sample_df, parent=ctx_df, config=_DEFAULT_CONFIG)
        assert mock_req.call_args_list[0][1]["json"]["metadata"]["has_context"] is True

    def test_metadata_cardinality_count_unchanged_by_context(
        self,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        """Adding context must not leak the parent's cardinalities into the body."""
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            Model.create(data=sample_df, parent=ctx_df, config=_DEFAULT_CONFIG)
        body = mock_req.call_args_list[0][1]["json"]
        no_ctx = Encoder(embedding_dim=EMBEDDING_DIM, model_size="small")
        meta_flat = no_ctx.analyze(sample_df)
        assert len(body["metadata"]["cardinalities"]) == len(meta_flat["cardinalities"])


class TestGenerateWithContext:
    """Generate honours both the training-time and call-time context tables."""

    def test_generate_uses_training_context(
        self,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, parent=ctx_df, config=_DEFAULT_CONFIG)
            result = model.generate()
        assert len(result) == N_ROWS

    def test_generate_with_explicit_context(
        self,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, parent=ctx_df, config=_DEFAULT_CONFIG)
            result = model.generate(parent=ctx_df)
        assert len(result) == N_ROWS


class TestSequentialModelCreate:
    """Sequential (time series) mode adds entity / sequence-length metadata."""

    def test_returns_sequential_model(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
        assert model.is_sequential is True

    def test_metadata_carries_sequence_lengths(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
        meta = mock_req.call_args_list[0][1]["json"]["metadata"]
        assert meta["is_sequential"] is True
        assert meta["seq_len_max"] == 3
        assert meta["seq_len_median"] >= 2

    def test_positional_cardinalities_in_metadata(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        from dataxid.encoder._nn import SIDX_PREFIX
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
        cards = mock_req.call_args_list[0][1]["json"]["metadata"]["cardinalities"]
        assert any(k.startswith(SIDX_PREFIX) for k in cards)


class TestSequentialGenerate:
    def test_payload_carries_sequential_flag(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_seq_decode):
            d = _Driver(mock_req, decode_stub=_seq_decode)
            d.queue(_make_create_response(), _make_sequential_generate_response())
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
            model.generate()
        body = d.json_at(1)
        assert body["is_sequential"] is True
        assert body["seq_len_max"] == SEQ_LEN_MAX

    def test_returns_flattened_dataframe(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_seq_decode):
            d = _Driver(mock_req, decode_stub=_seq_decode)
            d.queue(_make_create_response(), _make_sequential_generate_response())
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
            result = model.generate()
        assert "account_id" in result.columns
        assert len(result) == 8

    def test_conditions_send_2d_fixed_values(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        """Sequential conditions (long-format) → 2D fixed_values in the payload."""
        conds = pd.DataFrame({
            "account_id": [1, 1, 2, 3],
            "type": ["debit", "credit", "debit", "credit"],
        })
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_seq_decode):
            d = _Driver(mock_req, decode_stub=_seq_decode)
            d.queue(_make_create_response(), _make_sequential_generate_response())
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
            model.generate(conditions=conds)
        fv = d.json_at(1)["fixed_values"]
        assert isinstance(fv, dict)
        for vals in fv.values():
            assert all(isinstance(row, list) for row in vals), \
                "Sequential fixed_values must be 2D (list of lists)"
            assert len(vals) == SEQ_N_ENTITIES

    def test_conditions_send_column_order(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        conds = pd.DataFrame({
            "account_id": [1, 2, 3],
            "type": ["debit", "debit", "debit"],
        })
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_seq_decode):
            d = _Driver(mock_req, decode_stub=_seq_decode)
            d.queue(_make_create_response(), _make_sequential_generate_response())
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
            model.generate(conditions=conds)
        assert d.json_at(1)["column_order"][0] == "feat:/type"

    def test_conditions_missing_foreign_key_raises(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        conds = pd.DataFrame({"type": ["debit", "credit"]})
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_seq_decode):
            mock_req.side_effect = [
                _make_create_response(),
                _make_sequential_generate_response(),
            ]
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
            with pytest.raises(ValueError, match="foreign key column"):
                model.generate(conditions=conds)

    def test_no_conditions_omits_fixed_values(
        self,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
        mock_train_frozen: None,
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_seq_decode):
            d = _Driver(mock_req, decode_stub=_seq_decode)
            d.queue(_make_create_response(), _make_sequential_generate_response())
            model = Model.create(
                data=seq_df, parent=seq_ctx_df,
                foreign_key="account_id", parent_key="id",
                config=_DEFAULT_CONFIG,
            )
            model.generate()
        assert "fixed_values" not in d.json_at(1)


class TestBackwardCompatModel:
    """Flat mode is unaffected by sequential additions."""

    def test_flat_create_metadata_not_sequential(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()) as mock_req:
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        body = mock_req.call_args_list[0][1]["json"]
        assert body["metadata"]["is_sequential"] is False
        assert model.is_sequential is False

    def test_flat_generate_omits_sequential_flag(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode):
            d = _Driver(mock_req)
            d.queue(_make_create_response(), _make_generate_response(N_ROWS))
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate()
        assert "is_sequential" not in d.json_at(1)


class TestInputValidation:
    """Invalid context/sequential parameter combinations raise InvalidRequestError early."""

    def test_parent_encoding_types_without_parent(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(InvalidRequestError, match="without parent"):
            Model.create(
                data=sample_df,
                parent_encoding_types={"col": "categorical"},
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )

    def test_parent_key_without_parent(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="without parent"):
            Model.create(
                data=sample_df,
                parent_key="id",
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )

    def test_foreign_key_not_in_data(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(InvalidRequestError, match="not found in data"):
            Model.create(
                data=sample_df,
                foreign_key="nonexistent_col",
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )

    def test_parent_key_not_in_parent(self, sample_df: pd.DataFrame) -> None:
        ctx = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]})
        with pytest.raises(InvalidRequestError, match="not found in parent"):
            Model.create(
                data=sample_df,
                parent=ctx,
                parent_key="missing_pk",
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )

    def test_parent_row_count_mismatch_flat(self, sample_df: pd.DataFrame) -> None:
        ctx = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(InvalidRequestError, match="Row count mismatch"):
            Model.create(
                data=sample_df,
                parent=ctx,
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )

    def test_sequential_without_parent(self) -> None:
        df = pd.DataFrame({
            "entity_id": [1, 1, 2, 2, 3, 3],
            "value": [10, 20, 30, 40, 50, 60],
        })
        with pytest.raises(InvalidRequestError, match="Sequential mode requires parent"):
            Model.create(
                data=df,
                foreign_key="entity_id",
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )

    def test_invalid_request_error_carries_param_attr(
        self, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(InvalidRequestError) as exc_info:
            Model.create(
                data=sample_df,
                parent_encoding_types={"col": "categorical"},
                config={"embedding_dim": EMBEDDING_DIM, "model_size": "small"},
            )
        assert exc_info.value.param == "parent_encoding_types"

    def test_valid_flat_context_passes(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        ctx = pd.DataFrame({"pop": [100] * 8})
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, parent=ctx, config=_DEFAULT_CONFIG)
        assert model.has_context is True


class TestGenerationSamplingParams:
    """Generation knobs (diversity / rare_cutoff / rare_strategy) — validation + payload forwarding."""

    def _train_and_generate(
        self, sample_df: pd.DataFrame, **generate_kwargs: Any
    ) -> dict[str, Any]:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode), \
             patch("dataxid.training._model.train_frozen",
                   lambda model, **_: setattr(model, "status", "ready")):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(**generate_kwargs)
        return mock_req.call_args_list[1][1]["json"]

    @pytest.mark.parametrize("value", [0, -0.1, -1.0])
    def test_diversity_non_positive_rejected(
        self, sample_df: pd.DataFrame, value: float, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        with pytest.raises(ValueError, match="diversity"):
            model.generate(diversity=value)

    @pytest.mark.parametrize("value", [0, -0.1, 1.01, 2.0])
    def test_rare_cutoff_out_of_range_rejected(
        self, sample_df: pd.DataFrame, value: float, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        with pytest.raises(ValueError, match="rare_cutoff"):
            model.generate(rare_cutoff=value)

    def test_invalid_rare_strategy_rejected(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        with pytest.raises(ValueError, match="rare_strategy"):
            model.generate(rare_strategy="invalid")  # type: ignore[arg-type]

    def test_defaults_omit_diversity_and_rare_cutoff(
        self, sample_df: pd.DataFrame
    ) -> None:
        body = self._train_and_generate(sample_df)
        assert "diversity" not in body
        assert "rare_cutoff" not in body

    def test_non_default_diversity_sent(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(sample_df, diversity=0.5)
        assert body["diversity"] == 0.5

    def test_non_default_rare_cutoff_sent(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(sample_df, rare_cutoff=0.9)
        assert body["rare_cutoff"] == 0.9

    def test_both_non_default_sent(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(sample_df, diversity=0.7, rare_cutoff=0.8)
        assert body["diversity"] == 0.7
        assert body["rare_cutoff"] == 0.8

    @pytest.mark.parametrize("strategy", ["mask", "sample"])
    def test_rare_strategy_forwarded_to_decode(
        self, sample_df: pd.DataFrame, strategy: str
    ) -> None:
        """Regression guard: client and server must agree on whether
        ``<protected>`` is visible in the output."""
        seen: dict[str, Any] = {}

        def _spy_decode(raw_codes: Any, features: Any, column_stats: Any, **kwargs: Any) -> pd.DataFrame:
            seen.update(kwargs)
            return _flat_decode(raw_codes, features, column_stats)

        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_spy_decode), \
             patch("dataxid.training._model.train_frozen",
                   lambda model, **_: setattr(model, "status", "ready")):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(rare_strategy=strategy)

        assert seen.get("rare_strategy") == strategy

    def test_rare_strategy_default_inherits_from_privacy(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Omitted ``rare_strategy=`` falls through to ``ModelConfig.privacy.rare_strategy``."""
        seen: dict[str, Any] = {}

        def _spy_decode(raw_codes: Any, features: Any, column_stats: Any, **kwargs: Any) -> pd.DataFrame:
            seen.update(kwargs)
            return _flat_decode(raw_codes, features, column_stats)

        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_spy_decode), \
             patch("dataxid.training._model.train_frozen",
                   lambda model, **_: setattr(model, "status", "ready")):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = Model.create(
                data=sample_df,
                config={**_DEFAULT_CONFIG, "privacy": {"rare_strategy": "sample"}},
            )
            model.generate()

        assert seen.get("rare_strategy") == "sample"

    def test_rare_strategy_explicit_overrides_privacy(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Explicit ``rare_strategy=`` on ``generate()`` wins over Privacy's setting."""
        seen: dict[str, Any] = {}

        def _spy_decode(raw_codes: Any, features: Any, column_stats: Any, **kwargs: Any) -> pd.DataFrame:
            seen.update(kwargs)
            return _flat_decode(raw_codes, features, column_stats)

        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_spy_decode), \
             patch("dataxid.training._model.train_frozen",
                   lambda model, **_: setattr(model, "status", "ready")):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = Model.create(
                data=sample_df,
                config={**_DEFAULT_CONFIG, "privacy": {"rare_strategy": "sample"}},
            )
            model.generate(rare_strategy="mask")

        assert seen.get("rare_strategy") == "mask"


class TestGenerateWithSynthetic:
    """``Synthetic`` preset override semantics.

    Rule: explicit keyword > ``synthetic.<field>`` > ``generate()`` default. A
    preset is never silently overridden by a default — only by an explicit
    keyword argument.
    """

    def _train_and_generate(
        self, sample_df: pd.DataFrame, **generate_kwargs: Any
    ) -> dict[str, Any]:
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_flat_decode), \
             patch("dataxid.training._model.train_frozen",
                   lambda model, **_: setattr(model, "status", "ready")):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(**generate_kwargs)
        return mock_req.call_args_list[1][1]["json"]

    def _train_and_generate_with_decode_spy(
        self, sample_df: pd.DataFrame, **generate_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seen: dict[str, Any] = {}

        def _spy_decode(raw_codes: Any, features: Any, column_stats: Any, **kwargs: Any) -> pd.DataFrame:
            seen.update(kwargs)
            return _flat_decode(raw_codes, features, column_stats)

        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_spy_decode), \
             patch("dataxid.training._model.train_frozen",
                   lambda model, **_: setattr(model, "status", "ready")):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            model.generate(**generate_kwargs)
        return mock_req.call_args_list[1][1]["json"], seen

    def test_preset_scalar_fields_forwarded(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(
            sample_df, synthetic=Synthetic(diversity=0.7, rare_cutoff=0.9),
        )
        assert body["diversity"] == 0.7
        assert body["rare_cutoff"] == 0.9

    def test_preset_seed_forwarded_to_payload(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(sample_df, synthetic=Synthetic(seed=123))
        assert body.get("seed") == 123

    def test_preset_rare_strategy_reaches_decode(
        self, sample_df: pd.DataFrame
    ) -> None:
        _, decode_kwargs = self._train_and_generate_with_decode_spy(
            sample_df, synthetic=Synthetic(rare_strategy="mask"),
        )
        assert decode_kwargs.get("rare_strategy") == "mask"

    def test_kwarg_overrides_preset(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(
            sample_df, synthetic=Synthetic(diversity=0.5), diversity=0.9,
        )
        assert body["diversity"] == 0.9

    def test_preset_alone_leaves_unset_kwargs_at_defaults(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Fields missing from the preset must not appear in the payload."""
        body = self._train_and_generate(sample_df, synthetic=Synthetic(diversity=0.5))
        assert body["diversity"] == 0.5
        assert "rare_cutoff" not in body

    def test_preset_n_sets_n_samples(self, sample_df: pd.DataFrame) -> None:
        body = self._train_and_generate(sample_df, synthetic=Synthetic(n=5))
        assert body["embedding"]["shape"][0] == 5

    def test_explicit_n_samples_matching_preset_is_ok(
        self, sample_df: pd.DataFrame
    ) -> None:
        body = self._train_and_generate(
            sample_df, n_samples=5, synthetic=Synthetic(n=5),
        )
        assert body["embedding"]["shape"][0] == 5

    def test_explicit_n_samples_conflicting_preset_raises(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        with pytest.raises(ValueError, match="conflicts with synthetic.n"):
            model.generate(n_samples=5, synthetic=Synthetic(n=10))

    def test_non_synthetic_preset_rejected(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request", return_value=_make_create_response()):
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        with pytest.raises(TypeError, match="synthetic must be a Synthetic instance"):
            model.generate(synthetic={"diversity": 0.5})  # type: ignore[arg-type]


class TestImpute:
    """Model.impute() drives generate-based imputation, then merges with the input."""

    @staticmethod
    def _impute_decode(
        raw_codes: dict[str, Any],
        features: Any,
        column_stats: Any,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        n = len(next(iter(raw_codes.values())))
        return pd.DataFrame({"age": [99] * n, "city": ["Z"] * n})

    def _create_model(self, sample_df: pd.DataFrame, mock_req: Any) -> Model:
        mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
        return Model.create(data=sample_df, config=_DEFAULT_CONFIG)

    def test_returns_dataframe(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            mock_req.side_effect = [_make_create_response(), _make_generate_response(N_ROWS)]
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)]
            result = model.impute(dirty)
        assert len(result) == N_ROWS

    def test_imputation_columns_sent_in_payload(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)]
            model.impute(dirty)
        body = mock_req.call_args_list[-1][1]["json"]
        assert set(body["imputation_columns"]) == {"age"}

    def test_fixed_probs_include_imputation_suppression(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "city"] = None
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)]
            model.impute(dirty)
        assert "fixed_probs" in mock_req.call_args_list[-1][1]["json"]

    def test_payload_includes_column_order(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)]
            model.impute(dirty)
        assert "column_order" in mock_req.call_args_list[-1][1]["json"]

    def test_seed_merge_preserves_non_null_values(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        """Non-null values in the input survive the imputation pass."""
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None  # row 0 NULL → must be filled
        # row 1 age=30 must be preserved even when generate emits 99
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)]
            result = model.impute(dirty)
        assert result.loc[1, "age"] == 30
        assert result.loc[0, "age"] == 99

    def test_multi_draw_calls_generate_n_times(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)] * 3
            model.impute(dirty, trials=3)
        post_create = mock_req.call_args_list[1:]
        assert len(post_create) == 3

    def test_pick_mode_returns_modal_value(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None

        call_count = [0]

        def _decode_varied(
            raw_codes: dict[str, Any], features: Any, column_stats: Any, **_kwargs: Any
        ) -> pd.DataFrame:
            n = len(next(iter(raw_codes.values())))
            call_count[0] += 1
            age_val = 10 if call_count[0] <= 2 else 20  # mode = 10
            return pd.DataFrame({"age": [age_val] * n, "city": ["Z"] * n})

        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_decode_varied):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)] * 3
            result = model.impute(dirty, trials=3, pick="mode")
        assert result.loc[0, "age"] == 10

    def test_pick_all_returns_full_draw_list(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None

        call_count = [0]

        def _decode_seq(
            raw_codes: dict[str, Any], features: Any, column_stats: Any, **_kwargs: Any
        ) -> pd.DataFrame:
            n = len(next(iter(raw_codes.values())))
            call_count[0] += 1
            return pd.DataFrame({"age": [call_count[0] * 10] * n, "city": ["Z"] * n})

        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=_decode_seq):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)] * 2
            result = model.impute(dirty, trials=2, pick="all")
        cell = result.loc[0, "age"]
        assert isinstance(cell, (list, np.ndarray))

    def test_seed_kwarg_forwarded_to_payload(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        dirty = sample_df.copy()
        dirty.loc[0, "age"] = None
        with patch.object(DataxidClient, "_request") as mock_req, \
             patch("dataxid.training._model.decode_columns", side_effect=self._impute_decode):
            model = self._create_model(sample_df, mock_req)
            mock_req.side_effect = [_make_generate_response(N_ROWS)]
            model.impute(dirty, seed=42)
        assert mock_req.call_args_list[-1][1]["json"]["seed"] == 42


class TestErrorPropagation:
    """The orchestration layer must let HTTP errors bubble up unmodified.

    The full HTTP-status → exception-class mapping is exercised in
    ``test_client.py``; here we only assert that ``Model`` does not catch,
    transform, or swallow these exceptions on the user-facing path.
    """

    def test_create_propagates_authentication_error(
        self, sample_df: pd.DataFrame
    ) -> None:
        with patch.object(
            DataxidClient, "_request",
            side_effect=AuthenticationError("Invalid API key"),
        ):
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                Model.create(data=sample_df, config=_DEFAULT_CONFIG)

    def test_create_propagates_invalid_request_error_with_param(
        self, sample_df: pd.DataFrame
    ) -> None:
        """``InvalidRequestError.param`` must survive the propagation —
        users rely on it to identify which field was rejected."""
        original = InvalidRequestError("embedding_dim too large", param="embedding_dim")
        with patch.object(DataxidClient, "_request", side_effect=original):
            with pytest.raises(InvalidRequestError) as exc_info:
                Model.create(data=sample_df, config=_DEFAULT_CONFIG)
        assert exc_info.value.param == "embedding_dim"

    def test_generate_propagates_not_found_error(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        """A model deleted on the server side surfaces as ``NotFoundError``
        on subsequent ``generate`` calls — the SDK must not retry-loop or
        fall back."""
        with patch.object(DataxidClient, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                NotFoundError(f"Model {MODEL_ID} not found"),
            ]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            with pytest.raises(NotFoundError, match="not found"):
                model.generate()

    def test_delete_propagates_not_found_error(
        self, sample_df: pd.DataFrame, mock_train_frozen: None
    ) -> None:
        with patch.object(DataxidClient, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                NotFoundError(f"Model {MODEL_ID} not found"),
            ]
            model = Model.create(data=sample_df, config=_DEFAULT_CONFIG)
            with pytest.raises(NotFoundError):
                model.delete()
