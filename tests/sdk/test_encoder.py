# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :class:`dataxid.encoder.Encoder`, the SDK's client-side encoder.

The encoder owns the analyze → prepare → encode pipeline that turns raw
DataFrames into the wire payload sent to the API. Tests are organized by
public surface and data shape:

- ``analyze`` / ``prepare`` / ``encode_batch`` / ``_generation_embedding``
  contracts on flat, contextual, and sequential data.
- State machine: methods refuse to run before their prerequisites.
- ``protect_rare`` toggle and the ``wire_key`` helper.

Tests that intentionally depend on private state (underscore-prefixed
attributes, ``_encode_batches``, ``_nn`` constants) live in
``TestInternalContracts`` so the tradeoff is visible.
"""

import json

import pandas as pd
import pytest
import torch

from dataxid.client._serialization import ENCODING_NUMPY_B64, deserialize_embedding
from dataxid.encoder import Encoder
from dataxid.encoder._nn import SIDX_PREFIX
from dataxid.encoder._ports import (
    WIRE_COLUMN_SEP,
    WIRE_PREFIX,
    WIRE_SUB_COLUMN_SEP,
    WIRE_TABLE_SEP,
    wire_key,
)
from dataxid.exceptions import ModelNotReadyError


class TestAnalyze:
    """``Encoder.analyze()`` — raw data to API metadata."""

    def test_contains_required_keys(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        required = {
            "cardinalities",
            "features",
            "column_stats",
            "value_mappings",
            "empirical_probs",
        }
        assert required.issubset(meta.keys())

    def test_features_match_columns(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert meta["features"] == list(sample_df.columns)

    def test_cardinalities_are_positive_ints(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        for card in meta["cardinalities"].values():
            assert isinstance(card, int)
            assert card > 0

    def test_empirical_probs_are_lists(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        for probs in meta["empirical_probs"].values():
            assert isinstance(probs, list)
            assert all(isinstance(p, float) for p in probs)

    def test_empirical_probs_sum_near_one(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        for probs in meta["empirical_probs"].values():
            assert abs(sum(probs) - 1.0) < 0.05

    def test_column_stats_has_entries(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert len(meta["column_stats"]) > 0

    def test_metadata_is_json_serializable(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert len(json.dumps(meta)) > 0

    def test_is_deterministic_for_same_input(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        """Two analyze() calls on the same DataFrame must produce identical
        metadata so downstream training and decoding are reproducible."""
        first = encoder.analyze(sample_df)
        second = encoder.analyze(sample_df)
        assert first["features"] == second["features"]
        assert first["cardinalities"] == second["cardinalities"]
        assert first["column_stats"] == second["column_stats"]
        assert first["empirical_probs"] == second["empirical_probs"]


class TestPrepare:
    """``Encoder.prepare()`` — pre-encode data into tensors."""

    def test_succeeds_after_analyze(
        self, analyzed_encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        analyzed_encoder.prepare(sample_df)

    def test_fails_before_analyze(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ModelNotReadyError):
            encoder.prepare(sample_df)


class TestEncodeBatch:
    """``Encoder.encode_batch()`` produces a wire payload + targets + id."""

    def test_returns_tuple(self, prepared_encoder: Encoder) -> None:
        embedding, targets, eid = prepared_encoder.encode_batch()
        assert isinstance(embedding, dict)
        assert isinstance(targets, dict)
        assert isinstance(eid, int)

    def test_embedding_is_binary_payload(
        self, prepared_encoder: Encoder
    ) -> None:
        embedding, _, _ = prepared_encoder.encode_batch()
        assert embedding["encoding"] == ENCODING_NUMPY_B64
        assert embedding["dtype"] == "float32"
        assert "embedding_b64" in embedding
        assert "shape" in embedding

    def test_embedding_shape(
        self, prepared_encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        embedding, _, _ = prepared_encoder.encode_batch()
        assert embedding["shape"][0] == len(sample_df)
        assert embedding["shape"][1] == 16

    def test_targets_have_entries(self, prepared_encoder: Encoder) -> None:
        _, targets, _ = prepared_encoder.encode_batch()
        assert len(targets) > 0

    def test_targets_values_are_int_lists(
        self, prepared_encoder: Encoder
    ) -> None:
        _, targets, _ = prepared_encoder.encode_batch()
        for values in targets.values():
            assert isinstance(values, list)
            assert all(isinstance(v, int) for v in values)

    def test_batch_with_indices(self, prepared_encoder: Encoder) -> None:
        embedding, _, _ = prepared_encoder.encode_batch(indices=[0, 1, 2])
        assert embedding["shape"][0] == 3

    def test_batch_with_empty_indices_returns_empty_payload(
        self, prepared_encoder: Encoder
    ) -> None:
        """``indices=[]`` must round-trip as an empty batch — no crash, full
        payload structure preserved — so callers can build batches safely
        from filtered index sets without special-casing the empty result."""
        embedding, targets, _ = prepared_encoder.encode_batch(indices=[])
        assert embedding["shape"][0] == 0
        assert embedding["shape"][1] == 16
        assert embedding["encoding"] == ENCODING_NUMPY_B64
        for values in targets.values():
            assert values == []

    def test_embed_id_increments(self, prepared_encoder: Encoder) -> None:
        _, _, eid1 = prepared_encoder.encode_batch()
        _, _, eid2 = prepared_encoder.encode_batch()
        assert eid2 == eid1 + 1

    def test_fails_before_prepare(
        self, analyzed_encoder: Encoder
    ) -> None:
        with pytest.raises(RuntimeError, match="prepare"):
            analyzed_encoder.encode_batch()

    def test_fails_before_analyze(self, encoder: Encoder) -> None:
        with pytest.raises(ModelNotReadyError):
            encoder.encode_batch()

    def test_json_serializable(self, prepared_encoder: Encoder) -> None:
        embedding, targets, _ = prepared_encoder.encode_batch()
        json.dumps({"embedding": embedding, "targets": targets})


class TestGenerationEmbedding:
    """``Encoder._generation_embedding()`` — context-conditioned embedding."""

    def test_returns_binary_payload(
        self, analyzed_encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        result = analyzed_encoder._generation_embedding(
            n_samples=len(sample_df)
        )
        assert isinstance(result, dict)
        assert result["encoding"] == ENCODING_NUMPY_B64

    def test_zero_embedding_by_default(
        self, analyzed_encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        """Default path emits a zero embedding for classifier-free guidance."""
        result = analyzed_encoder._generation_embedding(
            n_samples=len(sample_df)
        )
        assert result["shape"] == [len(sample_df), 16]
        emb = deserialize_embedding(result)
        assert torch.all(emb == 0)

    def test_conditions_produces_real_embedding(
        self, analyzed_encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        """Passing conditions switches to conditional generation."""
        result = analyzed_encoder._generation_embedding(
            n_samples=len(sample_df), conditions=sample_df,
        )
        assert result["shape"] == [len(sample_df), 16]
        emb = deserialize_embedding(result)
        assert not torch.all(emb == 0)

    def test_fails_before_analyze(self, encoder: Encoder) -> None:
        with pytest.raises(ModelNotReadyError):
            encoder._generation_embedding(n_samples=10)

    def test_json_serializable(self, analyzed_encoder: Encoder) -> None:
        result = analyzed_encoder._generation_embedding(n_samples=5)
        json.dumps(result)

    def test_zero_n_samples_returns_empty_payload(
        self, analyzed_encoder: Encoder
    ) -> None:
        """Generating zero samples must return an empty embedding rather
        than raise, so callers can drive generation from arbitrary loop
        sizes without guarding against the zero case."""
        result = analyzed_encoder._generation_embedding(n_samples=0)
        assert result["shape"] == [0, 16]
        assert result["encoding"] == ENCODING_NUMPY_B64


class TestStateMachine:
    """Encoder enforces ``analyze`` → ``prepare`` → ``encode`` ordering."""

    def test_fresh_encoder_rejects_prepare(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ModelNotReadyError):
            encoder.prepare(sample_df)

    def test_fresh_encoder_rejects_encode_batch(
        self, encoder: Encoder
    ) -> None:
        with pytest.raises(ModelNotReadyError):
            encoder.encode_batch()

    def test_fresh_encoder_rejects_generation_embedding(
        self, encoder: Encoder
    ) -> None:
        with pytest.raises(ModelNotReadyError):
            encoder._generation_embedding(n_samples=10)

    def test_analyzed_encoder_rejects_encode_batch(
        self, analyzed_encoder: Encoder
    ) -> None:
        with pytest.raises(RuntimeError, match="prepare"):
            analyzed_encoder.encode_batch()

    def test_full_flow(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert "features" in meta

        encoder.prepare(sample_df)
        embedding, _, _ = encoder.encode_batch()
        assert embedding["shape"][0] == len(sample_df)

        gen_emb = encoder._generation_embedding(n_samples=len(sample_df))
        assert gen_emb["shape"][0] == len(sample_df)


class TestContextAnalyze:
    """``Encoder.analyze(parent=)`` — parent-table metadata merging."""

    def test_has_context_flag_set(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(sample_df, parent=ctx_df)
        assert meta["has_context"] is True

    def test_has_context_flag_false_without_ctx(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert meta["has_context"] is False

    def test_cardinalities_exclude_context_keys(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(sample_df, parent=ctx_df)
        target_only = encoder.analyze(sample_df)
        assert len(meta["cardinalities"]) == len(target_only["cardinalities"])

    def test_empirical_probs_exclude_context(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(sample_df, parent=ctx_df)
        target_only = encoder.analyze(sample_df)
        assert len(meta["empirical_probs"]) == len(
            target_only["empirical_probs"]
        )

    def test_features_are_target_only(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(sample_df, parent=ctx_df)
        assert meta["features"] == list(sample_df.columns)

    def test_metadata_is_json_serializable(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(sample_df, parent=ctx_df)
        json.dumps(meta)


class TestContextPrepareAndEncode:
    """``prepare(parent=)`` + ``encode_batch()`` with context tensors."""

    def test_prepare_with_context(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(sample_df, parent=ctx_df)
        encoder.prepare(sample_df, parent=ctx_df)
        assert encoder.has_context is True

    def test_encode_batch_with_context_shape(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(sample_df, parent=ctx_df)
        encoder.prepare(sample_df, parent=ctx_df)
        embedding, _, _ = encoder.encode_batch()
        assert embedding["shape"][0] == len(sample_df)
        assert embedding["shape"][1] == 16

    def test_encode_batch_indices_with_context(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(sample_df, parent=ctx_df)
        encoder.prepare(sample_df, parent=ctx_df)
        embedding, _, _ = encoder.encode_batch(indices=[0, 1, 2])
        assert embedding["shape"][0] == 3

    def test_generation_embedding_with_context(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(sample_df, parent=ctx_df)
        result = encoder._generation_embedding(
            n_samples=len(sample_df), conditions=sample_df, parent=ctx_df,
        )
        assert result["shape"] == [len(sample_df), 16]
        emb = deserialize_embedding(result)
        assert not torch.all(emb == 0)


class TestBackwardCompatEncoder:
    """Context-free calls must keep the pre-context behavior."""

    def test_analyze_without_context_unchanged(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert meta["has_context"] is False
        assert "cardinalities" in meta
        assert meta["features"] == list(sample_df.columns)

    def test_prepare_without_context_unchanged(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        encoder.analyze(sample_df)
        encoder.prepare(sample_df)
        assert encoder.has_context is False

    def test_encode_batch_without_context_unchanged(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        encoder.analyze(sample_df)
        encoder.prepare(sample_df)
        embedding, _, _ = encoder.encode_batch()
        assert embedding["shape"] == [len(sample_df), 16]


class TestSequentialAnalyze:
    """``Encoder.analyze(foreign_key=)`` — sequential mode detection."""

    def test_sequential_detected(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        assert meta["is_sequential"] is True

    def test_seq_len_params(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        assert meta["seq_len_max"] == 3
        assert meta["seq_len_median"] == 3

    def test_features_exclude_foreign_key(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        assert "account_id" not in meta["features"]

    def test_has_context_true(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        assert meta["has_context"] is True

    def test_not_sequential_without_key(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert meta["is_sequential"] is False
        assert meta["seq_len_max"] == 1


class TestSequentialPrepare:
    """``prepare()`` in sequential mode — entity-level tensor padding."""

    def test_prepare_sequential(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        encoder.prepare(seq_df, parent=seq_ctx_df, parent_key="id")
        assert encoder.is_sequential is True


class TestSequentialGenerationEmbedding:
    """``_generation_embedding()`` in sequential mode."""

    def test_returns_embedding(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        result = encoder._generation_embedding(n_samples=3, parent=seq_ctx_df)
        assert isinstance(result, dict)
        assert result["shape"][0] == 3

    def test_zero_embedding_without_ctx(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        result = encoder._generation_embedding(n_samples=5)
        assert result["shape"][0] == 5
        emb = deserialize_embedding(result)
        assert torch.all(emb == 0)

    def test_ctx_embedding_not_zero(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        result = encoder._generation_embedding(n_samples=3, parent=seq_ctx_df)
        emb = deserialize_embedding(result)
        assert not torch.all(emb == 0)


class TestBackwardCompatSequential:
    """Sequential additions must not affect flat-mode behavior."""

    def test_flat_analyze_unchanged(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        meta = encoder.analyze(sample_df)
        assert meta["is_sequential"] is False
        assert meta["seq_len_max"] == 1
        assert meta["seq_len_median"] == 1

    def test_flat_encode_unchanged(
        self, encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        encoder.analyze(sample_df)
        encoder.prepare(sample_df)
        embedding, _, _ = encoder.encode_batch()
        assert embedding["shape"] == [len(sample_df), 16]


class TestProtectRare:
    """``Encoder(protect_rare=...)`` — rare-category suppression toggle."""

    @staticmethod
    def _make_encoder(protect_rare: bool) -> Encoder:
        return Encoder(
            embedding_dim=16,
            model_size="small",
            device="cpu",
            protect_rare=protect_rare,
        )

    @staticmethod
    def _rare_df() -> pd.DataFrame:
        """Four frequent categories plus one rare label that triggers protection."""
        return pd.DataFrame({
            "cat": ["A"] * 50 + ["B"] * 50 + ["C"] * 50 + ["D"] * 50
                   + ["Z_rare"] * 2,
        })

    def test_default_is_true(self) -> None:
        assert Encoder().protect_rare_enabled

    def test_protection_on_replaces_rare_with_token(self) -> None:
        enc = self._make_encoder(protect_rare=True)
        enc.analyze(self._rare_df())
        codes = enc.column_stats["cat"]["codes"]
        assert "Z_rare" not in codes
        assert "<protected>" in codes

    def test_protection_off_preserves_rare_value(self) -> None:
        enc = self._make_encoder(protect_rare=False)
        enc.analyze(self._rare_df())
        codes = enc.column_stats["cat"]["codes"]
        assert "Z_rare" in codes
        assert enc.column_stats["cat"].get("no_of_rare_categories", 0) == 0


class TestWireKey:
    """``wire_key()`` — feature/sub-column → wire identifier."""

    def test_format(self) -> None:
        assert wire_key("age", "cat") == "feat:/age__cat"

    def test_uses_constants(self) -> None:
        expected = (
            f"{WIRE_PREFIX}{WIRE_TABLE_SEP}{WIRE_COLUMN_SEP}col"
            f"{WIRE_SUB_COLUMN_SEP}sub"
        )
        assert wire_key("col", "sub") == expected

    def test_special_chars_in_feature(self) -> None:
        assert wire_key("first_name", "char_0") == "feat:/first_name__char_0"


class TestInternalContracts:
    """Tests that intentionally depend on private state.

    These exist because the public surface does not yet expose enough hooks
    to assert the underlying invariant. They guard against regressions in
    code paths that have no observable proxy today (e.g. the internal batch
    builder, the rare-category constructor flag, and the positional index
    cardinalities used by the sequential model).

    When the corresponding public attribute or method is added, the matching
    test should be moved out of this class and rewritten against it.
    """

    def test_freeze_disables_grad_on_all_backend_parameters(
        self, prepared_encoder: Encoder
    ) -> None:
        """Frozen-encoder training mode must not propagate gradients into the
        encoder, otherwise the API would have to round-trip weight updates."""
        prepared_encoder.freeze()
        assert all(not p.requires_grad for p in prepared_encoder._backend.parameters())

    def test_freeze_before_analyze_raises(self) -> None:
        with pytest.raises(ModelNotReadyError, match="analyze"):
            Encoder(embedding_dim=16, model_size="small").freeze()

    def test_encode_batches_partitions_rows_into_train_and_validation(
        self, prepared_encoder: Encoder, sample_df: pd.DataFrame
    ) -> None:
        """Every row appears exactly once across train+val with the expected
        per-batch split (``batch_size=4``, ``val_split=0.25`` → 6 train rows
        in two batches, 2 validation rows in one batch)."""
        prepared_encoder.freeze()
        batches = prepared_encoder._encode_batches(
            batch_size=4, val_split=0.25
        )

        train = [b for b in batches if not b["is_validation"]]
        val = [b for b in batches if b["is_validation"]]

        assert len(train) == 2
        assert len(val) == 1

        train_rows = sum(b["embedding"]["shape"][0] for b in train)
        val_rows = sum(b["embedding"]["shape"][0] for b in val)
        assert train_rows + val_rows == len(sample_df)
        assert val_rows == 2

        for b in batches:
            assert "embedding" in b
            assert "targets" in b

    def test_encode_batches_clears_live_embeddings_after_serialization(
        self, prepared_encoder: Encoder
    ) -> None:
        """``_encode_batches`` must release every staged tensor so a long
        training run does not leak GPU memory across epochs."""
        prepared_encoder.freeze()
        prepared_encoder._encode_batches(batch_size=4, val_split=0.25)
        assert prepared_encoder._live_embeddings == {}

    def test_encode_batches_before_prepare_raises(
        self, analyzed_encoder: Encoder
    ) -> None:
        analyzed_encoder.freeze()
        with pytest.raises(RuntimeError, match="prepare"):
            analyzed_encoder._encode_batches()

    def test_encode_batches_with_context_produces_train_and_validation(
        self,
        encoder: Encoder,
        sample_df: pd.DataFrame,
        ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(sample_df, parent=ctx_df)
        encoder.prepare(sample_df, parent=ctx_df)
        encoder.freeze()
        encoder.eval_mode()
        batches = encoder._encode_batches(batch_size=4)
        train = [b for b in batches if not b["is_validation"]]
        val = [b for b in batches if b["is_validation"]]
        assert len(train) >= 1
        assert len(val) == 1

    def test_encode_batches_marks_sequential_flag(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        encoder.prepare(seq_df, parent=seq_ctx_df, parent_key="id")
        encoder.freeze()
        encoder.eval_mode()
        batches = encoder._encode_batches(batch_size=4)
        train_batches = [b for b in batches if not b.get("is_validation")]
        assert train_batches[0].get("is_sequential") is True

    def test_seq_data_populated_after_sequential_prepare(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        encoder.prepare(seq_df, parent=seq_ctx_df, parent_key="id")
        first_key = next(iter(encoder._seq_data))
        assert len(encoder._seq_data[first_key]) == 3

    def test_positional_cardinalities_present_in_sequential_meta(
        self,
        encoder: Encoder,
        seq_df: pd.DataFrame,
        seq_ctx_df: pd.DataFrame,
    ) -> None:
        meta = encoder.analyze(
            seq_df, parent=seq_ctx_df,
            foreign_key="account_id", parent_key="id",
        )
        sidx_keys = [
            k for k in meta["cardinalities"] if k.startswith(SIDX_PREFIX)
        ]
        assert len(sidx_keys) == 1

    def test_apply_gradient_accumulates_without_stepping(
        self, prepared_encoder: Encoder
    ) -> None:
        """``_apply_gradient`` is the frozen-training hook: it stores gradients
        on the backend parameters but must never call ``optimizer.step()`` —
        otherwise the encoder would silently drift between API calls."""
        params_before = [p.clone() for p in prepared_encoder._backend.parameters()]
        n_rows = len(prepared_encoder._tensors[next(iter(prepared_encoder._tensors))])
        _, _, eid = prepared_encoder.encode_batch(list(range(n_rows)))
        grad = torch.randn(n_rows, prepared_encoder._embedding_dim)

        prepared_encoder._apply_gradient(grad, embed_id=eid)

        for before, after in zip(
            params_before, prepared_encoder._backend.parameters(), strict=True
        ):
            assert torch.equal(before, after), (
                "_apply_gradient must accumulate gradients only, never step"
            )
        assert any(
            p.grad is not None and p.grad.any()
            for p in prepared_encoder._backend.parameters()
        )

    def test_step_after_apply_gradient_updates_weights(
        self, prepared_encoder: Encoder
    ) -> None:
        """The companion test: ``step()`` *does* commit the accumulated gradient."""
        prepared_encoder.zero_grad()
        n_rows = len(prepared_encoder._tensors[next(iter(prepared_encoder._tensors))])
        _, _, eid = prepared_encoder.encode_batch(list(range(n_rows)))
        grad = torch.randn(n_rows, prepared_encoder._embedding_dim)
        params_before = [p.clone() for p in prepared_encoder._backend.parameters()]

        prepared_encoder._apply_gradient(grad, embed_id=eid)
        prepared_encoder.step()

        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                params_before, prepared_encoder._backend.parameters(), strict=True
            )
        )

    def test_apply_gradient_with_subset_indices(
        self, prepared_encoder: Encoder
    ) -> None:
        """Subset batches (e.g. validation slices) must accept gradients of the
        matching length without index errors."""
        indices = [0, 2, 4]
        prepared_encoder.zero_grad()
        _, _, eid = prepared_encoder.encode_batch(indices)
        grad = torch.randn(len(indices), prepared_encoder._embedding_dim)
        params_before = [p.clone() for p in prepared_encoder._backend.parameters()]

        prepared_encoder._apply_gradient(grad, embed_id=eid)
        prepared_encoder.step()

        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                params_before, prepared_encoder._backend.parameters(), strict=True
            )
        )

    def test_discard_embedding_removes_from_live_set(
        self, prepared_encoder: Encoder
    ) -> None:
        """``_discard_embedding`` is the cleanup hook called after each batch."""
        n_rows = len(prepared_encoder._tensors[next(iter(prepared_encoder._tensors))])
        _, _, eid = prepared_encoder.encode_batch(list(range(n_rows)))
        assert eid in prepared_encoder._live_embeddings

        prepared_encoder._discard_embedding(eid)

        assert eid not in prepared_encoder._live_embeddings

    def test_apply_gradient_without_live_embedding_raises(
        self, prepared_encoder: Encoder
    ) -> None:
        with pytest.raises(RuntimeError, match="No live embedding"):
            prepared_encoder._apply_gradient(
                torch.randn(4, prepared_encoder._embedding_dim), embed_id=999
            )
