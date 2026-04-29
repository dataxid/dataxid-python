# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :mod:`dataxid.client._serialization`, the SDK's wire-format codec.

Two payload families are covered:

* **Embeddings** — :func:`serialize_embedding` / :func:`deserialize_embedding`
  for the ``numpy-b64`` (current) and ``json-flat`` (legacy) encodings, with
  auto-detection on the deserialize side so older servers stay compatible.
* **Encoder state** — :func:`serialize_encoder_state` /
  :func:`deserialize_encoder_state` checkpoint helpers used by the frozen
  training loop to ship encoder weights through the API.

Properties asserted: bidirectional equality on roundtrip, format auto-detection,
defaulted fields (``dtype``, ``encoding``), shape edge cases (scalar, empty,
high-rank), and gradient detachment.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest
import torch

from dataxid.client._serialization import (
    ENCODING_JSON_FLAT,
    ENCODING_NUMPY_B64,
    deserialize_embedding,
    deserialize_encoder_state,
    serialize_embedding,
    serialize_encoder_state,
)


class TestSerializeEmbedding:
    """Wire-format invariants for ``serialize_embedding``."""

    def test_payload_has_self_describing_fields(self) -> None:
        """The payload must carry every field a fresh deserializer needs:
        encoding tag, dtype, shape, and the b64 body."""
        payload = serialize_embedding(torch.randn(10, 64))
        assert payload["encoding"] == ENCODING_NUMPY_B64
        assert payload["dtype"] == "float32"
        assert payload["shape"] == [10, 64]
        assert isinstance(payload["embedding_b64"], str)

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.randn(64),         # 1D
            torch.randn(4, 8),       # 2D
            torch.randn(2, 3, 4),    # 3D
            torch.tensor(3.14),      # scalar
            torch.empty(0),          # empty
        ],
        ids=["1d", "2d", "3d", "scalar", "empty"],
    )
    def test_shape_preserved(self, tensor: torch.Tensor) -> None:
        payload = serialize_embedding(tensor)
        assert payload["shape"] == list(tensor.shape)

    def test_detaches_gradient(self) -> None:
        """A tensor with ``requires_grad=True`` would crash ``.numpy()`` if
        the implementation did not call ``.detach()``. Reaching the assert
        therefore proves detachment happened."""
        t = torch.randn(4, 8, requires_grad=True)
        payload = serialize_embedding(t)
        restored = deserialize_embedding(payload)
        assert restored.requires_grad is False


class TestDeserializeEmbedding:
    """Wire-format ingestion + auto-detection for ``deserialize_embedding``."""

    def test_numpy_b64_roundtrip_preserves_values(self) -> None:
        original = torch.randn(10, 64)
        restored = deserialize_embedding(serialize_embedding(original))
        assert restored.shape == original.shape
        assert restored.dtype == torch.float32
        torch.testing.assert_close(restored, original, atol=1e-6, rtol=1e-6)

    def test_legacy_list_input_full_value_match(self) -> None:
        """Legacy callers pass a raw nested list — every element must be
        preserved with float32 precision, not just the shape."""
        data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = deserialize_embedding(data)
        assert result.shape == (2, 3)
        torch.testing.assert_close(
            result, torch.tensor(data, dtype=torch.float32), atol=1e-6, rtol=1e-6,
        )

    def test_legacy_json_flat_dict_full_value_match(self) -> None:
        """Same legacy contract, this time wrapped in the ``json-flat`` envelope."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        payload = {"data": data, "encoding": ENCODING_JSON_FLAT}
        result = deserialize_embedding(payload)
        torch.testing.assert_close(
            result, torch.tensor(data, dtype=torch.float32), atol=1e-6, rtol=1e-6,
        )

    def test_dtype_field_is_optional(self) -> None:
        """A payload missing ``dtype`` must default to float32, otherwise
        older servers that emit shape+b64 only would break."""
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        payload = {
            "embedding_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": [1, 2],
            "encoding": ENCODING_NUMPY_B64,
        }
        result = deserialize_embedding(payload)
        assert result.dtype == torch.float32
        torch.testing.assert_close(
            result, torch.tensor(arr, dtype=torch.float32), atol=1e-6, rtol=1e-6,
        )

    def test_encoding_field_is_optional_defaults_to_json_flat(self) -> None:
        """Pre-encoding-tag payloads (no ``encoding`` field) must be treated
        as legacy ``json-flat`` rather than rejected — protects backward
        compatibility with older API responses."""
        result = deserialize_embedding({"data": [[1.0, 2.0]]})
        torch.testing.assert_close(
            result, torch.tensor([[1.0, 2.0]], dtype=torch.float32), atol=1e-6, rtol=1e-6,
        )

    def test_default_device_is_cpu(self) -> None:
        result = deserialize_embedding(serialize_embedding(torch.randn(3, 8)))
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="cuda not available in this environment",
    )
    def test_device_parameter_routes_to_cuda(self) -> None:
        result = deserialize_embedding(
            serialize_embedding(torch.randn(3, 8)), device="cuda",
        )
        assert result.device.type == "cuda"

    def test_unknown_encoding_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding encoding"):
            deserialize_embedding({"encoding": "arrow-v99"})


class TestCrossFormatConsistency:
    """The same logical tensor must compare equal across both wire encodings.

    Pinning this prevents a future codec tweak from silently introducing a
    precision mismatch between the two paths.
    """

    def test_numpy_b64_matches_json_flat_for_same_tensor(self) -> None:
        original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        from_b64 = deserialize_embedding(
            serialize_embedding(torch.tensor(original, dtype=torch.float32))
        )
        from_legacy = deserialize_embedding(original)
        torch.testing.assert_close(from_b64, from_legacy, atol=1e-6, rtol=1e-6)


class _MockBackend:
    """Minimal stand-in for the encoder backend used by the checkpoint helpers.

    The real backend wraps a much larger module, but the checkpoint contract
    only reads ``backend.encoder.state_dict()`` — so any ``nn.Module``
    behind ``.encoder`` is enough to exercise the roundtrip.
    """

    def __init__(self, in_features: int = 4, out_features: int = 8) -> None:
        self.encoder = torch.nn.Linear(in_features, out_features)


def _adamw(backend: _MockBackend) -> torch.optim.AdamW:
    return torch.optim.AdamW(backend.encoder.parameters(), lr=0.01)


class TestEncoderStateRoundTrip:
    """Checkpoint helpers — frozen-training relies on a lossless save/load cycle."""

    def test_serialize_returns_bytes(self) -> None:
        backend = _MockBackend()
        data = serialize_encoder_state(backend, _adamw(backend))
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialize_restores_weights_exactly(self) -> None:
        """Trained weights must come back bit-identical — any drift would
        invalidate the encoder mid-training."""
        original = _MockBackend()
        data = serialize_encoder_state(original, _adamw(original))

        restored = _MockBackend()
        deserialize_encoder_state(data, restored, _adamw(restored))

        for p1, p2 in zip(
            original.encoder.parameters(), restored.encoder.parameters(), strict=True
        ):
            assert torch.equal(p1, p2)

    def test_deserialize_restores_optimizer_state(self) -> None:
        """The optimizer's running statistics (Adam's m, v) survive the
        roundtrip after a step has been taken — without this, training
        could not be paused and resumed."""
        original = _MockBackend()
        original_opt = _adamw(original)

        # Take one step so the optimizer has live state to serialize.
        loss = original.encoder(torch.randn(2, 4)).sum()
        loss.backward()
        original_opt.step()

        data = serialize_encoder_state(original, original_opt)

        restored = _MockBackend()
        restored.encoder.load_state_dict(original.encoder.state_dict())
        restored_opt = _adamw(restored)
        deserialize_encoder_state(data, restored, restored_opt)

        original_state = original_opt.state_dict()["state"]
        restored_state = restored_opt.state_dict()["state"]
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            for field, value in original_state[key].items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, restored_state[key][field])
                else:
                    assert value == restored_state[key][field]

    def test_deserialize_mutates_in_place(self) -> None:
        """The helper returns ``None`` and mutates the supplied backend +
        optimizer; callers rely on this to thread state through."""
        original = _MockBackend()
        data = serialize_encoder_state(original, _adamw(original))

        restored = _MockBackend()
        opt = _adamw(restored)
        result: Any = deserialize_encoder_state(data, restored, opt)
        assert result is None
