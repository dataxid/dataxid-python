# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Embedding and gradient binary serialization.

Auto-detects numpy-b64 and list[list[float]] wire formats.
"""

import base64
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

ENCODING_NUMPY_B64 = "numpy-b64"
ENCODING_JSON_FLAT = "json-flat"


# ---------------------------------------------------------------------------
# Serialize — tensor → wire payload
# ---------------------------------------------------------------------------

def serialize_embedding(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert torch.Tensor to self-describing binary payload."""
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    return {
        "embedding_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": "float32",
        "shape": list(arr.shape),
        "encoding": ENCODING_NUMPY_B64,
    }


# ---------------------------------------------------------------------------
# Deserialize — wire payload → tensor
# ---------------------------------------------------------------------------

def deserialize_embedding(
    payload: dict[str, Any] | list,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert wire payload to torch.Tensor. Auto-detects format."""
    if isinstance(payload, list):
        return torch.tensor(payload, dtype=torch.float32, device=device)

    encoding = payload.get("encoding", ENCODING_JSON_FLAT)

    if encoding == ENCODING_NUMPY_B64:
        raw = base64.b64decode(payload["embedding_b64"])
        dtype = np.dtype(payload.get("dtype", "float32"))
        shape = tuple(payload["shape"])
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        return torch.from_numpy(arr.copy()).to(device)

    if encoding == ENCODING_JSON_FLAT:
        return torch.tensor(payload.get("data", payload), dtype=torch.float32, device=device)

    raise ValueError(f"Unknown embedding encoding: {encoding}")


# ---------------------------------------------------------------------------
# Encoder state serialization
# ---------------------------------------------------------------------------

def serialize_encoder_state(
    backend: Any,
    optimizer: torch.optim.Optimizer,
) -> bytes:
    """Serialize encoder weights + optimizer state to bytes for checkpoint."""
    import io

    buffer = io.BytesIO()
    torch.save(
        {
            "encoder_state_dict": backend.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        buffer,
    )
    return buffer.getvalue()


def deserialize_encoder_state(
    data: bytes,
    backend: Any,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Load encoder weights + optimizer state from checkpoint bytes. Mutates in-place."""
    import io

    buffer = io.BytesIO(data)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    backend.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
