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

import logging as _logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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

_logger = _logging.getLogger("dataxid")

api_key: str | None = None
base_url: str = "https://api.dataxid.com"


def synthesize(
    data: pd.DataFrame,
    n_samples: int = 100,
    config: dict | ModelConfig | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
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

    Returns:
        DataFrame with synthetic data
    """
    model = Model.create(
        data=data,
        n_samples=n_samples,
        config=config,
        api_key=api_key,
        base_url=base_url,
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


__all__ = [
    "api_key",
    "base_url",
    "synthesize",
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
