# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Model class — training and generation API.

Raw data never leaves the SDK. Only embeddings cross the wire.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch

from dataxid.client._http import DataxidClient
from dataxid.encoder._wrapper import Encoder
from dataxid.pipeline._decode import compute_fixed_probs, decode_columns
from dataxid.training._config import ModelConfig, _resolve_config
from dataxid.training._frozen import train_frozen

logger = logging.getLogger(__name__)

class Model:
    """A Dataxid model. Trains on your data locally, generates synthetic data via API."""

    def __init__(
        self,
        model_id: str,
        client: DataxidClient,
        encoder: Encoder,
        data: pd.DataFrame,
        config: dict[str, Any],
    ):
        self.id = model_id
        self.status: str = "training"
        self._client = client
        self._encoder = encoder
        self._data = data
        self._config = config
        self._n_samples: int | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.stopped_early: bool = False
        self._best_encoder_state: bytes | None = None
        self._encoder_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        n_samples: int | None = None,
        config: dict[str, Any] | ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> Model:
        """
        Create a model: analyze data locally, register via API, and train.

        Args:
            data: Training DataFrame (stays local — only embeddings sent to API)
            n_samples: Number of synthetic rows (stored for generate())
            config: Training config — ModelConfig instance or plain dict
            api_key: Override dataxid.api_key
            base_url: Override dataxid.base_url

        Returns:
            Trained Model ready for generate()
        """
        config = _resolve_config(config)
        http = DataxidClient(api_key=api_key, base_url=base_url)

        embedding_dim = config["embedding_dim"]
        model_size = config["model_size"]
        batch_size = config["batch_size"]
        max_epochs = config["max_epochs"]
        early_stop_patience = config["early_stop_patience"]
        privacy_enabled = config["privacy_enabled"]
        privacy_noise = config["privacy_noise"]
        encoding_types = config["encoding_types"]
        val_split = config["val_split"]

        # --- 1. Analyze locally ---
        logger.info("Analyzing data locally...")
        encoder = Encoder(
            embedding_dim=embedding_dim,
            model_size=model_size,
            privacy_enabled=privacy_enabled,
            privacy_noise=privacy_noise,
        )
        metadata = encoder.analyze(data, encoding_types=encoding_types)
        encoder.prepare(data)

        # --- 2. Register model on API ---
        logger.info("Creating model on API...")
        resp = http.post("/v1/models", json={
            "metadata": metadata,
            "config": {
                "embedding_dim": embedding_dim,
                "model_size": model_size,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
            },
        })
        model_id = resp["data"]["id"]
        logger.info("Model created: %s", model_id)

        model = cls(
            model_id=model_id,
            client=http,
            encoder=encoder,
            data=data,
            config=config,
        )
        model._n_samples = n_samples

        # --- 3. Train ---
        train_frozen(
            model,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            val_split=val_split,
        )

        return model

    def generate(
        self,
        n_samples: int | None = None,
        seed_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic data from the trained model.

        Args:
            n_samples: Number of rows. Defaults to len(training_data).
            seed_data: DataFrame for conditional generation (optional).

        Returns:
            DataFrame with synthetic data
        """
        n = n_samples or getattr(self, "_n_samples", None) or len(self._data)

        logger.info("Generating %d synthetic rows...", n)
        embedding = self._encoder._generation_embedding(
            n_samples=n, seed_data=seed_data,
        )

        column_stats = self._encoder.column_stats
        fixed_probs = compute_fixed_probs(column_stats) if column_stats else None

        payload: dict[str, Any] = {"embedding": embedding}
        if fixed_probs:
            payload["fixed_probs"] = fixed_probs

        resp = self._client.post(f"/v1/models/{self.id}/generate", json=payload)

        raw_codes = resp["data"]["codes"]
        features = self._encoder.features
        df = decode_columns(raw_codes, features, column_stats)
        self.status = "ready"

        logger.info("Generated %d rows, %d columns", len(df), len(df.columns))
        return df

    def delete(self) -> None:
        """Delete the model and free server resources."""
        self._client.delete(f"/v1/models/{self.id}")
        self.status = "deleted"
        logger.info("Model %s deleted", self.id)

    def refresh(self) -> dict[str, Any]:
        """Fetch latest model status from API."""
        resp = self._client.get(f"/v1/models/{self.id}")
        self.status = resp["data"]["status"]
        return resp["data"]

