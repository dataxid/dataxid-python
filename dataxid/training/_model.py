# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Model class — training and generation API.

Raw data never leaves the user's machine; only abstract embeddings are
shared with the API.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import torch

from dataxid.client._http import DataxidClient
from dataxid.encoder._wrapper import Encoder
from dataxid.exceptions import InvalidRequestError
from dataxid.pipeline._decode import (
    bias_column_order,
    compute_distribution_probs,
    compute_fixed_probs,
    compute_imputation_probs,
    conditional_column_order,
    decode_columns,
    distribution_column_order,
    imputation_column_order,
    merge_fixed_probs,
    resolve_bias_payload,
)
from dataxid.pipeline._encode import (
    encode_conditions_fixed_values,
    encode_sequential_conditions_fixed_values,
)
from dataxid.training._config import (
    Bias,
    Distribution,
    ModelConfig,
    RareStrategy,
    Synthetic,
    _resolve_config,
    _validate_encoding_types,
)
from dataxid.training._frozen import train_frozen
from dataxid.training._seed import _set_seed

logger = logging.getLogger(__name__)

_RARE_STRATEGY_VALUES: tuple[str, ...] = get_args(RareStrategy)

# Sentinel for "caller did not pass this kwarg" — enables Synthetic preset
# merging without colliding with legitimate None defaults.
_UNSET: Any = object()


def _normalize_distribution(distribution: Distribution | None) -> dict | None:
    """Normalize a :class:`Distribution` input into its plain-dict form.

    Accepts only :class:`Distribution` instances (or ``None``). Dict inputs
    are rejected with :class:`InvalidRequestError`; use
    ``Distribution(column=..., ...)`` instead.
    """
    if distribution is None:
        return None
    if isinstance(distribution, Distribution):
        return asdict(distribution)
    raise InvalidRequestError(
        "distribution must be a Distribution instance or None, got "
        f"{type(distribution).__name__}. Use dataxid.Distribution(...) — "
        "dict inputs are no longer supported as of v0.3.0.",
        param="distribution",
    )


def _merge_field(
    kwarg_value: Any,
    synthetic: Synthetic | None,
    field_name: str,
    default: Any,
) -> Any:
    """Resolve a scalar ``generate`` kwarg against a ``Synthetic`` preset.

    Priority: explicit kwarg > ``synthetic.<field>`` > ``default``.

    ``kwarg_value is _UNSET`` means the caller did not pass the argument,
    so the preset (or the ``default`` fallback) is consulted.
    """
    if kwarg_value is not _UNSET:
        return kwarg_value
    if synthetic is not None:
        return getattr(synthetic, field_name)
    return default


def _merge_n(n_samples: int | None, synthetic: Synthetic | None) -> int | None:
    """Resolve effective row count from ``n_samples`` and ``Synthetic.n``.

    - Both unset → ``None`` (caller / downstream decides).
    - Only one set → that value.
    - Both set and equal → that value.
    - Both set but conflicting → :class:`InvalidRequestError`.
    """
    preset_n = synthetic.n if synthetic is not None else None
    if n_samples is None:
        return preset_n
    if preset_n is None or preset_n == n_samples:
        return n_samples
    raise InvalidRequestError(
        f"n_samples ({n_samples}) conflicts with synthetic.n ({preset_n}). "
        "Pass only one.",
        param="n_samples",
    )


def _normalize_bias(bias: Bias | None) -> dict | None:
    """Normalize a :class:`Bias` input into its plain-dict form.

    Accepts only :class:`Bias` instances (or ``None``). Dict inputs are
    rejected with :class:`InvalidRequestError`; use
    ``Bias(target=..., sensitive=...)`` instead.
    """
    if bias is None:
        return None
    if isinstance(bias, Bias):
        return asdict(bias)
    raise InvalidRequestError(
        "bias must be a Bias instance or None, got "
        f"{type(bias).__name__}. Use dataxid.Bias(...) — "
        "dict inputs are no longer supported as of v0.3.0.",
        param="bias",
    )


# ---------------------------------------------------------------------------
# Imputation aggregation helpers
# ---------------------------------------------------------------------------

def _mode_fn(x: np.ndarray) -> Any:
    """Most frequent value, ignoring NaN."""
    if pd.isna(x).all():
        return np.nan
    x_notna = x[~pd.isna(x)]
    values, counts = np.unique(x_notna, return_counts=True)
    return values[np.argmax(counts)]


def _mean_fn(x: np.ndarray) -> float:
    """Arithmetic mean, ignoring NaN."""
    if pd.isna(x).all():
        return np.nan
    return float(np.nanmean(x[~pd.isna(x)]))


def _median_fn(x: np.ndarray) -> float:
    """Median, ignoring NaN."""
    if pd.isna(x).all():
        return np.nan
    return float(np.nanmedian(x[~pd.isna(x)]))


def _list_fn(x: np.ndarray) -> np.ndarray:
    """Return all draws as-is."""
    return np.array(x)


_PICK_FN_MAP: dict[str, Callable] = {
    "mode": _mode_fn,
    "mean": _mean_fn,
    "median": _median_fn,
    "all": _list_fn,
}


def _resolve_pick(
    pick: str | Callable,
) -> tuple[Callable, bool]:
    """Return ``(func, is_list_pick)``."""
    if callable(pick) and not isinstance(pick, str):
        return pick, False
    if pick not in _PICK_FN_MAP:
        raise InvalidRequestError(
            f"Unknown pick {pick!r}. Choose from {list(_PICK_FN_MAP)}",
            param="pick",
        )
    return _PICK_FN_MAP[pick], pick == "all"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _infer_parent_key(
    foreign_key: str,
    parent: pd.DataFrame,
) -> str:
    """Infer parent_key from foreign_key if the column exists in parent."""
    if foreign_key in parent.columns:
        return foreign_key
    raise InvalidRequestError(
        f"parent_key not provided and column '{foreign_key}' not found in parent. "
        f"Specify parent_key explicitly when FK and PK column names differ.",
        param="parent_key",
    )


def _validate_context_params(
    data: pd.DataFrame,
    parent: pd.DataFrame | None,
    parent_encoding_types: dict[str, str] | None,
    foreign_key: str | None,
    parent_key: str | None,
) -> None:
    if parent_encoding_types is not None and parent is None:
        raise InvalidRequestError(
            "parent_encoding_types was provided without parent.",
            param="parent_encoding_types",
        )

    if parent_key is not None and parent is None:
        raise InvalidRequestError(
            "parent_key was provided without parent.",
            param="parent_key",
        )

    if foreign_key is not None and foreign_key not in data.columns:
        raise InvalidRequestError(
            f"Column '{foreign_key}' not found in data.",
            param="foreign_key",
        )

    if (
        parent_key is not None
        and parent is not None
        and parent_key not in parent.columns
    ):
        raise InvalidRequestError(
            f"Column '{parent_key}' not found in parent.",
            param="parent_key",
        )

    if parent is not None and foreign_key is None:
        if len(parent) != len(data):
            raise InvalidRequestError(
                f"Row count mismatch: parent has {len(parent)} rows, "
                f"data has {len(data)}. In flat context mode rows must be "
                f"aligned 1:1. For 1:N joins use foreign_key.",
                param="parent",
            )


class Model:
    """A Dataxid model. Trains on your data locally, generates synthetic data via API."""

    def __init__(
        self,
        model_id: str,
        client: DataxidClient,
        encoder: Encoder,
        data: pd.DataFrame,
        config: ModelConfig,
        parent: pd.DataFrame | None = None,
    ):
        self.id = model_id
        self.status: str = "training"
        self._client = client
        self._encoder = encoder
        self._data = data
        self._parent = parent
        self._config = config
        self._n_samples: int | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.stopped_early: bool = False
        self._best_encoder_state: bytes | None = None
        self._encoder_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    @property
    def is_sequential(self) -> bool:
        """Whether this model was trained on sequential (time-ordered) data."""
        return self._encoder.is_sequential

    @property
    def has_context(self) -> bool:
        """Whether this model uses a parent (context) table."""
        return self._encoder.has_context

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        n_samples: int | None = None,
        config: dict[str, Any] | ModelConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        parent: pd.DataFrame | None = None,
        parent_encoding_types: dict[str, str] | None = None,
        foreign_key: str | None = None,
        parent_key: str | None = None,
    ) -> Model:
        """
        Create a model: analyze data locally, register via API, and train.

        Args:
            data: Training DataFrame (stays local — only embeddings sent to API)
            n_samples: Number of synthetic rows (stored for generate())
            config: Training config — ModelConfig instance or plain dict
            api_key: Override dataxid.api_key
            base_url: Override dataxid.base_url
            parent: Parent table for context-aware generation
            parent_encoding_types: Encoding overrides for parent columns
            foreign_key: FK column in data linking rows to parent (enables sequential mode)
            parent_key: PK column in parent table (inferred from foreign_key if same name)

        Returns:
            Trained Model ready for generate()

        Raises:
            InvalidRequestError: If any user input fails validation.
                Inherits from :class:`ValueError`.
        """
        if not isinstance(data, pd.DataFrame):
            raise InvalidRequestError(
                f"data must be a pandas DataFrame, got {type(data).__name__}",
                param="data",
            )
        if n_samples is not None and (
            not isinstance(n_samples, int)
            or isinstance(n_samples, bool)
            or n_samples <= 0
        ):
            raise InvalidRequestError(
                f"n_samples must be a positive integer or None, got {n_samples!r}",
                param="n_samples",
            )
        if parent is not None and not isinstance(parent, pd.DataFrame):
            raise InvalidRequestError(
                f"parent must be a pandas DataFrame or None, got {type(parent).__name__}",
                param="parent",
            )
        if foreign_key is not None and not isinstance(foreign_key, str):
            raise InvalidRequestError(
                f"foreign_key must be a string or None, got {type(foreign_key).__name__}",
                param="foreign_key",
            )
        if parent_key is not None and not isinstance(parent_key, str):
            raise InvalidRequestError(
                f"parent_key must be a string or None, got {type(parent_key).__name__}",
                param="parent_key",
            )

        config = _resolve_config(config)

        if parent is not None and foreign_key is not None and parent_key is None:
            parent_key = _infer_parent_key(foreign_key, parent)

        _validate_context_params(
            data, parent, parent_encoding_types, foreign_key, parent_key,
        )
        _validate_encoding_types(parent_encoding_types, "parent_encoding_types")
        http = DataxidClient(api_key=api_key, base_url=base_url)

        privacy = config.privacy

        if config.seed is not None:
            _set_seed(config.seed)

        # --- 1. Analyze locally ---
        logger.info("Analyzing data locally...")
        encoder = Encoder(
            embedding_dim=config.embedding_dim,
            model_size=config.model_size,
            privacy_enabled=privacy.enabled,
            privacy_noise=privacy.noise,
            protect_rare=privacy.protect_rare,
        )
        metadata = encoder.analyze(
            data,
            encoding_types=config.encoding_types,
            parent=parent,
            parent_encoding_types=parent_encoding_types,
            foreign_key=foreign_key,
            parent_key=parent_key,
        )

        if encoder.is_sequential and parent is None:
            raise InvalidRequestError(
                "Sequential mode requires parent. Groups in "
                "foreign_key have multiple rows, which triggers "
                "entity-level training.",
                param="parent",
            )

        encoder.prepare(data, parent=parent, parent_key=parent_key)

        # --- 2. Register model on API ---
        effective_embedding_dim = config.embedding_dim
        if encoder.is_sequential and encoder._backend.encoder.ctx_compressor is not None:
            effective_embedding_dim = encoder._backend.encoder.ctx_compressor.dim_output

        logger.info("Creating model on API...")
        resp = http.post("/v1/models", json={
            "metadata": metadata,
            "config": {
                "embedding_dim": effective_embedding_dim,
                "model_size": config.model_size,
                "batch_size": config.batch_size,
                "max_epochs": config.max_epochs,
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
            parent=parent,
        )
        model._n_samples = n_samples

        # --- 3. Train ---
        train_frozen(
            model,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            early_stop_patience=config.early_stop_patience,
            val_split=config.val_split,
        )

        return model

    def generate(
        self,
        n_samples: int | None = None,
        *,
        synthetic: Synthetic | None = None,
        conditions: pd.DataFrame | None = None,
        parent: pd.DataFrame | None = None,
        seed: int | Any = _UNSET,
        distribution: Distribution | None = None,
        bias: Bias | None = None,
        diversity: float | Any = _UNSET,
        rare_cutoff: float | Any = _UNSET,
        rare_strategy: RareStrategy | None | Any = _UNSET,
    ) -> pd.DataFrame:
        """
        Generate synthetic data from the trained model.

        Flat mode: generates n_samples independent rows.
        Sequential mode: generates variable-length sequences per entity,
        then flattens to a single DataFrame with entity key column.

        Args:
            n_samples: Number of rows (flat) or entities (sequential) to generate.
                For flat mode with ``conditions``, must be omitted or match
                ``len(conditions)``; otherwise defaults to the training data size.
            synthetic: Optional :class:`Synthetic` preset bundling scalar
                generation-time tuning (``n``, ``seed``, ``diversity``,
                ``rare_cutoff``, ``rare_strategy``). Keyword arguments passed
                explicitly to ``generate`` always override fields of the
                preset. ``conditions``, ``parent``, ``distribution`` and
                ``bias`` are orthogonal and stay as separate arguments.
            conditions: DataFrame for conditional generation. Columns present in
                conditions are kept as-is; remaining columns are generated.
                Flat: one row per output row.
                Sequential: long-format with the foreign key column linking
                events to entities (steps beyond condition length are free).
            parent: Parent table for generation. Falls back to training parent.
            seed: Random seed for reproducible generation. None = non-deterministic.
            distribution: Override the empirical distribution of a single
                categorical column. See :class:`Distribution`.
            bias: Bias-correction config for statistical parity across
                sensitive groups. See :class:`Bias`.
            diversity: Sampling diversity. ``1.0`` = model distribution,
                ``<1.0`` = sharper (more deterministic), ``>1.0`` = flatter
                (more diverse). Must be > 0.
            rare_cutoff: Nucleus sampling cutoff in ``(0, 1]``. ``1.0`` disables
                nucleus sampling (uses the full distribution). Lower values
                restrict sampling to the smallest set of tokens whose cumulative
                probability mass reaches ``rare_cutoff`` — trimming the rare tail.
            rare_strategy: How to handle rare (infrequent) categorical values
                during generation. When ``None`` (default), inherits from the
                training-time setting on :class:`Privacy` (``privacy.rare_strategy``).
                Explicit values override the config.
                - ``"mask"``: Rare values in categorical columns may appear as
                  the literal token ``"<protected>"``. Preserves the original
                  distribution, including the presence of rare values.
                - ``"sample"``: Suppress the ``"<protected>"`` token entirely.
                  Rare slots are filled by sampling from the frequent
                  categories. Use when you need clean string values in
                  categorical columns.

        Returns:
            DataFrame with synthetic data

        Raises:
            InvalidRequestError: For any user-input error, including: both
                ``n_samples`` and ``conditions`` are provided with
                disagreeing sizes; ``synthetic.n`` conflicts with
                ``n_samples``; ``bias`` is used in sequential mode;
                ``diversity`` / ``rare_cutoff`` / ``rare_strategy`` are
                out of range; ``synthetic`` is not a :class:`Synthetic`
                instance. Inherits from :class:`ValueError` for backward
                compatibility.

        Note:
            For filling missing values, use :meth:`impute` — do not attempt
            to emulate imputation through ``generate`` + ``conditions``.
        """
        if n_samples is not None and (
            not isinstance(n_samples, int)
            or isinstance(n_samples, bool)
            or n_samples <= 0
        ):
            raise InvalidRequestError(
                f"n_samples must be a positive integer or None, got {n_samples!r}",
                param="n_samples",
            )
        if conditions is not None and not isinstance(conditions, pd.DataFrame):
            raise InvalidRequestError(
                f"conditions must be a pandas DataFrame or None, got "
                f"{type(conditions).__name__}",
                param="conditions",
            )
        if parent is not None and not isinstance(parent, pd.DataFrame):
            raise InvalidRequestError(
                f"parent must be a pandas DataFrame or None, got "
                f"{type(parent).__name__}",
                param="parent",
            )
        if synthetic is not None and not isinstance(synthetic, Synthetic):
            raise InvalidRequestError(
                "synthetic must be a Synthetic instance or None, got "
                f"{type(synthetic).__name__}. Use dataxid.Synthetic(...).",
                param="synthetic",
            )

        n_effective = _merge_n(n_samples, synthetic)
        seed_effective = _merge_field(seed, synthetic, "seed", None)
        diversity_effective = _merge_field(diversity, synthetic, "diversity", 1.0)
        rare_cutoff_effective = _merge_field(rare_cutoff, synthetic, "rare_cutoff", 1.0)
        rare_strategy_effective = _merge_field(
            rare_strategy, synthetic, "rare_strategy", None
        )

        if (
            not isinstance(diversity_effective, (int, float))
            or isinstance(diversity_effective, bool)
            or diversity_effective <= 0
        ):
            raise InvalidRequestError(
                f"diversity must be > 0, got {diversity_effective!r}",
                param="diversity",
            )
        if (
            not isinstance(rare_cutoff_effective, (int, float))
            or isinstance(rare_cutoff_effective, bool)
            or not (0 < rare_cutoff_effective <= 1.0)
        ):
            raise InvalidRequestError(
                f"rare_cutoff must be in (0, 1], got {rare_cutoff_effective!r}",
                param="rare_cutoff",
            )
        if (
            rare_strategy_effective is not None
            and rare_strategy_effective not in _RARE_STRATEGY_VALUES
        ):
            raise InvalidRequestError(
                f"rare_strategy must be one of {_RARE_STRATEGY_VALUES}, "
                f"got {rare_strategy_effective!r}",
                param="rare_strategy",
            )
        if seed_effective is not None and (
            not isinstance(seed_effective, int) or isinstance(seed_effective, bool)
        ):
            raise InvalidRequestError(
                f"seed must be an integer or None, got {seed_effective!r}",
                param="seed",
            )

        return self._generate_core(
            n_samples=n_effective,
            conditions=conditions,
            parent=parent,
            seed=seed_effective,
            distribution=_normalize_distribution(distribution),
            imputation=None,
            bias=_normalize_bias(bias),
            diversity=diversity_effective,
            rare_cutoff=rare_cutoff_effective,
            rare_strategy=rare_strategy_effective,
        )

    def _generate_core(
        self,
        n_samples: int | None = None,
        conditions: pd.DataFrame | None = None,
        parent: pd.DataFrame | None = None,
        seed: int | None = None,
        distribution: dict | None = None,
        imputation: dict | None = None,
        bias: dict | None = None,
        diversity: float = 1.0,
        rare_cutoff: float = 1.0,
        rare_strategy: RareStrategy | None = None,
    ) -> pd.DataFrame:
        """Shared generation path used by :meth:`generate` and :meth:`impute`.

        The ``imputation`` argument is only set by :meth:`impute` — public
        callers should go through :meth:`generate` instead.

        Invariant (enforced by every public caller):
            * ``diversity > 0``
            * ``0 < rare_cutoff <= 1``
            * ``rare_strategy`` is one of ``_RARE_STRATEGY_VALUES`` or ``None``
            * ``seed`` is an ``int`` or ``None``
        """
        if rare_strategy is None:
            rare_strategy = self._config.privacy.rare_strategy

        ctx = parent if parent is not None else self._parent

        if self._encoder.is_sequential:
            if bias:
                raise InvalidRequestError(
                    "bias is not supported in sequential mode",
                    param="bias",
                )
            return self._generate_sequential(
                n_samples=n_samples, conditions=conditions, parent=ctx,
                seed=seed, distribution=distribution, imputation=imputation,
                diversity=diversity, rare_cutoff=rare_cutoff,
                rare_strategy=rare_strategy,
            )

        # flat mode: conditions row count = output row count
        if conditions is not None and n_samples is not None and n_samples != len(conditions):
            raise InvalidRequestError(
                f"n_samples ({n_samples}) conflicts with conditions length "
                f"({len(conditions)}). When conditions is provided, omit n_samples "
                f"or set it to len(conditions).",
                param="n_samples",
            )

        n = (
            len(conditions) if conditions is not None
            else n_samples or getattr(self, "_n_samples", None) or len(self._data)
        )
        logger.info("Generating %d synthetic rows...", n)
        embedding = self._encoder._generation_embedding(
            n_samples=n, conditions=None, parent=ctx,
        )

        column_stats = self._encoder.column_stats
        features = self._encoder.features

        impute_cols = imputation.get("columns", []) if imputation else []

        # --- fixed_probs ---
        rare_probs = (
            compute_fixed_probs(column_stats, mode=rare_strategy)
            if column_stats else {}
        )
        impute_probs = (
            compute_imputation_probs(column_stats, impute_cols)
            if impute_cols and column_stats else {}
        )
        dist_probs = (
            compute_distribution_probs(column_stats, distribution)
            if distribution and column_stats else {}
        )
        all_probs = [p for p in (rare_probs, impute_probs, dist_probs) if p]
        fixed_probs = merge_fixed_probs(*all_probs) if all_probs else None

        # --- fixed_values + column_order ---
        fixed_values = None
        conditions_cols: list[str] = []
        if conditions is not None and column_stats:
            fixed_values = encode_conditions_fixed_values(
                conditions, features, column_stats,
                imputed_columns=impute_cols or None,
            )
            conditions_cols = [c for c in features if c in conditions.columns]

        if bias and column_stats:
            column_order = bias_column_order(
                features, bias,
                conditions_columns=conditions_cols or None,
                distribution=distribution,
                imputed_columns=impute_cols or None,
            )
        elif conditions is not None and column_stats:
            if impute_cols:
                column_order = imputation_column_order(
                    features, impute_cols,
                    conditions_columns=conditions_cols, distribution=distribution,
                )
            else:
                column_order = conditional_column_order(features, conditions_cols, distribution)
        elif distribution:
            column_order = distribution_column_order(features, distribution)
        else:
            column_order = None

        # --- bias payload (resolved column names + integer codes) ---
        bias_payload = None
        if bias and column_stats:
            bias_payload = resolve_bias_payload(column_stats, bias)

        payload: dict[str, Any] = {"embedding": embedding}
        if fixed_probs:
            payload["fixed_probs"] = fixed_probs
        if fixed_values:
            payload["fixed_values"] = fixed_values
        if column_order:
            payload["column_order"] = column_order
        if seed is not None:
            payload["seed"] = seed
        if impute_cols:
            payload["imputation_columns"] = impute_cols
        if bias_payload:
            payload["bias"] = bias_payload
        if diversity != 1.0:
            payload["diversity"] = diversity
        if rare_cutoff != 1.0:
            payload["rare_cutoff"] = rare_cutoff

        resp = self._client.post(f"/v1/models/{self.id}/generate", json=payload)

        raw_codes = resp["data"]["codes"]
        df = decode_columns(
            raw_codes, features, column_stats,
            rare_strategy=rare_strategy,
        )

        fk_col = self._encoder._foreign_key
        if fk_col and ctx is not None and self._encoder._parent_key:
            pk_col = self._encoder._parent_key
            if pk_col in ctx.columns:
                key_values = ctx[pk_col].values
                df.insert(0, fk_col, [
                    key_values[i] if i < len(key_values) else i
                    for i in range(len(df))
                ])

        self.status = "ready"
        logger.info("Generated %d rows, %d columns", len(df), len(df.columns))
        return df

    def _generate_sequential(
        self,
        n_samples: int | None = None,
        conditions: pd.DataFrame | None = None,
        parent: pd.DataFrame | None = None,
        seed: int | None = None,
        distribution: dict | None = None,
        imputation: dict | None = None,
        diversity: float = 1.0,
        rare_cutoff: float = 1.0,
        rare_strategy: RareStrategy = "mask",
    ) -> pd.DataFrame:
        """Sequential generation: context embedding → autoregressive decode → flatten.

        When ``conditions`` is provided it must be long-format (one row per
        time step) with the foreign-key column linking events to entities.
        Condition steps are encoded per-entity and sent as 2D fixed_values;
        steps beyond the condition length are freely generated.
        """
        from dataxid.encoder._nn import RIDX_PREFIX, SIDX_PREFIX, SLEN_PREFIX

        if parent is not None and self._encoder.has_context:
            n_entities = len(parent)
        else:
            n_entities = n_samples or getattr(self, "_n_samples", None) or 100

        logger.info("Generating sequences for %d entities (max_len=%d)...",
                     n_entities, self._encoder.seq_len_max)

        embedding = self._encoder._generation_embedding(
            n_samples=n_entities, parent=parent,
        )

        column_stats = self._encoder.column_stats
        features = self._encoder.features
        context_key = self._encoder._foreign_key
        impute_cols = imputation.get("columns", []) if imputation else []

        # --- fixed_probs ---
        rare_probs = (
            compute_fixed_probs(column_stats, mode=rare_strategy)
            if column_stats else {}
        )
        impute_probs = (
            compute_imputation_probs(column_stats, impute_cols)
            if impute_cols and column_stats else {}
        )
        dist_probs = (
            compute_distribution_probs(column_stats, distribution)
            if distribution and column_stats else {}
        )
        all_probs = [p for p in (rare_probs, impute_probs, dist_probs) if p]
        fixed_probs = merge_fixed_probs(*all_probs) if all_probs else None

        # --- fixed_values + column_order ---
        fixed_values = None
        if conditions is not None and column_stats and context_key:
            if context_key not in conditions.columns:
                raise InvalidRequestError(
                    f"Sequential conditions must contain the foreign key column "
                    f"'{context_key}'. Got columns: {list(conditions.columns)}",
                    param="conditions",
                )
            pk_col = self._encoder._parent_key
            training_parent = self._parent
            entity_order = (
                training_parent[pk_col].values
                if training_parent is not None and pk_col and pk_col in training_parent.columns
                else None
            )
            fixed_values, _ = encode_sequential_conditions_fixed_values(
                conditions, features, column_stats, context_key, n_entities,
                entity_order=entity_order,
                imputed_columns=impute_cols or None,
            )
            conditions_cols = [
                c for c in features
                if c in conditions.columns and c != context_key
            ]
            if impute_cols:
                column_order = imputation_column_order(
                    features, impute_cols,
                    conditions_columns=conditions_cols, distribution=distribution,
                )
            else:
                column_order = conditional_column_order(features, conditions_cols, distribution)
        elif distribution:
            column_order = distribution_column_order(features, distribution)
        else:
            column_order = None

        payload: dict[str, Any] = {
            "embedding": embedding,
            "is_sequential": True,
            "seq_len_max": self._encoder.seq_len_max,
        }
        if fixed_probs:
            payload["fixed_probs"] = fixed_probs
        if fixed_values:
            payload["fixed_values"] = fixed_values
        if column_order:
            payload["column_order"] = column_order
        if seed is not None:
            payload["seed"] = seed
        if impute_cols:
            payload["imputation_columns"] = impute_cols
        if diversity != 1.0:
            payload["diversity"] = diversity
        if rare_cutoff != 1.0:
            payload["rare_cutoff"] = rare_cutoff

        resp = self._client.post(f"/v1/models/{self.id}/generate", json=payload)
        raw_codes = resp["data"]["codes"]

        pos_prefixes = (SIDX_PREFIX, SLEN_PREFIX, RIDX_PREFIX)
        ridx_cols = [k for k in raw_codes if k.startswith(RIDX_PREFIX)]

        if ridx_cols:
            ridx_lists = raw_codes[ridx_cols[0]]
            seq_lens = []
            for entity_ridx in ridx_lists:
                if entity_ridx[0] == 0:
                    seq_lens.append(0)
                    continue
                length = len(entity_ridx)
                for t in range(1, len(entity_ridx)):
                    if entity_ridx[t] == 0:
                        length = t
                        break
                seq_lens.append(length)
        else:
            first_key = next(iter(raw_codes))
            seq_lens = [len(raw_codes[first_key][i]) for i in range(n_entities)]

        data_cols = {k: v for k, v in raw_codes.items()
                     if not k.startswith(pos_prefixes)}

        rows: list[dict[str, int]] = []
        entity_indices: list[int] = []
        for i in range(n_entities):
            for t in range(seq_lens[i]):
                row = {sub_col: vals[i][t] for sub_col, vals in data_cols.items()}
                rows.append(row)
                entity_indices.append(i)

        if not rows:
            return pd.DataFrame(columns=self._encoder.features)

        df_flat = pd.DataFrame(rows)
        features = self._encoder.features
        df = decode_columns(
            df_flat.to_dict(orient="list"), features, column_stats,
            rare_strategy=rare_strategy,
        )

        fk_col = self._encoder._foreign_key
        if fk_col:
            if (parent is not None
                    and self._encoder._parent_key
                    and self._encoder._parent_key in parent.columns):
                key_values = parent[self._encoder._parent_key].values
                df[fk_col] = [
                    key_values[idx] if idx < len(key_values) else idx
                    for idx in entity_indices
                ]
            else:
                df[fk_col] = entity_indices
            cols = [fk_col] + [c for c in df.columns if c != fk_col]
            df = df[cols]

        self.status = "ready"
        logger.info("Generated %d event rows for %d entities", len(df), n_entities)
        return df

    def impute(
        self,
        X: pd.DataFrame,
        parent: pd.DataFrame | None = None,
        trials: int = 1,
        pick: Literal["mode", "mean", "median", "all"] | Callable[[np.ndarray], Any] = "mode",
        **kwargs,
    ) -> pd.DataFrame:
        """Fill missing values in *X* using the trained model.

        Each ``NULL`` cell is replaced by the model's prediction, conditioned
        on the non-NULL cells in the same row (flat) or entity (sequential).
        Non-NULL values are always preserved.

        Args:
            X: Data with missing values.

                * **Flat models** — one row per sample, columns a subset of
                  the training features.
                * **Sequential models** — long-format with the foreign-key
                  column linking events to entities.

            parent: Parent / context table. Falls back to the training parent.
            trials: How many independent generation passes to run per row.
                When ``trials > 1`` the results are aggregated cell-wise
                using *pick*.
            pick: Aggregation strategy across trials.

                * ``"mode"`` — most frequent value (default, works for all dtypes).
                * ``"mean"`` — arithmetic mean (numeric columns only).
                * ``"median"`` — median (numeric columns only).
                * ``"all"`` — return every trial as a Python list per cell.
                * Any ``Callable[[np.ndarray], scalar]``.

            **kwargs: Forwarded to :meth:`generate` (e.g. ``seed``,
                ``distribution``).

        Returns:
            DataFrame shaped like *X* with NULL cells filled.

        Example::

            model = dataxid.train(df)
            df_dirty = df.copy()
            df_dirty.loc[0:10, "age"] = None
            df_clean = model.impute(df_dirty)

        Raises:
            InvalidRequestError: If any user input fails validation.
                Inherits from :class:`ValueError`.
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidRequestError(
                f"X must be a pandas DataFrame, got {type(X).__name__}",
                param="X",
            )
        if parent is not None and not isinstance(parent, pd.DataFrame):
            raise InvalidRequestError(
                f"parent must be a pandas DataFrame or None, got "
                f"{type(parent).__name__}",
                param="parent",
            )
        if (
            not isinstance(trials, int)
            or isinstance(trials, bool)
            or trials <= 0
        ):
            raise InvalidRequestError(
                f"trials must be a positive integer, got {trials!r}",
                param="trials",
            )

        X_df = X.copy().reset_index(drop=True)

        fk = self._encoder._foreign_key
        feature_cols = [c for c in X_df.columns if c != fk]
        null_cols = [c for c in feature_cols if c in X_df.columns and X_df[c].isna().any()]
        imputation_config = {"columns": null_cols}

        if "distribution" in kwargs:
            kwargs["distribution"] = _normalize_distribution(kwargs["distribution"])
        if "bias" in kwargs:
            kwargs["bias"] = _normalize_bias(kwargs["bias"])

        synthetic = kwargs.pop("synthetic", None)
        if synthetic is not None and not isinstance(synthetic, Synthetic):
            raise InvalidRequestError(
                "synthetic must be a Synthetic instance or None, got "
                f"{type(synthetic).__name__}. Use dataxid.Synthetic(...).",
                param="synthetic",
            )
        if synthetic is not None:
            if "n_samples" in kwargs:
                kwargs["n_samples"] = _merge_n(kwargs["n_samples"], synthetic)
            elif synthetic.n is not None:
                kwargs["n_samples"] = synthetic.n
            for field_name in ("seed", "diversity", "rare_cutoff", "rare_strategy"):
                if field_name not in kwargs:
                    kwargs[field_name] = getattr(synthetic, field_name)
                # else: explicit kwarg wins over preset

        if "diversity" in kwargs:
            div = kwargs["diversity"]
            if (
                not isinstance(div, (int, float))
                or isinstance(div, bool)
                or div <= 0
            ):
                raise InvalidRequestError(
                    f"diversity must be > 0, got {div!r}", param="diversity"
                )
        if "rare_cutoff" in kwargs:
            rc = kwargs["rare_cutoff"]
            if (
                not isinstance(rc, (int, float))
                or isinstance(rc, bool)
                or not (0 < rc <= 1.0)
            ):
                raise InvalidRequestError(
                    f"rare_cutoff must be in (0, 1], got {rc!r}",
                    param="rare_cutoff",
                )
        if kwargs.get("rare_strategy") is not None and (
            kwargs["rare_strategy"] not in _RARE_STRATEGY_VALUES
        ):
            raise InvalidRequestError(
                f"rare_strategy must be one of {_RARE_STRATEGY_VALUES}, "
                f"got {kwargs['rare_strategy']!r}",
                param="rare_strategy",
            )
        if kwargs.get("seed") is not None and (
            not isinstance(kwargs["seed"], int) or isinstance(kwargs["seed"], bool)
        ):
            raise InvalidRequestError(
                f"seed must be an integer or None, got {kwargs['seed']!r}",
                param="seed",
            )

        def _single_draw() -> pd.DataFrame:
            return self._generate_core(
                conditions=X_df,
                parent=parent,
                imputation=imputation_config,
                **kwargs,
            )

        if trials == 1:
            result = _single_draw()
        else:
            pick_fn, is_list_pick = _resolve_pick(pick)
            all_trials = [_single_draw() for _ in range(trials)]

            result = all_trials[0].copy()
            for col in feature_cols:
                if col not in result.columns:
                    continue
                col_values = np.column_stack([d[col].values for d in all_trials])
                if is_list_pick:
                    result[col] = [pick_fn(row) for row in col_values]
                else:
                    result[col] = np.apply_along_axis(pick_fn, 1, col_values)

        # Preserve original non-NULL values from seed data.
        result = result.reset_index(drop=True)
        X_aligned = X_df.reset_index(drop=True)

        if fk is None:
            # Flat mode: 1:1 row correspondence
            n = min(len(result), len(X_aligned))
            for col in feature_cols:
                if col in result.columns and col in X_aligned.columns:
                    mask = X_aligned[col].iloc[:n].notna()
                    result.loc[mask[mask].index, col] = X_aligned.loc[mask[mask].index, col]
        else:
            # Sequential mode: align per entity using the foreign key
            orig_groups = X_aligned.groupby(fk, sort=False)
            imp_groups = result.groupby(fk, sort=False)
            for eid, orig_grp in orig_groups:
                if eid not in imp_groups.groups:
                    continue
                imp_idx = imp_groups.groups[eid]
                n_overlap = min(len(orig_grp), len(imp_idx))
                for col in feature_cols:
                    if col not in result.columns or col not in X_aligned.columns:
                        continue
                    orig_vals = orig_grp[col].iloc[:n_overlap]
                    known = orig_vals.notna()
                    if not known.any():
                        continue
                    target_idx = imp_idx[:n_overlap][known.values]
                    result.loc[target_idx, col] = orig_vals[known].values

        return result

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

