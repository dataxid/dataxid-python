# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
ModelConfig — typed training configuration for Model.create().

Replaces the untyped config dict with IDE-discoverable, validated fields.
Plain dicts are accepted for backwards-compatible ergonomics; see
:func:`_resolve_config` for the normalization rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from dataxid.encoder._ports import EncodingType
from dataxid.exceptions import InvalidRequestError

RareStrategy = Literal["mask", "sample"]
_RARE_STRATEGIES: tuple[str, ...] = ("mask", "sample")
_MODEL_SIZES: tuple[str, ...] = ("small", "medium", "large")
_VALID_ENCODING_TYPES: frozenset[str] = frozenset(et.value for et in EncodingType)


def _validate_encoding_types(value: object, param: str) -> None:
    """Validate a ``dict[str, str | EncodingType]`` encoding-override mapping.

    Accepts ``None``. Raises :class:`InvalidRequestError` (param=``param``)
    if the value is not a dict, has non-string keys, or contains a value
    outside the :class:`EncodingType` enum.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise InvalidRequestError(
            f"{param} must be a dict or None, got {type(value).__name__}",
            param=param,
        )
    for key, val in value.items():
        if not isinstance(key, str):
            raise InvalidRequestError(
                f"{param} keys must be strings, got {type(key).__name__} ({key!r})",
                param=param,
            )
        if isinstance(val, EncodingType):
            continue
        if isinstance(val, str) and val in _VALID_ENCODING_TYPES:
            continue
        raise InvalidRequestError(
            f"{param}[{key!r}] must be one of {sorted(_VALID_ENCODING_TYPES)} "
            f"or an EncodingType enum member, got {val!r}",
            param=param,
        )


@dataclass(frozen=True)
class Synthetic:
    """
    Generation-time tuning preset for :meth:`Model.generate`.

    Bundles scalar hyperparameters that control *how* synthetic rows are
    produced. Structured constraints (bias-correction, distribution
    override) are passed as separate arguments to :meth:`Model.generate`
    and are intentionally **not** fields of this class — they are
    orthogonal concepts.

    Example::

        from dataxid import Synthetic

        preset = Synthetic(diversity=0.8, rare_cutoff=0.95, seed=42)
        df = model.generate(synthetic=preset, n=1000)

    Override rule:
        When passed via ``Model.generate(synthetic=...)``, an explicit
        keyword argument to ``generate`` always wins over the preset's
        field. Fields left at their default values do not override any
        ``generate`` default.

    Args:
        n: Number of rows (flat) / entities (sequential) to generate.
            ``None`` means "use training data size".
        seed: PRNG seed for deterministic generation. ``None`` means
            non-deterministic.
        diversity: Softmax temperature; lower is more peaked / less
            diverse. Must be > 0.
        rare_cutoff: Nucleus-sampling mass cutoff. Must be in ``(0, 1]``.
        rare_strategy: How rare categories are rendered during generation.
            ``None`` means "fall back to ``ModelConfig.privacy.rare_strategy``".

    Raises:
        InvalidRequestError: If ``diversity <= 0``, ``rare_cutoff`` is
            outside ``(0, 1]``, or ``n`` is not a positive integer.
            Inherits from :class:`ValueError` for Python-idiomatic
            ``except ValueError`` handling.
    """

    n: int | None = None
    seed: int | None = None
    diversity: float = 1.0
    rare_cutoff: float = 1.0
    rare_strategy: RareStrategy | None = None

    def __post_init__(self) -> None:
        if self.n is not None and (
            not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0
        ):
            raise InvalidRequestError(
                f"Synthetic.n must be a positive integer or None, got {self.n!r}",
                param="n",
            )
        if self.diversity <= 0:
            raise InvalidRequestError(
                f"Synthetic.diversity must be > 0, got {self.diversity!r}",
                param="diversity",
            )
        if not (0 < self.rare_cutoff <= 1):
            raise InvalidRequestError(
                f"Synthetic.rare_cutoff must be in (0, 1], got {self.rare_cutoff!r}",
                param="rare_cutoff",
            )
        if self.rare_strategy is not None and self.rare_strategy not in _RARE_STRATEGIES:
            raise InvalidRequestError(
                f"Synthetic.rare_strategy must be 'mask', 'sample', or None, "
                f"got {self.rare_strategy!r}",
                param="rare_strategy",
            )


@dataclass(frozen=True)
class Bias:
    """
    Bias-correction config for statistical parity across sensitive groups.

    Used with :meth:`Model.generate` to enforce that the predicted
    distribution of ``target`` is independent of the group membership
    defined by ``sensitive`` columns. Sequential models are not supported.

    Example::

        from dataxid import Bias

        bias = Bias(
            target="income",
            sensitive=["gender", "race"],
        )

    Args:
        target: Name of the target (outcome) column. Must be a non-empty
            string and a categorical column in the training data.
        sensitive: Protected-attribute columns. Must be a non-empty list
            of categorical column names, and must not include ``target``.

    Raises:
        InvalidRequestError: If ``target`` is empty / not a string,
            ``sensitive`` is empty, contains non-strings, or includes
            ``target``. Inherits from :class:`ValueError`.
    """

    target: str
    sensitive: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise InvalidRequestError(
                f"Bias.target must be a non-empty string, got {self.target!r}",
                param="target",
            )
        if not self.sensitive:
            raise InvalidRequestError(
                "Bias.sensitive must be a non-empty list",
                param="sensitive",
            )
        if any(not isinstance(s, str) or not s.strip() for s in self.sensitive):
            raise InvalidRequestError(
                "Bias.sensitive entries must be non-empty strings",
                param="sensitive",
            )
        if self.target in self.sensitive:
            raise InvalidRequestError(
                f"Bias.target ({self.target!r}) cannot appear in sensitive",
                param="target",
            )


@dataclass(frozen=True)
class Distribution:
    """
    Override the empirical distribution of a single categorical column.

    Used with :meth:`Model.generate` and :func:`synthesize` to force the
    output distribution of one column to match a target probability map.
    All other columns are generated by the model normally.

    Example::

        from dataxid import Distribution

        dist = Distribution(
            column="gender",
            probabilities={"M": 0.7, "F": 0.3},
        )

    Args:
        column: Target column name. Must be a non-empty string.
        probabilities: Category → probability mapping. Must be non-empty
            with non-negative values. Values are normalized downstream;
            they need not sum to exactly 1.0.

    Raises:
        InvalidRequestError: If ``column`` is empty / not a string,
            ``probabilities`` is empty, contains non-string keys, or has
            negative / non-finite values. Inherits from
            :class:`ValueError`.
    """

    column: str
    probabilities: dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise InvalidRequestError(
                f"Distribution.column must be a non-empty string, got {self.column!r}",
                param="column",
            )
        if not self.probabilities:
            raise InvalidRequestError(
                "Distribution.probabilities must be non-empty",
                param="probabilities",
            )
        for key, value in self.probabilities.items():
            if not isinstance(key, str):
                raise InvalidRequestError(
                    f"Distribution.probabilities keys must be strings, got "
                    f"{type(key).__name__} ({key!r}). Numeric category values "
                    f"must be passed as their string representation, e.g. "
                    f"{{'25': 0.5}} not {{25: 0.5}}.",
                    param="probabilities",
                )
            if not key.strip():
                raise InvalidRequestError(
                    "Distribution.probabilities keys must be non-empty / "
                    f"non-whitespace strings, got {key!r}",
                    param="probabilities",
                )
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidRequestError(
                    f"Distribution.probabilities[{key!r}] must be a real number, "
                    f"got {value!r}",
                    param="probabilities",
                )
            if math.isnan(value) or math.isinf(value):
                raise InvalidRequestError(
                    f"Distribution.probabilities[{key!r}] must be finite, "
                    f"got {value!r}",
                    param="probabilities",
                )
            if value < 0:
                raise InvalidRequestError(
                    f"Distribution.probabilities[{key!r}] must be non-negative, "
                    f"got {value!r}",
                    param="probabilities",
                )


@dataclass
class Privacy:
    """
    Privacy settings applied at training time.

    Two independent protections:

    * **Rare-category protection** (``protect_rare`` + ``rare_strategy``)
      runs during analysis — infrequent values are hidden from the model
      before it ever sees them.
    * **Embedding noise** (``enabled`` + ``noise``) adds Gaussian
      perturbation to embeddings before they cross the API boundary.

    Example::

        from dataxid import Privacy

        Privacy(
            protect_rare=True,
            rare_strategy="mask",  # keep <protected> sentinel visible
            enabled=False,
            noise=0.1,
        )

    Args:
        protect_rare: When True, rare categorical values are replaced
            with the ``<protected>`` sentinel before the vocabulary is
            built. Disable only when the training data contains no
            sensitive rare values.
        rare_strategy: How the ``<protected>`` sentinel is rendered at
            generation time.
            - ``"mask"`` (default): keep ``<protected>`` literal in the
              output, preserving the training distribution and leaving
              an auditable protection trace.
            - ``"sample"``: replace each ``<protected>`` with a random
              draw from the remaining frequent categories.
        enabled: Toggle for embedding noise. When True, Gaussian noise
            is injected into embeddings before they leave the client.
        noise: Standard deviation of the Gaussian noise injected into
            embeddings when ``enabled`` is True.
    """

    protect_rare: bool = True
    rare_strategy: RareStrategy = "mask"
    enabled: bool = False
    noise: float = 0.1

    def __post_init__(self) -> None:
        if self.rare_strategy not in _RARE_STRATEGIES:
            raise InvalidRequestError(
                f"Privacy.rare_strategy must be 'mask' or 'sample', "
                f"got {self.rare_strategy!r}",
                param="rare_strategy",
            )
        if not isinstance(self.noise, (int, float)) or isinstance(self.noise, bool):
            raise InvalidRequestError(
                f"Privacy.noise must be a real number, got {self.noise!r}",
                param="noise",
            )
        if math.isnan(self.noise) or math.isinf(self.noise) or self.noise < 0:
            raise InvalidRequestError(
                f"Privacy.noise must be a non-negative finite number, got {self.noise!r}",
                param="noise",
            )


@dataclass
class ModelConfig:
    """
    Training configuration for Model.create().

    Example::

        import dataxid

        model = dataxid.Model.create(
            data=df,
            config=dataxid.ModelConfig(
                embedding_dim=128,
                model_size="large",
                max_epochs=50,
                privacy=dataxid.Privacy(rare_strategy="mask"),
            ),
        )
    """

    embedding_dim: int = 64
    model_size: Literal["small", "medium", "large"] = "medium"
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 4
    val_split: float = 0.1
    privacy: Privacy = field(default_factory=Privacy)
    encoding_types: dict[str, str] | None = None
    accumulation_steps: int = 1
    learning_rate: float | None = None
    label_smoothing: float = 0.0
    embedding_dropout: float = 0.5
    time_limit_seconds: float = 0.0
    seed: int | None = None
    timeout: float = 14400.0

    def __post_init__(self) -> None:
        if self.model_size not in _MODEL_SIZES:
            raise InvalidRequestError(
                f"ModelConfig.model_size must be one of {_MODEL_SIZES}, "
                f"got {self.model_size!r}",
                param="model_size",
            )
        self._check_positive_int("embedding_dim", self.embedding_dim)
        self._check_positive_int("batch_size", self.batch_size)
        self._check_positive_int("max_epochs", self.max_epochs)
        self._check_positive_int("accumulation_steps", self.accumulation_steps)
        self._check_non_negative_int("early_stop_patience", self.early_stop_patience)
        self._check_unit_interval_open("val_split", self.val_split)
        self._check_unit_interval_closed_left("label_smoothing", self.label_smoothing)
        self._check_unit_interval_closed_left("embedding_dropout", self.embedding_dropout)
        self._check_non_negative_finite("time_limit_seconds", self.time_limit_seconds)
        self._check_positive_finite("timeout", self.timeout)
        if self.learning_rate is not None:
            self._check_positive_finite("learning_rate", self.learning_rate)
        _validate_encoding_types(self.encoding_types, "encoding_types")

    @staticmethod
    def _check_positive_int(name: str, value: object) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InvalidRequestError(
                f"ModelConfig.{name} must be a positive integer, got {value!r}",
                param=name,
            )

    @staticmethod
    def _check_non_negative_int(name: str, value: object) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise InvalidRequestError(
                f"ModelConfig.{name} must be a non-negative integer, got {value!r}",
                param=name,
            )

    @staticmethod
    def _check_unit_interval_open(name: str, value: object) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or math.isnan(value)
            or not (0 < value < 1)
        ):
            raise InvalidRequestError(
                f"ModelConfig.{name} must be in (0, 1), got {value!r}",
                param=name,
            )

    @staticmethod
    def _check_unit_interval_closed_left(name: str, value: object) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or math.isnan(value)
            or not (0 <= value < 1)
        ):
            raise InvalidRequestError(
                f"ModelConfig.{name} must be in [0, 1), got {value!r}",
                param=name,
            )

    @staticmethod
    def _check_non_negative_finite(name: str, value: object) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or math.isnan(value)
            or math.isinf(value)
            or value < 0
        ):
            raise InvalidRequestError(
                f"ModelConfig.{name} must be a non-negative finite number, got {value!r}",
                param=name,
            )

    @staticmethod
    def _check_positive_finite(name: str, value: object) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or math.isnan(value)
            or math.isinf(value)
            or value <= 0
        ):
            raise InvalidRequestError(
                f"ModelConfig.{name} must be a positive finite number, got {value!r}",
                param=name,
            )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        data = dict(self.__dict__)
        data["privacy"] = dict(self.privacy.__dict__)
        return data


def _resolve_config(config: dict | ModelConfig | None) -> ModelConfig:
    """Normalize config input → ``ModelConfig`` instance.

    Dict inputs may nest ``privacy`` as a dict; it is promoted to a
    :class:`Privacy` instance. Returning the dataclass (not a plain dict)
    lets downstream code access fields with attribute syntax and keeps
    :class:`Privacy` as a single source of truth.
    """
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, dict):
        data = dict(config)
        if "privacy" in data:
            privacy = data["privacy"]
            if privacy is None:
                data["privacy"] = Privacy()
            elif isinstance(privacy, dict):
                try:
                    data["privacy"] = Privacy(**privacy)
                except TypeError as exc:
                    raise InvalidRequestError(
                        f"config['privacy'] contains an unknown field: {exc}",
                        param="config",
                    ) from exc
            elif not isinstance(privacy, Privacy):
                raise InvalidRequestError(
                    f"config['privacy'] must be a Privacy instance, dict, or None, "
                    f"got {type(privacy).__name__}",
                    param="config",
                )
        try:
            return ModelConfig(**data)
        except TypeError as exc:
            raise InvalidRequestError(
                f"config contains an unknown field: {exc}",
                param="config",
            ) from exc
    raise InvalidRequestError(
        f"config must be a ModelConfig instance or dict, got {type(config).__name__}",
        param="config",
    )


__all__ = [
    "Bias",
    "Distribution",
    "ModelConfig",
    "Privacy",
    "RareStrategy",
    "Synthetic",
    "_resolve_config",
    "replace",
]
