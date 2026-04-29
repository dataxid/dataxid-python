# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Typed training-configuration contract.

These tests pin the public surface of :class:`dataxid.ModelConfig` and the
helper dataclasses that live next to it (:class:`dataxid.Privacy`,
:class:`dataxid.Distribution`, :class:`dataxid.Bias`,
:class:`dataxid.Synthetic`): defaults, field assignment, wire-format
serialization, frozen-immutability and constructor-time validation.

A small slice covers ``_resolve_config`` — the internal normalizer that
turns ``None`` / ``dict`` / ``ModelConfig`` inputs into a single
``ModelConfig`` instance — because it is the one place where the public
ergonomics (accepting plain dicts) meet the typed contract.
"""

from dataclasses import FrozenInstanceError, fields

import pytest

import dataxid
from dataxid import Bias, Distribution, ModelConfig, Privacy, Synthetic
from dataxid.exceptions import InvalidRequestError
from dataxid.training._config import _resolve_config


class TestModelConfigDefaults:
    """ModelConfig() produces the documented default training profile."""

    @pytest.mark.parametrize(
        "field_name,expected",
        [
            ("embedding_dim", 64),
            ("model_size", "medium"),
            ("batch_size", 256),
            ("max_epochs", 100),
            ("early_stop_patience", 4),
            ("val_split", 0.1),
            ("encoding_types", None),
            ("accumulation_steps", 1),
            ("learning_rate", None),
            ("label_smoothing", 0.0),
            ("embedding_dropout", 0.5),
            ("time_limit_seconds", 0.0),
            ("seed", None),
            ("timeout", 14400.0),
        ],
    )
    def test_default(self, field_name: str, expected: object) -> None:
        assert getattr(ModelConfig(), field_name) == expected

    def test_privacy_default_is_privacy_instance(self) -> None:
        assert isinstance(ModelConfig().privacy, Privacy)


class TestPrivacyDefaults:
    """Privacy() defaults match the documented privacy contract."""

    @pytest.mark.parametrize(
        "field_name,expected",
        [
            ("protect_rare", True),
            ("rare_strategy", "mask"),
            ("enabled", False),
            ("noise", 0.1),
        ],
    )
    def test_default(self, field_name: str, expected: object) -> None:
        assert getattr(Privacy(), field_name) == expected

    def test_each_modelconfig_gets_fresh_privacy(self) -> None:
        """``default_factory`` prevents shared mutable state across instances."""
        cfg_a = ModelConfig()
        cfg_b = ModelConfig()
        assert cfg_a.privacy is not cfg_b.privacy


class TestPrivacyValidation:
    """Privacy rejects malformed inputs at construction time."""

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            (
                {"rare_strategy": "invalid"},
                "rare_strategy must be 'mask' or 'sample'",
            ),
            (
                {"rare_strategy": None},
                "rare_strategy must be 'mask' or 'sample'",
            ),
            ({"noise": -0.1}, "noise must be a non-negative finite"),
            ({"noise": float("nan")}, "noise must be a non-negative finite"),
            ({"noise": float("inf")}, "noise must be a non-negative finite"),
            ({"noise": "loud"}, "noise must be a real number"),
        ],
    )
    def test_invalid_input_rejected(
        self, kwargs: dict, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            Privacy(**kwargs)


class TestModelConfigFields:
    """Field assignment is honored verbatim by the dataclass constructor."""

    @pytest.mark.parametrize(
        "kwargs,field_name,expected",
        [
            ({"embedding_dim": 128}, "embedding_dim", 128),
            ({"model_size": "small"}, "model_size", "small"),
            ({"model_size": "large"}, "model_size", "large"),
            ({"encoding_types": {"age": "TABULAR_NUMERIC_BINNED"}},
             "encoding_types", {"age": "TABULAR_NUMERIC_BINNED"}),
            ({"seed": 42}, "seed", 42),
            ({"learning_rate": 0.001}, "learning_rate", 0.001),
        ],
    )
    def test_assignment(
        self, kwargs: dict, field_name: str, expected: object
    ) -> None:
        assert getattr(ModelConfig(**kwargs), field_name) == expected

    def test_privacy_block_replaces_default(self) -> None:
        privacy = Privacy(enabled=True, noise=0.5)
        cfg = ModelConfig(privacy=privacy)
        assert cfg.privacy.enabled is True
        assert cfg.privacy.noise == 0.5

    def test_privacy_protect_rare_false(self) -> None:
        cfg = ModelConfig(privacy=Privacy(protect_rare=False))
        assert cfg.privacy.protect_rare is False

    def test_privacy_rare_strategy_sample(self) -> None:
        cfg = ModelConfig(privacy=Privacy(rare_strategy="sample"))
        assert cfg.privacy.rare_strategy == "sample"


class TestModelConfigToDict:
    """to_dict() is the wire format: flat dict with privacy nested as dict."""

    def test_scalar_fields_round_trip(self) -> None:
        d = ModelConfig(embedding_dim=32, model_size="small").to_dict()
        assert d["embedding_dim"] == 32
        assert d["model_size"] == "small"

    def test_privacy_default_serialized_as_nested_dict(self) -> None:
        d = ModelConfig().to_dict()
        assert d["privacy"] == {
            "protect_rare": True,
            "rare_strategy": "mask",
            "enabled": False,
            "noise": 0.1,
        }

    def test_privacy_custom_values_serialized(self) -> None:
        cfg = ModelConfig(
            privacy=Privacy(
                protect_rare=False,
                rare_strategy="sample",
                enabled=True,
                noise=0.25,
            )
        )
        assert cfg.to_dict()["privacy"] == {
            "protect_rare": False,
            "rare_strategy": "sample",
            "enabled": True,
            "noise": 0.25,
        }

    def test_none_fields_included(self) -> None:
        d = ModelConfig().to_dict()
        assert d["learning_rate"] is None
        assert d["seed"] is None
        assert d["encoding_types"] is None

    def test_dict_keys_match_dataclass_fields(self) -> None:
        """to_dict() exposes every dataclass field — no silent omission."""
        expected = {f.name for f in fields(ModelConfig)}
        assert set(ModelConfig().to_dict().keys()) == expected


class TestModelConfigValidation:
    """ModelConfig rejects out-of-range numeric inputs at construction time.

    Server-side schemas (``ModelConfigSchema`` in the API) enforce the same
    invariants; the client-side checks turn obviously wrong values into
    immediate, local failures instead of network round-trips.
    """

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            ({"embedding_dim": 0}, "embedding_dim must be a positive integer"),
            ({"embedding_dim": -1}, "embedding_dim must be a positive integer"),
            ({"embedding_dim": 1.5}, "embedding_dim must be a positive integer"),
            ({"batch_size": 0}, "batch_size must be a positive integer"),
            ({"max_epochs": 0}, "max_epochs must be a positive integer"),
            ({"accumulation_steps": 0}, "accumulation_steps must be a positive integer"),
            (
                {"early_stop_patience": -1},
                "early_stop_patience must be a non-negative integer",
            ),
            ({"val_split": 0.0}, r"val_split must be in \(0, 1\)"),
            ({"val_split": 1.0}, r"val_split must be in \(0, 1\)"),
            ({"val_split": 1.5}, r"val_split must be in \(0, 1\)"),
            ({"label_smoothing": -0.1}, r"label_smoothing must be in \[0, 1\)"),
            ({"label_smoothing": 1.0}, r"label_smoothing must be in \[0, 1\)"),
            ({"embedding_dropout": 1.0}, r"embedding_dropout must be in \[0, 1\)"),
            (
                {"time_limit_seconds": -1.0},
                "time_limit_seconds must be a non-negative finite",
            ),
            (
                {"time_limit_seconds": float("inf")},
                "time_limit_seconds must be a non-negative finite",
            ),
            ({"timeout": 0.0}, "timeout must be a positive finite"),
            ({"timeout": -1.0}, "timeout must be a positive finite"),
            ({"learning_rate": 0.0}, "learning_rate must be a positive finite"),
            ({"learning_rate": -0.001}, "learning_rate must be a positive finite"),
            (
                {"learning_rate": float("nan")},
                "learning_rate must be a positive finite",
            ),
            ({"model_size": "xl"}, "model_size must be one of"),
            ({"model_size": "MEDIUM"}, "model_size must be one of"),
            ({"model_size": ""}, "model_size must be one of"),
            ({"model_size": None}, "model_size must be one of"),
        ],
    )
    def test_invalid_input_rejected(
        self, kwargs: dict, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            ModelConfig(**kwargs)

    def test_learning_rate_none_accepted(self) -> None:
        """``None`` is the documented sentinel for 'auto-pick'; never rejected."""
        assert ModelConfig(learning_rate=None).learning_rate is None

    @pytest.mark.parametrize("size", ["small", "medium", "large"])
    def test_model_size_valid_values_accepted(self, size: str) -> None:
        """Every documented model_size literal is accepted."""
        assert ModelConfig(model_size=size).model_size == size

    def test_invalid_model_size_param_attribute(self) -> None:
        """The error names the offending field for programmatic handling."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ModelConfig(model_size="xl")
        assert exc_info.value.param == "model_size"


class TestEncodingTypesValidation:
    """ModelConfig.encoding_types is a typed mapping; the contract is enforced
    at construction time so that the ``EncodingType(...)`` lookup deep inside
    ``analyze()`` never sees an unknown value.
    """

    def test_none_is_accepted(self) -> None:
        assert ModelConfig(encoding_types=None).encoding_types is None

    def test_empty_dict_is_accepted(self) -> None:
        assert ModelConfig(encoding_types={}).encoding_types == {}

    @pytest.mark.parametrize("value", ["AUTO", "TABULAR_CATEGORICAL", "TABULAR_LAT_LONG"])
    def test_valid_string_value_accepted(self, value: str) -> None:
        cfg = ModelConfig(encoding_types={"col": value})
        assert cfg.encoding_types == {"col": value}

    def test_valid_enum_member_accepted(self) -> None:
        from dataxid.encoder._ports import EncodingType
        cfg = ModelConfig(encoding_types={"col": EncodingType.auto})
        assert cfg.encoding_types == {"col": EncodingType.auto}

    @pytest.mark.parametrize(
        "value,error_match",
        [
            (["col"], "encoding_types must be a dict or None"),
            ("AUTO", "encoding_types must be a dict or None"),
            (42, "encoding_types must be a dict or None"),
        ],
    )
    def test_non_dict_rejected(self, value: object, error_match: str) -> None:
        with pytest.raises(InvalidRequestError, match=error_match):
            ModelConfig(encoding_types=value)  # type: ignore[arg-type]

    def test_non_string_key_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="keys must be strings"):
            ModelConfig(encoding_types={1: "AUTO"})  # type: ignore[dict-item]

    @pytest.mark.parametrize(
        "value",
        ["FANCY_TYPE", "categorical", "tabular_categorical", "auto", ""],
    )
    def test_unknown_string_value_rejected(self, value: str) -> None:
        with pytest.raises(InvalidRequestError, match="must be one of"):
            ModelConfig(encoding_types={"col": value})

    def test_non_string_non_enum_value_rejected(self) -> None:
        with pytest.raises(InvalidRequestError, match="must be one of"):
            ModelConfig(encoding_types={"col": 42})  # type: ignore[dict-item]

    def test_param_attribute_is_encoding_types(self) -> None:
        with pytest.raises(InvalidRequestError) as exc_info:
            ModelConfig(encoding_types={"col": "FANCY_TYPE"})
        assert exc_info.value.param == "encoding_types"


class TestResolveConfig:
    """_resolve_config normalizes None/dict/instance into a ModelConfig."""

    def test_none_returns_defaults(self) -> None:
        cfg = _resolve_config(None)
        assert isinstance(cfg, ModelConfig)
        assert cfg == ModelConfig()

    def test_empty_dict_returns_defaults(self) -> None:
        assert _resolve_config({}) == ModelConfig()

    def test_dict_partial_fills_defaults(self) -> None:
        cfg = _resolve_config({"embedding_dim": 128})
        assert cfg.embedding_dim == 128
        assert cfg.model_size == "medium"

    def test_dict_privacy_nested_dict_promoted_to_instance(self) -> None:
        cfg = _resolve_config({"privacy": {"enabled": True, "noise": 0.7}})
        assert isinstance(cfg.privacy, Privacy)
        assert cfg.privacy.enabled is True
        assert cfg.privacy.noise == 0.7
        assert cfg.privacy.protect_rare is True

    def test_dict_privacy_instance_preserved(self) -> None:
        privacy = Privacy(rare_strategy="sample")
        cfg = _resolve_config({"privacy": privacy})
        assert cfg.privacy is privacy

    def test_dict_privacy_none_filled_with_defaults(self) -> None:
        """Explicit ``None`` for privacy is replaced with a default Privacy."""
        cfg = _resolve_config({"privacy": None})
        assert cfg.privacy == Privacy()

    @pytest.mark.parametrize("bad_privacy", ["a string", 42, 3.14, []])
    def test_dict_privacy_wrong_type_raises(self, bad_privacy: object) -> None:
        """Anything that is not Privacy/dict/None is rejected up-front."""
        with pytest.raises(InvalidRequestError, match="config\\['privacy'\\]") as exc_info:
            _resolve_config({"privacy": bad_privacy})
        assert exc_info.value.param == "config"

    def test_model_config_returned_as_is(self) -> None:
        cfg_in = ModelConfig(embedding_dim=128, model_size="large")
        assert _resolve_config(cfg_in) is cfg_in

    @pytest.mark.parametrize("bad_input", ["embedding_dim=128", 42, 3.14, []])
    def test_wrong_type_raises(self, bad_input: object) -> None:
        with pytest.raises(InvalidRequestError, match="ModelConfig") as exc_info:
            _resolve_config(bad_input)  # type: ignore[arg-type]
        assert exc_info.value.param == "config"

    def test_unknown_dict_key_raises(self) -> None:
        """Unknown keys in a dict-config surface as InvalidRequestError so
        callers see a consistent SDK error type instead of a raw ``TypeError``
        leaking out of ``ModelConfig.__init__``."""
        with pytest.raises(InvalidRequestError, match="unknown field") as exc_info:
            _resolve_config({"unknown_key": 99})
        assert exc_info.value.param == "config"

    def test_unknown_privacy_dict_key_raises(self) -> None:
        with pytest.raises(InvalidRequestError, match="config\\['privacy'\\]") as exc_info:
            _resolve_config({"privacy": {"unknown_privacy_key": True}})
        assert exc_info.value.param == "config"


class TestTopLevelImport:
    """Public dataclasses are re-exported at the package root."""

    @pytest.mark.parametrize(
        "name,obj",
        [
            ("ModelConfig", ModelConfig),
            ("Privacy", Privacy),
            ("Distribution", Distribution),
            ("Bias", Bias),
            ("Synthetic", Synthetic),
        ],
    )
    def test_importable(self, name: str, obj: type) -> None:
        assert getattr(dataxid, name) is obj

    @pytest.mark.parametrize(
        "name", ["ModelConfig", "Privacy", "Distribution", "Bias", "Synthetic"]
    )
    def test_in_all(self, name: str) -> None:
        assert name in dataxid.__all__


class TestDistributionDataclass:
    """Distribution: frozen value object describing a categorical override."""

    def test_basic_construction(self) -> None:
        d = Distribution(column="gender", probabilities={"M": 0.7, "F": 0.3})
        assert d.column == "gender"
        assert d.probabilities == {"M": 0.7, "F": 0.3}

    def test_frozen_cannot_reassign(self) -> None:
        d = Distribution(column="gender", probabilities={"M": 0.5, "F": 0.5})
        with pytest.raises(FrozenInstanceError):
            d.column = "age"  # type: ignore[misc]

    def test_unnormalized_probabilities_accepted(self) -> None:
        """Probabilities are normalized downstream; raw values pass through."""
        d = Distribution(column="gender", probabilities={"M": 2.0, "F": 3.0})
        assert d.probabilities == {"M": 2.0, "F": 3.0}

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            (
                {"column": "", "probabilities": {"A": 1.0}},
                "column must be a non-empty string",
            ),
            (
                {"column": "   ", "probabilities": {"A": 1.0}},
                "column must be a non-empty string",
            ),
            (
                {"column": 42, "probabilities": {"A": 1.0}},
                "column must be a non-empty string",
            ),
            (
                {"column": "gender", "probabilities": {}},
                "probabilities must be non-empty",
            ),
            (
                {"column": "gender", "probabilities": {"M": -0.1, "F": 0.9}},
                r"probabilities\['M'\] must be non-negative",
            ),
            (
                {"column": "gender", "probabilities": {"M": float("nan"), "F": 0.9}},
                r"probabilities\['M'\] must be finite",
            ),
            (
                {"column": "gender", "probabilities": {"M": float("inf"), "F": 0.9}},
                r"probabilities\['M'\] must be finite",
            ),
            (
                {"column": "gender", "probabilities": {"M": "not-a-number", "F": 0.9}},
                r"probabilities\['M'\] must be a real number",
            ),
        ],
    )
    def test_invalid_input_rejected(
        self, kwargs: dict, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            Distribution(**kwargs)

    @pytest.mark.parametrize(
        "bad_key,error_match",
        [
            (25, "must be strings, got int"),
            (None, "must be strings, got NoneType"),
            (("a",), "must be strings, got tuple"),
            ("", "non-empty / non-whitespace"),
            ("   ", "non-empty / non-whitespace"),
        ],
    )
    def test_non_string_or_blank_keys_rejected(
        self,
        bad_key: object,
        error_match: str,
    ) -> None:
        """Numeric or blank ``probabilities`` keys are rejected explicitly so
        the wire format stays consistent — server-side codes lookup uses
        ``isinstance(key, str)``, and a numeric key would silently no-op."""
        with pytest.raises(InvalidRequestError, match=error_match) as exc_info:
            Distribution(column="age", probabilities={bad_key: 1.0})  # type: ignore[dict-item]
        assert exc_info.value.param == "probabilities"


class TestBiasDataclass:
    """Bias: frozen value object describing statistical-parity correction."""

    def test_basic_construction(self) -> None:
        b = Bias(target="income", sensitive=["gender"])
        assert b.target == "income"
        assert b.sensitive == ["gender"]

    def test_multi_sensitive(self) -> None:
        b = Bias(target="income", sensitive=["gender", "race"])
        assert b.sensitive == ["gender", "race"]

    def test_frozen_cannot_reassign(self) -> None:
        b = Bias(target="income", sensitive=["gender"])
        with pytest.raises(FrozenInstanceError):
            b.target = "age"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            (
                {"target": "", "sensitive": ["gender"]},
                "target must be a non-empty string",
            ),
            (
                {"target": "   ", "sensitive": ["gender"]},
                "target must be a non-empty string",
            ),
            (
                {"target": 42, "sensitive": ["gender"]},
                "target must be a non-empty string",
            ),
            (
                {"target": "income", "sensitive": []},
                "sensitive must be a non-empty list",
            ),
            (
                {"target": "income", "sensitive": ["gender", ""]},
                "sensitive entries must be non-empty",
            ),
            (
                {"target": "income", "sensitive": ["gender", "   "]},
                "sensitive entries must be non-empty",
            ),
            (
                {"target": "income", "sensitive": ["gender", "income"]},
                "cannot appear in sensitive",
            ),
        ],
    )
    def test_invalid_input_rejected(
        self, kwargs: dict, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            Bias(**kwargs)


class TestSyntheticDataclass:
    """Synthetic: frozen generation-time tuning preset."""

    def test_defaults(self) -> None:
        s = Synthetic()
        assert s.n is None
        assert s.seed is None
        assert s.diversity == 1.0
        assert s.rare_cutoff == 1.0
        assert s.rare_strategy is None

    def test_basic_construction(self) -> None:
        s = Synthetic(
            n=100, seed=42, diversity=0.8, rare_cutoff=0.9, rare_strategy="mask"
        )
        assert s.n == 100
        assert s.seed == 42
        assert s.diversity == 0.8
        assert s.rare_cutoff == 0.9
        assert s.rare_strategy == "mask"

    def test_frozen_cannot_reassign(self) -> None:
        s = Synthetic(diversity=0.5)
        with pytest.raises(FrozenInstanceError):
            s.diversity = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs,error_match",
        [
            ({"n": 0}, "n must be a positive integer"),
            ({"n": -5}, "n must be a positive integer"),
            ({"n": 1.5}, "n must be a positive integer"),
            ({"diversity": 0.0}, "diversity must be > 0"),
            ({"diversity": -0.1}, "diversity must be > 0"),
            ({"rare_cutoff": 0.0}, r"rare_cutoff must be in \(0, 1\]"),
            ({"rare_cutoff": 1.5}, r"rare_cutoff must be in \(0, 1\]"),
            (
                {"rare_strategy": "invalid"},
                "rare_strategy must be 'mask', 'sample', or None",
            ),
        ],
    )
    def test_invalid_input_rejected(
        self, kwargs: dict, error_match: str
    ) -> None:
        with pytest.raises(ValueError, match=error_match):
            Synthetic(**kwargs)
