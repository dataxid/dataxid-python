# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the SDK exception hierarchy.

The hierarchy is part of the SDK's public contract: callers rely on
``except DataxidError`` to catch every error the SDK raises, and on the
typed ``status_code`` / ``request_id`` / ``param`` / ``retry_after`` /
``usage`` / ``upgrade_url`` attributes to make programmatic decisions.

Tests are organized by responsibility: hierarchy invariants, base-class
attributes, per-subclass extensions, top-level package re-exports, and
pickle round-trip safety (so exceptions survive async/multiprocess
boundaries without losing their typed attributes).
"""

import pickle

import pytest

import dataxid
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

_ALL_SUBCLASSES = [
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    QuotaExceededError,
    ConflictError,
    ModelNotReadyError,
    TrainingError,
    TrainingTimeoutError,
    APIError,
]


class TestHierarchy:
    """Every SDK exception is a ``DataxidError`` so callers can catch one."""

    @pytest.mark.parametrize("exc_class", _ALL_SUBCLASSES)
    def test_subclass_can_be_caught_as_dataxid_error(
        self, exc_class: type[DataxidError]
    ) -> None:
        with pytest.raises(DataxidError):
            raise exc_class("test")

    def test_dataxid_error_is_a_python_exception(self) -> None:
        assert issubclass(DataxidError, Exception)


class TestBaseAttributes:
    """``DataxidError`` carries ``message``, ``status_code``, ``request_id``."""

    def test_str_returns_constructor_message(self) -> None:
        assert str(DataxidError("something went wrong")) == "something went wrong"

    def test_status_code_is_stored_when_provided(self) -> None:
        exc = DataxidError("fail", status_code=500)
        assert exc.status_code == 500

    def test_request_id_is_stored_when_provided(self) -> None:
        exc = DataxidError("fail", request_id="req_abc123")
        assert exc.request_id == "req_abc123"

    def test_optional_attributes_default_to_none(self) -> None:
        exc = DataxidError("fail")
        assert exc.status_code is None
        assert exc.request_id is None

    def test_all_attributes_can_be_set_together(self) -> None:
        exc = DataxidError("fail", status_code=400, request_id="req_xyz")
        assert str(exc) == "fail"
        assert exc.status_code == 400
        assert exc.request_id == "req_xyz"


class TestInvalidRequestError:
    """``InvalidRequestError.param`` points at the offending field."""

    def test_param_is_stored_when_provided(self) -> None:
        exc = InvalidRequestError("bad field", param="metadata.features")
        assert exc.param == "metadata.features"

    def test_param_defaults_to_none(self) -> None:
        assert InvalidRequestError("bad request").param is None

    def test_inherits_base_attributes(self) -> None:
        exc = InvalidRequestError(
            "bad", param="x", status_code=400, request_id="req_1"
        )
        assert exc.status_code == 400
        assert exc.request_id == "req_1"
        assert exc.param == "x"
        assert str(exc) == "bad"


class TestInvalidRequestErrorContract:
    """``InvalidRequestError`` multi-inherits from ``DataxidError`` and
    ``ValueError`` so callers can use either ``except DataxidError`` for
    SDK-wide handling or the Python-idiomatic ``except ValueError`` for
    validation failures.
    """

    def test_is_a_dataxid_error(self) -> None:
        assert issubclass(InvalidRequestError, DataxidError)

    def test_is_a_value_error(self) -> None:
        """Multi-inheritance from ``ValueError`` keeps ``except ValueError``
        working for callers writing Python-idiomatic validation handlers."""
        assert issubclass(InvalidRequestError, ValueError)

    def test_can_be_caught_as_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise InvalidRequestError("bad", param="x")

    def test_can_be_caught_as_dataxid_error(self) -> None:
        with pytest.raises(DataxidError):
            raise InvalidRequestError("bad", param="x")

    def test_domain_object_rejection_is_an_invalid_request_error(self) -> None:
        """Domain object constructors (Bias, Synthetic, Distribution,
        Privacy) raise ``InvalidRequestError`` with a populated ``param``
        — the SDK's single contract for "bad caller input"."""
        from dataxid import Bias

        with pytest.raises(InvalidRequestError) as exc_info:
            Bias(target="", sensitive=["g"])
        assert exc_info.value.param == "target"

    def test_domain_object_rejection_is_still_a_value_error(self) -> None:
        """Backwards-compatibility: callers using ``except ValueError``
        continue to catch validation failures from domain constructors."""
        from dataxid import Bias

        with pytest.raises(ValueError):
            Bias(target="", sensitive=["g"])


class TestRateLimitError:
    """``RateLimitError.retry_after`` carries the server's back-off hint."""

    def test_retry_after_is_stored_when_provided(self) -> None:
        assert RateLimitError("slow down", retry_after=30.0).retry_after == 30.0

    def test_retry_after_defaults_to_none(self) -> None:
        assert RateLimitError("slow down").retry_after is None

    def test_inherits_base_attributes(self) -> None:
        exc = RateLimitError(
            "limit", retry_after=60.0, status_code=429, request_id="req_2"
        )
        assert exc.status_code == 429
        assert exc.request_id == "req_2"
        assert exc.retry_after == 60.0
        assert str(exc) == "limit"


class TestQuotaExceededError:
    """``QuotaExceededError`` exposes ``usage`` and ``upgrade_url`` for UIs."""

    def test_usage_is_stored_when_provided(self) -> None:
        usage = {"models_trained": 10, "limit": 5}
        assert QuotaExceededError("over quota", usage=usage).usage == usage

    def test_upgrade_url_is_stored_when_provided(self) -> None:
        url = "https://example.test/upgrade"
        assert QuotaExceededError("over quota", upgrade_url=url).upgrade_url == url

    def test_optional_attributes_default_to_none(self) -> None:
        exc = QuotaExceededError("over quota")
        assert exc.usage is None
        assert exc.upgrade_url is None

    def test_inherits_base_attributes(self) -> None:
        exc = QuotaExceededError(
            "over quota",
            usage={"x": 1},
            upgrade_url="https://u",
            status_code=402,
            request_id="req_q",
        )
        assert exc.status_code == 402
        assert exc.request_id == "req_q"
        assert exc.usage == {"x": 1}
        assert exc.upgrade_url == "https://u"
        assert str(exc) == "over quota"


class TestTopLevelImport:
    """Every exception is re-exported from the ``dataxid`` package root."""

    @pytest.mark.parametrize(
        "exc_class", [DataxidError, *_ALL_SUBCLASSES]
    )
    def test_reexported_at_package_root(
        self, exc_class: type[DataxidError]
    ) -> None:
        assert getattr(dataxid, exc_class.__name__) is exc_class

    def test_no_unexposed_exception_subclasses(self) -> None:
        """Catch the case where a new ``*Error`` is added to ``exceptions``
        but forgotten in ``dataxid/__init__.py`` re-exports."""
        from dataxid import exceptions as exc_mod
        defined_in_module = {
            name for name, obj in vars(exc_mod).items()
            if isinstance(obj, type)
            and issubclass(obj, DataxidError)
            and obj.__module__ == exc_mod.__name__
        }
        missing = {n for n in defined_in_module if not hasattr(dataxid, n)}
        assert missing == set(), f"Missing top-level re-exports: {missing}"


class TestPickleRoundTrip:
    """Exceptions must survive pickle so they cross process boundaries.

    SDK exceptions travel through ``concurrent.futures``, ``multiprocessing``,
    Celery workers, and similar transports that pickle their arguments.
    Losing a typed attribute en route would silently downgrade a typed error
    into a generic one at the call site, so every public exception is
    exercised round-trip with a fully-populated payload.
    """

    @pytest.mark.parametrize("exc_class", _ALL_SUBCLASSES)
    def test_base_attributes_survive_pickle(
        self, exc_class: type[DataxidError]
    ) -> None:
        original = exc_class("boom", status_code=500, request_id="req_p")
        restored = pickle.loads(pickle.dumps(original))
        assert type(restored) is exc_class
        assert str(restored) == "boom"
        assert restored.status_code == 500
        assert restored.request_id == "req_p"

    def test_invalid_request_param_survives_pickle(self) -> None:
        original = InvalidRequestError(
            "bad", param="features", status_code=400, request_id="req_p"
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.param == "features"
        assert restored.status_code == 400

    def test_rate_limit_retry_after_survives_pickle(self) -> None:
        original = RateLimitError(
            "slow", retry_after=12.5, status_code=429, request_id="req_p"
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.retry_after == 12.5
        assert restored.status_code == 429

    def test_quota_exceeded_payload_survives_pickle(self) -> None:
        original = QuotaExceededError(
            "over",
            usage={"models": 10},
            upgrade_url="https://u",
            status_code=402,
            request_id="req_p",
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.usage == {"models": 10}
        assert restored.upgrade_url == "https://u"
        assert restored.status_code == 402
