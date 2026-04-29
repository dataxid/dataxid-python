# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :class:`dataxid.exceptions.QuotaExceededError` and its HTTP wire mapping.

Quota errors are the only case where the SDK surfaces structured billing
context (``usage``, ``upgrade_url``) on top of the standard HTTP failure,
so this module locks down both the exception class itself and the
``_parse_error`` routing path that builds it.

Coverage:

* ``QuotaExceededError`` class contract (hierarchy, attributes, defaults)
* ``_ERROR_MAP`` directly: ``quota_exceeded`` type routes to the right class
* ``_parse_error`` end-to-end: 402 responses with and without the ``type``
  discriminator, with and without billing context fields
* Pickle round-trip: custom attributes survive cross-process serialization
"""

from __future__ import annotations

import pickle

import httpx
import pytest

from dataxid.client._http import _ERROR_MAP, DataxidClient
from dataxid.exceptions import APIError, DataxidError, QuotaExceededError


def _make_response(
    status_code: int,
    body: dict,
    headers: dict | None = None,
) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=body, headers=headers or {})


class TestQuotaExceededError:
    """Exception class: hierarchy, custom attributes, and inherited metadata."""

    def test_subclass_of_dataxid_error(self) -> None:
        assert issubclass(QuotaExceededError, DataxidError)

    def test_catchable_as_dataxid_error(self) -> None:
        with pytest.raises(DataxidError):
            raise QuotaExceededError("quota exceeded")

    def test_usage_attribute_stored_verbatim(self) -> None:
        """The SDK does not validate ``usage`` — it is an opaque server payload
        passed through to the caller for display / billing logic."""
        usage = {"used": 98_500, "limit": 100_000, "requested": 5000, "period": "2026-03"}
        exc = QuotaExceededError("quota exceeded", usage=usage)
        assert exc.usage == usage

    def test_upgrade_url_attribute(self) -> None:
        url = "https://example.test/checkout/pro"
        exc = QuotaExceededError("quota exceeded", upgrade_url=url)
        assert exc.upgrade_url == url

    def test_defaults_to_none(self) -> None:
        exc = QuotaExceededError("quota exceeded")
        assert exc.usage is None
        assert exc.upgrade_url is None

    def test_carries_request_metadata(self) -> None:
        """``status_code`` and ``request_id`` come from the base class — quota
        errors must not shadow them with their custom attributes."""
        exc = QuotaExceededError(
            "limit reached",
            usage={"used": 100_000},
            upgrade_url="https://example.test/upgrade",
            status_code=402,
            request_id="req_abc",
        )
        assert exc.status_code == 402
        assert exc.request_id == "req_abc"
        assert str(exc) == "limit reached"


class TestErrorMapRouting:
    """``_ERROR_MAP`` is the wire-format ↔ exception-class lookup table."""

    def test_quota_exceeded_type_routes_to_quota_exceeded_error(self) -> None:
        assert _ERROR_MAP["quota_exceeded"] is QuotaExceededError

    def test_map_contains_all_expected_types(self) -> None:
        """Pin the full set of mapped error types so adding a new HTTP error
        flow without touching the map is caught at review time."""
        assert set(_ERROR_MAP) == {
            "authentication_error",
            "invalid_request_error",
            "not_found_error",
            "quota_exceeded",
            "rate_limit_error",
        }


class TestParseErrorQuota:
    """``DataxidClient._parse_error`` — 402 responses → ``QuotaExceededError``."""

    def test_402_with_full_billing_context(self) -> None:
        body = {
            "error": {
                "type": "quota_exceeded",
                "code": "free_tier_limit_reached",
                "message": "Monthly free tier limit exceeded.",
                "usage": {
                    "used": 100_000, "limit": 100_000,
                    "requested": 5000, "period": "2026-03",
                },
                "upgrade_url": "https://example.test/checkout/pro",
            }
        }
        exc = DataxidClient._parse_error(_make_response(402, body))

        assert isinstance(exc, QuotaExceededError)
        assert exc.status_code == 402
        assert exc.usage == body["error"]["usage"]
        assert exc.upgrade_url == body["error"]["upgrade_url"]
        assert str(exc) == "Monthly free tier limit exceeded."

    def test_402_without_billing_context_still_routes(self) -> None:
        """Servers that signal ``type=quota_exceeded`` but omit ``usage`` /
        ``upgrade_url`` still produce a ``QuotaExceededError`` — the optional
        fields default to ``None`` rather than raising on the parse path."""
        body = {"error": {"type": "quota_exceeded", "message": "Upgrade to continue."}}
        exc = DataxidClient._parse_error(_make_response(402, body))

        assert isinstance(exc, QuotaExceededError)
        assert exc.usage is None
        assert exc.upgrade_url is None
        assert str(exc) == "Upgrade to continue."

    def test_402_without_type_falls_back_to_api_error(self) -> None:
        """The discriminator is the ``type`` field, not the status code:
        a 402 response missing ``type`` must surface as the generic
        ``APIError`` so the caller does not assume billing context exists."""
        body = {"error": {"message": "Payment Required"}}
        exc = DataxidClient._parse_error(_make_response(402, body))

        assert isinstance(exc, APIError)
        assert not isinstance(exc, QuotaExceededError)


class TestPickleRoundTrip:
    """Cross-process safety: custom attributes survive ``pickle.dumps`` / ``loads``.

    Users of multi-process executors (Dask, Ray, joblib) regularly pickle
    raised exceptions; losing ``usage`` / ``upgrade_url`` would break their
    error-handling code without an obvious failure mode.
    """

    def test_attributes_round_trip(self) -> None:
        original = QuotaExceededError(
            "quota exceeded",
            usage={"used": 100_000, "limit": 100_000},
            upgrade_url="https://example.test/upgrade",
            status_code=402,
            request_id="req_pickled",
        )
        restored = pickle.loads(pickle.dumps(original))

        assert isinstance(restored, QuotaExceededError)
        assert restored.usage == original.usage
        assert restored.upgrade_url == original.upgrade_url
        assert restored.status_code == 402
        assert restored.request_id == "req_pickled"
        assert str(restored) == str(original)
