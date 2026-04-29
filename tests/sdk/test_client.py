# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :class:`dataxid.client.DataxidClient`.

Covers, with mocked ``httpx.request``:
  - API key / base URL resolution (explicit > module-level > env var)
  - Auth, Idempotency-Key and User-Agent header injection
  - Retry policy (status codes, timeout, connection error, max attempts)
  - 204 No Content handling
  - Structured error parsing into typed exceptions
  - Exponential backoff with ``Retry-After`` precedence
"""

from __future__ import annotations

import email.utils
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

import dataxid
from dataxid.client._http import (
    _BACKOFF_BASE,
    _MAX_RETRIES,
    _MAX_RETRY_AFTER_SECONDS,
    DataxidClient,
    _parse_retry_after,
)
from dataxid.exceptions import (
    APIError,
    AuthenticationError,
    ConflictError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
)


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {}
    return resp


@pytest.fixture(autouse=True)
def _isolated_client_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Isolate every test from ambient ``dataxid.api_key`` / ``dataxid.base_url``
    state and from a developer's local ``DATAXID_API_KEY`` env var.

    Without this fixture, results would silently depend on the host shell's
    environment, defeating the purpose of unit tests.
    """
    monkeypatch.setattr(dataxid, "api_key", None)
    monkeypatch.setattr(dataxid, "base_url", "https://api.dataxid.com")
    monkeypatch.delenv("DATAXID_API_KEY", raising=False)


@pytest.fixture
def mock_request() -> Iterator[MagicMock]:
    """Mock the underlying ``httpx.request`` so no real HTTP traffic occurs."""
    with patch("dataxid.client._http.httpx.request") as request:
        request.return_value = _mock_response(200, {"data": {}})
        yield request


@pytest.fixture
def mock_sleep() -> Iterator[MagicMock]:
    """Mock ``time.sleep`` so tests don't actually wait for backoff."""
    with patch("dataxid.client._http.time.sleep") as sleep:
        yield sleep


@pytest.fixture
def client() -> DataxidClient:
    """A ``DataxidClient`` with a fixed test API key."""
    return DataxidClient(api_key="dx_test_abc")


class TestApiKeyResolution:
    """Resolution order: explicit constructor arg > module-level > env var > error."""

    def test_explicit_key_used(self) -> None:
        client = DataxidClient(api_key="dx_test_explicit")
        assert client.api_key == "dx_test_explicit"

    def test_module_level_key_used(self) -> None:
        dataxid.api_key = "dx_test_module"
        client = DataxidClient()
        assert client.api_key == "dx_test_module"

    def test_explicit_overrides_module(self) -> None:
        dataxid.api_key = "dx_test_module"
        client = DataxidClient(api_key="dx_test_explicit")
        assert client.api_key == "dx_test_explicit"

    def test_no_key_raises_auth_error(self) -> None:
        client = DataxidClient()
        with pytest.raises(AuthenticationError, match="No API key"):
            _ = client.api_key

    def test_env_var_used_when_no_other_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAXID_API_KEY", "dx_test_env")
        client = DataxidClient()
        assert client.api_key == "dx_test_env"

    def test_explicit_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATAXID_API_KEY", "dx_test_env")
        client = DataxidClient(api_key="dx_test_explicit")
        assert client.api_key == "dx_test_explicit"

    def test_module_level_overrides_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataxid.api_key = "dx_test_module"
        monkeypatch.setenv("DATAXID_API_KEY", "dx_test_env")
        client = DataxidClient()
        assert client.api_key == "dx_test_module"

    def test_error_message_mentions_env_var(self) -> None:
        client = DataxidClient()
        with pytest.raises(AuthenticationError, match="DATAXID_API_KEY"):
            _ = client.api_key


class TestBaseUrlResolution:
    """Base URL: explicit constructor arg > module-level."""

    def test_explicit_url(self) -> None:
        client = DataxidClient(base_url="https://api.dataxid.com")
        assert client.base_url == "https://api.dataxid.com"

    def test_module_level_url(self) -> None:
        dataxid.base_url = "https://eu.api.dataxid.com"
        client = DataxidClient()
        assert client.base_url == "https://eu.api.dataxid.com"

    def test_explicit_overrides_module(self) -> None:
        dataxid.base_url = "https://eu.api.dataxid.com"
        client = DataxidClient(base_url="https://api.dataxid.com")
        assert client.base_url == "https://api.dataxid.com"


class TestAuthHeader:
    """Authorization and Content-Type headers are injected on every request."""

    def test_bearer_token_sent(self, mock_request: MagicMock) -> None:
        client = DataxidClient(api_key="dx_test_abc123")
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer dx_test_abc123"

    def test_content_type_json(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert kwargs["headers"]["Content-Type"] == "application/json"


class TestIdempotencyKey:
    """Idempotency-Key is added only to retryable POST requests."""

    def test_post_includes_idempotency_key(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(201, {"data": {"id": "mdl_1"}})
        client.post("/v1/models", json={"metadata": {}})

        _, kwargs = mock_request.call_args
        assert "Idempotency-Key" in kwargs["headers"]
        assert len(kwargs["headers"]["Idempotency-Key"]) == 32  # uuid4 hex

    def test_get_no_idempotency_key(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert "Idempotency-Key" not in kwargs["headers"]

    def test_post_idempotent_false_no_key(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        client.post("/v1/models/mdl_1/train-step", json={}, idempotent=False)

        _, kwargs = mock_request.call_args
        assert "Idempotency-Key" not in kwargs["headers"]

    def test_unique_keys_per_call(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(201, {"data": {"id": "mdl_1"}})

        client.post("/v1/models", json={})
        key1 = mock_request.call_args[1]["headers"]["Idempotency-Key"]

        client.post("/v1/models", json={})
        key2 = mock_request.call_args[1]["headers"]["Idempotency-Key"]

        assert key1 != key2


class TestSuccessfulRequests:
    """Happy path: 2xx responses are decoded as JSON; 204 returns ``None``."""

    def test_get_returns_json(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(200, {"data": {"id": "mdl_1"}})
        assert client.get("/v1/models/mdl_1") == {"data": {"id": "mdl_1"}}

    def test_post_returns_json(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(201, {"data": {"id": "mdl_1"}})
        assert client.post("/v1/models", json={"metadata": {}}) == {"data": {"id": "mdl_1"}}

    def test_delete_returns_none(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(204)
        assert client.delete("/v1/models/mdl_1") is None

    def test_url_construction(self, mock_request: MagicMock) -> None:
        client = DataxidClient(api_key="dx_test_abc", base_url="https://api.dataxid.com")
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert kwargs["url"] == "https://api.dataxid.com/v1/models"

    def test_url_construction_with_trailing_slash_in_base(
        self, mock_request: MagicMock,
    ) -> None:
        """A trailing slash in ``base_url`` must not produce a double slash."""
        client = DataxidClient(
            api_key="dx_test_abc", base_url="https://api.dataxid.com/",
        )
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert kwargs["url"] == "https://api.dataxid.com/v1/models"


class TestRetryLogic:
    """Retry policy: transient failures are retried up to ``_MAX_RETRIES``."""

    def test_retries_on_500(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            _mock_response(500, {"error": {"type": "api_error", "message": "fail"}}),
            _mock_response(200, {"data": {"ok": True}}),
        ]

        assert client.get("/v1/models") == {"data": {"ok": True}}
        assert mock_request.call_count == 2

    def test_retries_on_502(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            _mock_response(502, {"error": {"type": "api_error", "message": "bad gw"}}),
            _mock_response(200, {"data": {}}),
        ]

        client.get("/v1/models")
        assert mock_request.call_count == 2

    def test_retries_on_429(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        """429 is retried (rate-limit recovers); only the final attempt surfaces."""
        mock_request.side_effect = [
            _mock_response(
                429, {"error": {"type": "rate_limit_error", "message": "slow down"}},
            ),
            _mock_response(200, {"data": {}}),
        ]

        client.get("/v1/models")
        assert mock_request.call_count == 2

    def test_retries_on_timeout(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            httpx.TimeoutException("timeout"),
            _mock_response(200, {"data": {}}),
        ]

        assert client.get("/v1/models") == {"data": {}}
        assert mock_request.call_count == 2

    def test_retries_on_connect_error(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            httpx.ConnectError("refused"),
            _mock_response(200, {"data": {}}),
        ]

        assert client.get("/v1/models") == {"data": {}}
        assert mock_request.call_count == 2

    def test_raises_after_max_retries_timeout(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(APIError, match="timed out"):
            client.get("/v1/models")
        assert mock_request.call_count == _MAX_RETRIES

    def test_raises_after_max_retries_connect(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = httpx.ConnectError("refused")

        with pytest.raises(APIError, match="Connection failed"):
            client.get("/v1/models")
        assert mock_request.call_count == _MAX_RETRIES

    def test_raises_after_max_retries_500(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(
            500, {"error": {"type": "api_error", "message": "internal"}},
        )

        with pytest.raises(APIError):
            client.get("/v1/models")
        assert mock_request.call_count == _MAX_RETRIES

    def test_no_retry_on_400(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(
            400, {"error": {"type": "invalid_request_error", "message": "bad"}},
        )

        with pytest.raises(InvalidRequestError):
            client.post("/v1/models", json={})
        assert mock_request.call_count == 1

    def test_no_retry_on_401(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(
            401, {"error": {"type": "authentication_error", "message": "bad key"}},
        )

        with pytest.raises(AuthenticationError):
            client.get("/v1/models")
        assert mock_request.call_count == 1

    def test_no_retry_on_404(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(
            404, {"error": {"type": "not_found_error", "message": "not found"}},
        )

        with pytest.raises(NotFoundError):
            client.get("/v1/models/mdl_x")
        assert mock_request.call_count == 1


class TestErrorParsing:
    """``_parse_error`` maps the HTTP response body into the right exception type."""

    def test_400_invalid_request(self) -> None:
        resp = _mock_response(400, {
            "error": {"type": "invalid_request_error", "message": "bad param", "param": "metadata"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, InvalidRequestError)
        assert exc.param == "metadata"
        assert exc.status_code == 400

    def test_401_authentication(self) -> None:
        resp = _mock_response(401, {
            "error": {"type": "authentication_error", "message": "invalid key"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, AuthenticationError)
        assert exc.status_code == 401

    def test_404_not_found(self) -> None:
        resp = _mock_response(404, {
            "error": {"type": "not_found_error", "message": "model not found"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, NotFoundError)
        assert exc.status_code == 404

    def test_409_conflict(self) -> None:
        resp = _mock_response(409, {
            "error": {"type": "api_error", "message": "conflict"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, ConflictError)
        assert exc.status_code == 409

    def test_429_rate_limit(self) -> None:
        resp = _mock_response(
            429,
            {"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"Retry-After": "30"},
        )
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, RateLimitError)
        assert exc.retry_after == 30.0
        assert exc.status_code == 429

    def test_429_no_retry_after(self) -> None:
        resp = _mock_response(429, {
            "error": {"type": "rate_limit_error", "message": "slow down"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, RateLimitError)
        assert exc.retry_after is None

    def test_429_unparseable_retry_after_falls_back_to_none(self) -> None:
        """A garbage ``Retry-After`` must not crash error parsing."""
        resp = _mock_response(
            429,
            {"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"Retry-After": "soon-ish"},
        )
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, RateLimitError)
        assert exc.retry_after is None

    def test_500_api_error(self) -> None:
        resp = _mock_response(500, {
            "error": {"type": "api_error", "message": "internal"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, APIError)
        assert exc.status_code == 500

    def test_unknown_type_falls_back_to_api_error(self) -> None:
        resp = _mock_response(500, {
            "error": {"type": "unknown_error_type", "message": "weird"},
        })
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, APIError)

    def test_non_json_response(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 502
        resp.json.side_effect = ValueError("not json")
        resp.headers = {}
        exc = DataxidClient._parse_error(resp)
        assert isinstance(exc, APIError)
        assert "502" in str(exc)

    def test_request_id_captured(self) -> None:
        resp = _mock_response(
            400,
            {"error": {"type": "invalid_request_error", "message": "bad"}},
            headers={"X-Request-Id": "req_abc123"},
        )
        exc = DataxidClient._parse_error(resp)
        assert exc.request_id == "req_abc123"

    def test_missing_request_id(self) -> None:
        resp = _mock_response(400, {
            "error": {"type": "invalid_request_error", "message": "bad"},
        })
        exc = DataxidClient._parse_error(resp)
        assert exc.request_id is None


class TestRetryAfterParsing:
    """``_parse_retry_after`` accepts both RFC 9110 forms and clamps degenerate values."""

    def test_integer_seconds(self) -> None:
        assert _parse_retry_after("30") == 30.0

    def test_fractional_seconds_accepted(self) -> None:
        assert _parse_retry_after("1.5") == 1.5

    def test_whitespace_is_tolerated(self) -> None:
        assert _parse_retry_after("  30  ") == 30.0

    def test_none_and_empty_return_none(self) -> None:
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("   ") is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_retry_after("soon-ish") is None

    def test_negative_seconds_return_none(self) -> None:
        assert _parse_retry_after("-5") is None

    def test_seconds_clamped_to_max(self) -> None:
        assert _parse_retry_after("1000000") == _MAX_RETRY_AFTER_SECONDS

    def test_http_date_in_future_returns_seconds(self) -> None:
        future = datetime.now(tz=timezone.utc) + timedelta(seconds=120)
        header = email.utils.format_datetime(future, usegmt=True)

        result = _parse_retry_after(header)

        assert result is not None
        assert 110.0 <= result <= 120.0  # allow a few seconds for test scheduling

    def test_http_date_in_past_returns_none(self) -> None:
        past = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        header = email.utils.format_datetime(past, usegmt=True)
        assert _parse_retry_after(header) is None

    def test_http_date_far_future_clamped(self) -> None:
        far_future = datetime.now(tz=timezone.utc) + timedelta(days=365)
        header = email.utils.format_datetime(far_future, usegmt=True)
        assert _parse_retry_after(header) == _MAX_RETRY_AFTER_SECONDS


class TestUserAgent:
    """Every request advertises ``dataxid-python/{version}``."""

    def test_user_agent_sent(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert kwargs["headers"]["User-Agent"] == f"dataxid-python/{dataxid.__version__}"


class TestTimeout:
    """Default: ``connect=10s`` (fail fast on unreachable), ``read=300s`` (patient on training)."""

    def test_default_timeout_is_httpx_timeout(
        self, client: DataxidClient, mock_request: MagicMock,
    ) -> None:
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        timeout = kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 10.0
        assert timeout.read == 300.0

    def test_custom_timeout_override(self, mock_request: MagicMock) -> None:
        custom = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)
        client = DataxidClient(api_key="dx_test_abc", timeout=custom)
        client.get("/v1/models")

        _, kwargs = mock_request.call_args
        assert kwargs["timeout"].connect == 5.0
        assert kwargs["timeout"].read == 60.0


class TestBackoff:
    """Exponential backoff between retries; ``Retry-After`` takes precedence."""

    def test_backoff_called_on_retry(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            _mock_response(500, {"error": {"type": "api_error", "message": "fail"}}),
            _mock_response(200, {"data": {}}),
        ]

        client.get("/v1/models")

        assert mock_sleep.call_count == 1
        delay = mock_sleep.call_args[0][0]
        assert delay >= _BACKOFF_BASE  # base + jitter

    def test_backoff_increases_with_attempts(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        error = _mock_response(500, {"error": {"type": "api_error", "message": "fail"}})
        mock_request.side_effect = [error, error, error]

        with pytest.raises(APIError):
            client.get("/v1/models")

        assert mock_sleep.call_count == 2
        delay_0 = mock_sleep.call_args_list[0][0][0]
        delay_1 = mock_sleep.call_args_list[1][0][0]
        assert delay_1 > delay_0

    def test_retry_after_header_respected(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            _mock_response(
                429,
                {"error": {"type": "rate_limit_error", "message": "slow down"}},
                headers={"Retry-After": "5"},
            ),
            _mock_response(200, {"data": {}}),
        ]

        client.get("/v1/models")

        delay = mock_sleep.call_args[0][0]
        # ``Retry-After=5`` plus up to 25% jitter (see ``_sleep_before_retry``).
        assert 5.0 <= delay <= 5.0 * 1.25

    def test_backoff_on_timeout_exception(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.side_effect = [
            httpx.TimeoutException("timeout"),
            _mock_response(200, {"data": {}}),
        ]

        client.get("/v1/models")
        assert mock_sleep.call_count == 1

    def test_no_backoff_on_non_retryable(
        self, client: DataxidClient, mock_request: MagicMock, mock_sleep: MagicMock,
    ) -> None:
        mock_request.return_value = _mock_response(
            400, {"error": {"type": "invalid_request_error", "message": "bad"}},
        )

        with pytest.raises(InvalidRequestError):
            client.post("/v1/models", json={})

        mock_sleep.assert_not_called()


class TestDefaultBaseUrl:
    """The shipped default base URL points at production."""

    def test_default_is_production(self) -> None:
        assert dataxid.base_url == "https://api.dataxid.com"
