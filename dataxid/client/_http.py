# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
HTTP client for DataXID API.

Handles auth, retry, timeout, idempotency keys, and structured error parsing.
"""

import os
import random
import time
import uuid
from typing import Any

import httpx

import dataxid
from dataxid.exceptions import (
    APIError,
    AuthenticationError,
    ConflictError,
    DataxidError,
    InvalidRequestError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
)

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0)
_MAX_RETRIES = 3
_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 8.0

_ERROR_MAP = {
    "authentication_error": AuthenticationError,
    "invalid_request_error": InvalidRequestError,
    "not_found_error": NotFoundError,
    "quota_exceeded": QuotaExceededError,
    "rate_limit_error": RateLimitError,
}


class DataxidClient:
    """Low-level HTTP client. Used internally by Model and synthesize()."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout = _DEFAULT_TIMEOUT,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    @property
    def api_key(self) -> str:
        key = self._api_key or dataxid.api_key or os.environ.get("DATAXID_API_KEY")
        if not key:
            raise AuthenticationError(
                "No API key provided. Set dataxid.api_key, pass api_key= to the client, "
                "or set the DATAXID_API_KEY environment variable."
            )
        return key

    @property
    def base_url(self) -> str:
        return self._base_url or dataxid.base_url

    def post(
        self, path: str, json: dict[str, Any] | None = None, idempotent: bool = True,
    ) -> dict[str, Any]:
        return self._request("POST", path, json=json, idempotent=idempotent)  # type: ignore[return-value]

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request("GET", path, params=params)  # type: ignore[return-value]

    def delete(self, path: str) -> None:
        self._request("DELETE", path)

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotent: bool = False,
    ) -> dict | None:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"dataxid-python/{dataxid.__version__}",
        }
        if idempotent and method == "POST":
            headers["Idempotency-Key"] = uuid.uuid4().hex

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = httpx.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )

                if response.status_code == 204:
                    return None

                if response.status_code < 400:
                    return response.json()

                if response.status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES - 1:
                    last_exc = self._parse_error(response)
                    self._sleep_before_retry(attempt, response)
                    continue

                raise self._parse_error(response)

            except httpx.TimeoutException as e:
                last_exc = APIError(f"Request timed out: {e}", status_code=None)
                if attempt < _MAX_RETRIES - 1:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_exc from e

            except httpx.ConnectError as e:
                last_exc = APIError(f"Connection failed: {e}", status_code=None)
                if attempt < _MAX_RETRIES - 1:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_exc from e

        raise last_exc or APIError("Request failed after retries")

    @staticmethod
    def _sleep_before_retry(
        attempt: int, response: httpx.Response | None = None,
    ) -> None:
        retry_after = None
        if response is not None:
            raw = response.headers.get("Retry-After")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None

        if retry_after is not None and retry_after > 0:
            delay = retry_after
        else:
            delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_MAX)

        jitter = random.uniform(0, delay * 0.25)  # noqa: S311
        time.sleep(delay + jitter)

    @staticmethod
    def _parse_error(response: httpx.Response) -> DataxidError:
        request_id = response.headers.get("X-Request-Id")

        try:
            body = response.json()
            error = body.get("error", {})
        except Exception:
            return APIError(
                f"HTTP {response.status_code}",
                status_code=response.status_code,
                request_id=request_id,
            )

        error_type = error.get("type", "api_error")
        message = error.get("message", f"HTTP {response.status_code}")

        exc_class = _ERROR_MAP.get(error_type, APIError)

        kwargs: dict[str, Any] = {
            "status_code": response.status_code,
            "request_id": request_id,
        }

        if exc_class == InvalidRequestError:
            kwargs["param"] = error.get("param")
        elif exc_class == RateLimitError:
            retry_after = response.headers.get("Retry-After")
            kwargs["retry_after"] = float(retry_after) if retry_after else None
        elif exc_class == QuotaExceededError:
            kwargs["usage"] = error.get("usage")
            kwargs["upgrade_url"] = error.get("upgrade_url")

        if response.status_code == 409:
            return ConflictError(message, **kwargs)

        return exc_class(message, **kwargs)
