# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
SDK exception hierarchy.
"""


class DataxidError(Exception):
    """Base exception for all Dataxid SDK errors."""
    def __init__(self, message: str, status_code: int | None = None, request_id: str | None = None):
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(message)


class AuthenticationError(DataxidError):
    """Invalid or missing API key."""
    pass


class InvalidRequestError(DataxidError):
    """Bad request parameters."""
    def __init__(self, message: str, param: str | None = None, **kwargs):
        self.param = param
        super().__init__(message, **kwargs)


class NotFoundError(DataxidError):
    """Resource not found."""
    pass


class RateLimitError(DataxidError):
    """Too many requests."""
    def __init__(self, message: str, retry_after: float | None = None, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class QuotaExceededError(DataxidError):
    """Monthly usage quota exceeded — upgrade required."""
    def __init__(
        self, message: str,
        usage: dict | None = None,
        upgrade_url: str | None = None,
        **kwargs,
    ):
        self.usage = usage
        self.upgrade_url = upgrade_url
        super().__init__(message, **kwargs)


class ConflictError(DataxidError):
    """Idempotency conflict — request already in progress."""
    pass


class ModelNotReadyError(DataxidError):
    """Method called before the model was initialized (e.g. analyze() not called)."""
    pass


class TrainingTimeoutError(DataxidError):
    """Training exceeded maximum wait time."""
    pass


class TrainingError(DataxidError):
    """Training failed."""
    pass


class APIError(DataxidError):
    """Unexpected server error."""
    pass
