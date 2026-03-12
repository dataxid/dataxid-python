# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
API client — HTTP transport and wire-format serialization.

Handles auth, retry, idempotency, and binary embedding payloads.
"""

from dataxid.client._http import DataxidClient

__all__ = ["DataxidClient"]
