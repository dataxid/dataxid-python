# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Primary key generators for synthetic tables.

Three supported formats:
    - ``dxid``: ``dxid_`` + 22 char base62 (128-bit entropy). Default.
    - ``int``:  1-based auto-increment integers.
    - ``uuid``: standard UUID v4 strings.

The ``dxid`` and ``int`` generators honour the global ``random`` state, so
callers that seed the RNG (e.g. via ``config.seed``) obtain reproducible
primary keys. ``uuid`` uses Python's ``uuid.uuid4`` which draws from its own
random source and is therefore non-deterministic even under a seed.
"""

from __future__ import annotations

import random
import string
import uuid
from typing import Literal, assert_never, get_args

PkType = Literal["dxid", "int", "uuid"]
PK_TYPES: tuple[str, ...] = get_args(PkType)

_BASE62_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase
_DXID_PREFIX = "dxid_"
_DXID_BODY_LEN = 22


def generate_dxid() -> str:
    """Return a new ``dxid_`` primary key with 128 bits of entropy."""
    n = random.getrandbits(128)
    chars = [""] * _DXID_BODY_LEN
    for i in range(_DXID_BODY_LEN - 1, -1, -1):
        n, rem = divmod(n, 62)
        chars[i] = _BASE62_ALPHABET[rem]
    return _DXID_PREFIX + "".join(chars)


def generate_uuid() -> str:
    """Return a new UUID v4 string."""
    return str(uuid.uuid4())


def generate_primary_keys(pk_type: PkType, n: int) -> list[str] | list[int]:
    """Return ``n`` primary keys for the requested ``pk_type``.

    Callers must pass a value validated against :data:`PK_TYPES` (e.g. via
    :class:`~dataxid.Table` construction). Unknown values are surfaced as a
    static-analysis error through :func:`typing.assert_never`.
    """
    match pk_type:
        case "dxid":
            return [generate_dxid() for _ in range(n)]
        case "uuid":
            return [generate_uuid() for _ in range(n)]
        case "int":
            return list(range(1, n + 1))
    assert_never(pk_type)
