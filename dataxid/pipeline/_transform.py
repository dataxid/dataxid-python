# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Type conversions and data splitting utilities.

Shared by both the analysis (_analyze.py) and encoding (_encode.py) pipelines.
Extracted to avoid _encode.py importing internal helpers from _analyze.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Token constants — shared across analyze + encode
# ---------------------------------------------------------------------------

RARE_TOKEN = "_RARE_"
NULL_TOKEN = "<<NULL>>"
ESCAPE_CHAR = "\x01"

BINNED_UNKNOWN = "<<UNK>>"
BINNED_NULL = "<<NULL>>"
BINNED_MIN = "<<MIN>>"
BINNED_MAX = "<<MAX>>"

CHARACTER_PAD = "\0"
CHARACTER_MAX_LEN = 50

DATETIME_PARTS = [
    "year", "month", "day", "hour", "minute", "second",
    "ms_E2", "ms_E1", "ms_E0",
]

PRECISION_HI = 18
PRECISION_LO = -8


# ---------------------------------------------------------------------------
# Safe type conversions
# ---------------------------------------------------------------------------

def to_string(values: pd.Series) -> pd.Series:
    return values.astype("string")


def to_numeric(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        values = values.astype("Int8")
    elif not pd.api.types.is_integer_dtype(values) and not pd.api.types.is_float_dtype(values):
        pattern = r"(-?[0-9]*[\.]?[0-9]+(?:[eE][+\-]?\d+)?)"
        values = values.astype(str).str.extract(pattern, expand=False)
    return pd.to_numeric(values, errors="coerce")


def to_datetime(values: pd.Series) -> pd.Series:
    if hasattr(values.dtype, "name") and values.dtype.name == "null[pyarrow]":
        values = values.astype("string")
    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed", dayfirst=False)
    fixed = pd.to_datetime(
        values.mask(parsed.isna(), pd.NA), errors="coerce", utc=True, dayfirst=False,
    )
    if fixed.isna().sum() > values.isna().sum():
        alt = pd.to_datetime(values, errors="coerce", utc=True, format="mixed", dayfirst=True)
        if alt.isna().sum() < fixed.isna().sum():
            fixed = alt
    result = fixed.fillna(parsed)
    result = result.dt.tz_localize(None)
    return result.astype("datetime64[us]")


# ---------------------------------------------------------------------------
# Categorical escape (reserved token collision prevention)
# ---------------------------------------------------------------------------

def escape_tokens(values: pd.Series) -> pd.Series:
    reserved = (RARE_TOKEN, NULL_TOKEN)
    replacement = {t: ESCAPE_CHAR + t for t in reserved}
    mask = values.str.startswith(ESCAPE_CHAR, na=False)
    values = values.copy()
    values.loc[mask] = values.loc[mask].str.slice_replace(stop=1, repl=ESCAPE_CHAR * 2)
    values = values.replace(replacement)
    return values


def unescape_tokens(values: pd.Series) -> pd.Series:
    """Reverse escape_tokens — restore original string values."""
    reserved = (RARE_TOKEN, NULL_TOKEN)
    replacement = {ESCAPE_CHAR + t: t for t in reserved}
    values = values.copy()
    values = values.replace(replacement)
    mask = values.str.startswith(ESCAPE_CHAR * 2, na=False)
    values.loc[mask] = values.loc[mask].str.slice_replace(stop=2, repl=ESCAPE_CHAR)
    return values


# ---------------------------------------------------------------------------
# Data splitting — shared by analyze + encode
# ---------------------------------------------------------------------------

def digitize(
    values: pd.Series,
    max_decimal: int = PRECISION_HI,
    min_decimal: int = PRECISION_LO,
) -> pd.DataFrame:
    """Split numeric values into per-digit sub-columns."""
    columns = [f"E{i}" for i in np.arange(max_decimal, min_decimal - 1, -1)]
    if values.isna().all():
        df = pd.DataFrame({c: [0] * len(values) for c in columns})
    else:
        values = values.astype("float64")
        formatted = (
            values.abs()
            .apply(
                lambda x: np.format_float_positional(
                    x, unique=True, pad_left=50, pad_right=20, precision=20,
                )
            )
            .astype("string[pyarrow]")
            .replace("nan", pd.NA)
        )
        formatted = formatted.str.replace(" ", "0")
        formatted = formatted.str.replace(".", "", n=1, regex=False)
        formatted = formatted.str[(49 - max_decimal):(49 - min_decimal + 1)]
        df = formatted.str.split("", n=max_decimal - min_decimal + 2, expand=True)
        df = df.drop(columns=[0, max_decimal - min_decimal + 2])
        df = df.fillna("0")
        df.columns = columns
    df.insert(0, "nan", values.isna())
    df.insert(1, "neg", (~values.isna()) & (values < 0))
    df = df.astype("int")
    return df


def split_datetime(values: pd.Series) -> pd.DataFrame:
    """Split datetime values into year/month/day/... sub-columns."""
    values = values.astype("datetime64[us]")
    dt = values.to_numpy()
    parts = np.empty(dt.shape + (7,), dtype="u4")
    year, month, day, hour, minute, second = (dt.astype(f"M8[{x}]") for x in "YMDhms")
    parts[:, 0] = year + 1970
    parts[:, 1] = (month - year) + 1
    parts[:, 2] = (day - month) + 1
    parts[:, 3] = (dt - day).astype("m8[h]")
    parts[:, 4] = (dt - hour).astype("m8[m]")
    parts[:, 5] = (dt - minute).astype("m8[s]")
    parts[:, 6] = (dt - second).astype("m8[us]")
    sub = {
        "nan": values.reset_index(drop=True).isna(),
        "year": pd.Series(parts[:, 0]),
        "month": pd.Series(parts[:, 1]),
        "day": pd.Series(parts[:, 2]),
        "hour": pd.Series(parts[:, 3]),
        "minute": pd.Series(parts[:, 4]),
        "second": pd.Series(parts[:, 5]),
        "ms_E2": pd.Series(np.floor(parts[:, 6] / 100_000) % 10),
        "ms_E1": pd.Series(np.floor((parts[:, 6] / 10_000) % 10)),
        "ms_E0": pd.Series(np.floor((parts[:, 6] / 1_000) % 10)),
    }
    return pd.DataFrame(sub).fillna(0).astype("int")


def split_chars(values: pd.Series, max_len: int | None = None) -> pd.DataFrame:
    """Split string values into per-position character sub-columns."""
    is_na = pd.Series(values.isna().astype("int"), name="nan").to_frame()
    values = values.fillna("")
    values = values.str.slice(stop=CHARACTER_MAX_LEN)
    if max_len is None:
        ml = values.str.len().max()
        max_len = int(ml) if np.isscalar(ml) and not np.isnan(ml) else 0
    else:
        values = values.str.slice(stop=max_len)
    padded = values.str.ljust(max_len, CHARACTER_PAD)
    chars_df = padded.str.split("", expand=True)
    if not chars_df.empty:
        chars_df = chars_df.drop([0, max_len + 1], axis=1)
        chars_df.columns = [f"P{i}" for i in range(max_len)]
    else:
        chars_df = pd.DataFrame(columns=[f"P{i}" for i in range(max_len)])
    return pd.concat([is_na, chars_df], axis=1)
