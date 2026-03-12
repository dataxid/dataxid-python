# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Encoding pipeline for the Dataxid SDK.

Converts raw tabular data into integer-coded sub-columns using the column_stats
produced by the analysis pipeline (_analyze.py). Each sub-column maps to one
nn.Embedding in the encoder network.

Supported encoding types:
  - categorical
  - numeric: binned, discrete, digit
  - datetime
  - character
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dataxid.encoder._ports import EncodingType, wire_key
from dataxid.pipeline._transform import (
    BINNED_MAX,
    BINNED_MIN,
    BINNED_NULL,
    BINNED_UNKNOWN,
    DATETIME_PARTS,
    NULL_TOKEN,
    RARE_TOKEN,
    digitize,
    escape_tokens,
    split_chars,
    split_datetime,
    to_datetime,
    to_numeric,
    to_string,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_missing(values: pd.Series, stats: dict) -> tuple[pd.Series, pd.Series]:  # noqa: ARG001
    """Fill NaN values by sampling from the non-null distribution."""
    nan_mask = values.isna()
    vc = values.value_counts(normalize=True)
    if vc.empty:
        return values, nan_mask.astype(int)
    values = values.copy()
    values[nan_mask] = np.random.choice(vc.index, size=nan_mask.sum(), p=vc.values)
    return values, nan_mask.astype(int)


def _safe_series(arr, pd_dtype: str) -> pd.Series:
    """Create a numeric Series with safe int overflow handling."""
    np_dtype = int if pd_dtype == "Int64" else float
    i_min, i_max = np.iinfo(int).min, np.iinfo(int).max

    def _clip_int(vals):
        return [min(max(int(v), i_min), i_max) for v in vals]

    if isinstance(arr, list):
        if np_dtype is int:
            arr = _clip_int(arr)
        arr = np.array(arr)
    elif np_dtype is int:
        try:
            arr.astype(int, casting="safe")
        except TypeError:
            arr = np.array(_clip_int(arr))
    return pd.Series(np.array([v for v in arr]).astype(np_dtype), dtype=pd_dtype)


def _cast_precision(values: pd.Series, min_decimal: int) -> pd.Series:
    """Cast numeric values to Int64 or Float64 based on decimal precision."""
    dtype = "Float64" if min_decimal < 0 else "Int64"
    if dtype == "Int64":
        values = values.round()
    try:
        values = values.astype(dtype)
    except TypeError:
        if dtype == "Int64":
            values = values.astype("Float64")
    return values


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def _encode_categorical(values: pd.Series, stats: dict) -> pd.DataFrame:
    values = escape_tokens(to_string(values))
    known = [str(k) for k in stats["codes"].keys()]
    values = values.copy()
    if NULL_TOKEN in known:
        values[values.isna()] = NULL_TOKEN
    values[~values.isin(known)] = RARE_TOKEN
    codes = pd.Series(
        pd.Categorical(values, categories=known).codes,
        name="cat", index=values.index,
    )
    return codes.to_frame()


# ---------------------------------------------------------------------------
# Numeric — discrete
# ---------------------------------------------------------------------------

def _encode_numeric_discrete(values: pd.Series, stats: dict) -> pd.DataFrame:
    values = _cast_precision(values, stats["min_decimal"])
    return _encode_categorical(values, stats)


# ---------------------------------------------------------------------------
# Numeric — digit
# ---------------------------------------------------------------------------

def _encode_numeric_digit(values: pd.Series, stats: dict) -> pd.DataFrame:
    values = _cast_precision(values, stats["min_decimal"])
    values = values.reset_index(drop=True)
    if stats["min"] is not None:
        rm = _safe_series([stats["min"]], values.dtype).iloc[0]
        values = values.where((values.isna()) | (values >= rm), rm)
    if stats["max"] is not None:
        rx = _safe_series([stats["max"]], values.dtype).iloc[0]
        values = values.where((values.isna()) | (values <= rx), rx)
    values, nan_mask = _fill_missing(values, stats)
    df = digitize(values, stats["max_decimal"], stats["min_decimal"])
    for d in np.arange(stats["max_decimal"], stats["min_decimal"] - 1, -1):
        key = f"E{d}"
        df[key] = df[key] - stats["min_digits"][key]
        df[key] = np.minimum(df[key], stats["max_digits"][key] - stats["min_digits"][key])
        df[key] = np.maximum(df[key], 0)
    for d in np.arange(stats["max_decimal"], stats["min_decimal"] - 1, -1):
        df[f"E{d}"] = np.minimum(df[f"E{d}"], stats["max_digits"][f"E{d}"])
    if not stats["has_neg"]:
        df.drop("neg", inplace=True, axis=1)
    if stats["has_nan"]:
        df["nan"] = nan_mask
    else:
        df.drop("nan", inplace=True, axis=1)
    return df


# ---------------------------------------------------------------------------
# Numeric — binned
# ---------------------------------------------------------------------------

def _encode_numeric_binned(values: pd.Series, stats: dict) -> pd.DataFrame:
    bins = stats["bins"].copy()
    bins[0] = -np.inf
    bins[-1] = np.inf
    codes = pd.Series(
        pd.cut(values, bins=bins, right=False).cat.codes,
        name="cat", index=values.index,
    )
    offset = len(stats["codes"])
    codes = codes + offset
    if BINNED_NULL in stats["codes"]:
        codes.mask(values.isna(), stats["codes"][BINNED_NULL], inplace=True)
    else:
        codes.mask(values.isna(), stats["codes"][BINNED_UNKNOWN], inplace=True)
    if BINNED_MIN in stats["codes"]:
        codes.mask(values == stats["bins"][0], stats["codes"][BINNED_MIN], inplace=True)
    if BINNED_MAX in stats["codes"]:
        codes.mask(values == stats["bins"][-1], stats["codes"][BINNED_MAX], inplace=True)
    return pd.DataFrame({"bin": codes})


# ---------------------------------------------------------------------------
# Numeric — dispatcher
# ---------------------------------------------------------------------------

def _encode_numeric(values: pd.Series, stats: dict) -> pd.DataFrame:
    values = to_numeric(values)
    raw = stats["encoding_type"]
    et = EncodingType(raw) if isinstance(raw, str) else raw
    if et == EncodingType.numeric_discrete:
        return _encode_numeric_discrete(values, stats)
    elif et == EncodingType.numeric_digit:
        return _encode_numeric_digit(values, stats)
    elif et == EncodingType.numeric_binned:
        return _encode_numeric_binned(values, stats)
    else:
        raise ValueError(f"Unknown numeric encoding type: {et}")


# ---------------------------------------------------------------------------
# Datetime
# ---------------------------------------------------------------------------

def _encode_datetime(values: pd.Series, stats: dict) -> pd.DataFrame:
    values = to_datetime(values).copy().reset_index(drop=True)
    if stats["min"] is not None:
        rm = pd.Series([stats["min"]], dtype=values.dtype).iloc[0]
        values.loc[values < rm] = rm
    if stats["max"] is not None:
        rx = pd.Series([stats["max"]], dtype=values.dtype).iloc[0]
        values.loc[values > rx] = rx
    values, nan_mask = _fill_missing(values, stats)
    df = split_datetime(values)
    for key in DATETIME_PARTS:
        df[key] = df[key] - stats["min_values"][key]
        df[key] = np.minimum(df[key], stats["max_values"][key] - stats["min_values"][key])
        df[key] = np.maximum(df[key], 0)
    if not stats["has_time"]:
        df.drop(["hour", "minute", "second"], inplace=True, axis=1)
    if not stats["has_ms"]:
        df.drop(["ms_E2", "ms_E1", "ms_E0"], inplace=True, axis=1)
    if stats["has_nan"]:
        df["nan"] = nan_mask
    else:
        df.drop(["nan"], inplace=True, axis=1)
    return df


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------

def _encode_character(values: pd.Series, stats: dict) -> pd.DataFrame:
    values = to_string(values)
    values, nan_mask = _fill_missing(values, stats)
    max_len = stats["max_string_length"]
    df_split = split_chars(values, max_len)
    for idx in range(max_len):
        sc = f"P{idx}"
        np_codes = np.array(pd.Categorical(df_split[sc], categories=stats["codes"][sc]).codes)
        np.place(np_codes, np_codes == -1, 0)
        df_split[sc] = np_codes
    if stats["has_nan"]:
        df_split["nan"] = nan_mask
    else:
        df_split.drop(["nan"], axis=1, inplace=True)
    return df_split


# ===================================================================
# Public API
# ===================================================================

def encode_columns(
    df: pd.DataFrame,
    features: list[str],
    column_stats: dict[str, dict],
) -> dict[str, np.ndarray]:
    """
    Encode raw features into integer-coded sub-column arrays.

    Uses column_stats from compute_stats() to map each value to its
    integer code. Output keys follow the wire protocol naming convention:
    ``feat:/{feature}__{sub_col}``.
    """
    encoded: dict[str, np.ndarray] = {}

    for feature in features:
        if feature not in df.columns or feature not in column_stats:
            continue

        stats = column_stats[feature]
        raw = stats.get("encoding_type", EncodingType.categorical)
        et = EncodingType(raw) if isinstance(raw, str) else raw
        series = df[feature]

        if et == EncodingType.categorical:
            df_enc = _encode_categorical(series, stats)
        elif et in (
            EncodingType.numeric_auto, EncodingType.numeric_binned,
            EncodingType.numeric_discrete, EncodingType.numeric_digit,
        ):
            df_enc = _encode_numeric(series, stats)
        elif et == EncodingType.datetime:
            df_enc = _encode_datetime(series, stats)
        elif et == EncodingType.character:
            df_enc = _encode_character(series, stats)
        else:
            logger.warning(
                "Unsupported encoding type %s for %s, falling back to categorical",
                et, feature,
            )
            df_enc = _encode_categorical(series, stats)

        for col in df_enc.columns:
            encoded[wire_key(feature, col)] = df_enc[col].values

    return encoded
