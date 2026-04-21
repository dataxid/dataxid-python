# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Encoding pipeline for the DataXID SDK.

Converts raw tabular data into integer-coded sub-columns using the per-column
statistics produced by the analysis step. Each sub-column maps to one
embedding in the encoder network.

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
# Imputation helpers
# ===================================================================

def _build_null_masks(
    conditions: pd.DataFrame,
    conditions_features: list[str],
    column_stats: dict[str, dict],
    imputed_columns: list[str],
) -> dict[str, np.ndarray]:
    """Return ``{wire_key: bool_mask}`` where True = originally NaN in an imputed column.

    For each imputed feature that is part of *conditions_features*, checks
    which rows in ``conditions`` are NaN and maps them to the corresponding
    wire-key sub-columns. The mask is used to replace NULL token codes with
    the ``-1`` sentinel so the decoder samples freely.
    """
    masks: dict[str, np.ndarray] = {}
    for feature in imputed_columns:
        if feature not in conditions_features or feature not in column_stats:
            continue
        is_null = conditions[feature].isna().values
        if not is_null.any():
            continue
        stats = column_stats[feature]
        raw_et = stats.get("encoding_type", EncodingType.categorical)
        et = EncodingType(raw_et) if isinstance(raw_et, str) else raw_et
        if et == EncodingType.categorical:
            masks[wire_key(feature, "cat")] = is_null
        elif et == EncodingType.numeric_discrete:
            masks[wire_key(feature, "cat")] = is_null
        elif et == EncodingType.numeric_binned:
            masks[wire_key(feature, "bin")] = is_null
        elif stats.get("has_nan"):
            masks[wire_key(feature, "nan")] = is_null
    return masks


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


def encode_conditions_fixed_values(
    conditions: pd.DataFrame,
    features: list[str],
    column_stats: dict[str, dict],
    imputed_columns: list[str] | None = None,
) -> dict[str, list[int]]:
    """Encode conditions into wire-key fixed values for non-sequential generation.

    Only columns present in *both* ``conditions`` and ``features`` are encoded.
    The result is JSON-serialisable (``list[int]`` values) so it can be sent
    directly in the API payload.

    When *imputed_columns* is given, NULL cells in those columns are encoded as
    ``-1`` (sentinel) instead of the NULL token code, allowing the decoder to
    sample freely while ``fixed_probs`` suppresses the NULL token.

    Args:
        conditions: DataFrame whose columns are a subset of the training features.
        features: All training feature names (from ``analyze()``).
        column_stats: Column statistics from ``analyze()``.
        imputed_columns: Feature names being imputed — their NaN cells become ``-1``.

    Returns:
        ``{wire_key: [int, ...]}`` — one entry per sub-column of the conditions features.
    """
    conditions_features = [f for f in features if f in conditions.columns]
    if not conditions_features:
        return {}

    encoded = encode_columns(conditions, conditions_features, column_stats)

    if imputed_columns:
        null_masks = _build_null_masks(conditions, conditions_features, column_stats, imputed_columns)
        for wk, mask in null_masks.items():
            if wk in encoded:
                arr = encoded[wk].copy()
                arr[mask] = -1
                encoded[wk] = arr

    return {k: v.tolist() for k, v in encoded.items()}


def encode_sequential_conditions_fixed_values(
    conditions: pd.DataFrame,
    features: list[str],
    column_stats: dict[str, dict],
    context_key: str,
    n_entities: int,
    entity_order: list | np.ndarray | None = None,
    imputed_columns: list[str] | None = None,
) -> tuple[dict[str, list[list[int]]], int]:
    """Encode long-format conditions DataFrame into 2D fixed_values for sequential generation.

    Each row in ``conditions`` is one time step. Rows are grouped by
    ``context_key`` and aligned to ``entity_order``. Entities without
    condition rows and shorter sequences are right-padded with ``-1``
    (sentinel) to ``n_conditions_steps``. Sentinel values are skipped
    during generation, allowing those entity-steps to be freely sampled.

    When *imputed_columns* is given, NULL cells in those columns are encoded
    as ``-1`` instead of the NULL token code, so the decoder samples freely
    while ``fixed_probs`` suppresses the NULL token.

    Returns:
        Tuple of ``({wire_key: [[step0, step1, ...], ...]}, n_conditions_steps)``.
        Outer list has ``n_entities`` entries; inner lists have ``n_conditions_steps``.
    """
    conditions_features = [f for f in features if f in conditions.columns and f != context_key]
    if not conditions_features:
        return {}, 0

    encoded = encode_columns(conditions, conditions_features, column_stats)

    if imputed_columns:
        null_masks = _build_null_masks(conditions, conditions_features, column_stats, imputed_columns)
        for wk, mask in null_masks.items():
            if wk in encoded:
                arr = encoded[wk].copy()
                arr[mask] = -1
                encoded[wk] = arr

    groups = conditions.groupby(context_key, sort=False)
    n_conditions_steps = groups.size().max()

    if entity_order is None:
        entity_order = conditions[context_key].unique()

    result: dict[str, list[list[int]]] = {k: [] for k in encoded}
    row_offset = 0
    key_to_rows: dict = {}
    for ek, grp in groups:
        key_to_rows[ek] = (row_offset, len(grp))
        row_offset += len(grp)

    for entity_idx in range(n_entities):
        ek = entity_order[entity_idx] if entity_idx < len(entity_order) else None
        for wk, arr in encoded.items():
            if ek is not None and ek in key_to_rows:
                start, length = key_to_rows[ek]
                steps = arr[start:start + length].tolist()
            else:
                steps = []
            padded = steps + [-1] * (n_conditions_steps - len(steps))
            result[wk].append(padded)

    return result, int(n_conditions_steps)
