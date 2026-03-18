# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Decoding pipeline for the Dataxid SDK.

Converts raw integer codes (from server generation) back to original feature
values using column_stats. This is the inverse of _encode.py.

Privacy boundary: the server returns opaque integer codes. Decoding happens
client-side because only the client holds column_stats and value_mappings.
"""

from __future__ import annotations

import calendar
import logging

import numpy as np
import pandas as pd

from dataxid.encoder._ports import EncodingType, wire_key
from dataxid.pipeline._transform import (
    BINNED_MAX,
    BINNED_MIN,
    BINNED_NULL,
    BINNED_UNKNOWN,
    CHARACTER_PAD,
    NULL_TOKEN,
    RARE_TOKEN,
    unescape_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def _decode_categorical(codes: np.ndarray, stats: dict) -> pd.Series:
    categories = list(stats["codes"].keys())
    values = pd.Series(
        pd.Categorical.from_codes(codes, categories=categories),
        dtype="string",
    )
    values[values == NULL_TOKEN] = pd.NA
    values = unescape_tokens(values)
    return values


# ---------------------------------------------------------------------------
# Numeric — discrete
# ---------------------------------------------------------------------------

def _decode_numeric_discrete(codes: np.ndarray, stats: dict) -> pd.Series:
    dtype = "Float64" if stats["min_decimal"] < 0 else "Int64"
    values = _decode_categorical(codes, stats)

    is_rare = values == RARE_TOKEN
    cnt_rare = is_rare.sum()
    if cnt_rare > 0:
        if NULL_TOKEN in stats["codes"]:
            values[is_rare] = pd.NA
        else:
            valid = [
                str(k) for k in stats["codes"]
                if k not in (RARE_TOKEN, NULL_TOKEN)
            ]
            if valid:
                values[is_rare] = np.random.choice(valid, size=cnt_rare)
            else:
                values[is_rare] = pd.NA

    values = pd.to_numeric(values, errors="coerce")
    return values.astype(dtype)


# ---------------------------------------------------------------------------
# Numeric — binned
# ---------------------------------------------------------------------------

def _decode_numeric_binned(codes: np.ndarray, stats: dict) -> pd.Series:
    dtype = "Float64" if stats["min_decimal"] < 0 else "Int64"
    bins = stats["bins"]
    code_map = stats["codes"]
    offset = len(code_map)

    values = pd.Series(pd.NA, index=range(len(codes)), dtype="Float64")

    if BINNED_NULL in code_map:
        values[codes == code_map[BINNED_NULL]] = pd.NA
    if BINNED_MIN in code_map:
        values[codes == code_map[BINNED_MIN]] = bins[0]
    if BINNED_MAX in code_map:
        values[codes == code_map[BINNED_MAX]] = bins[-1]

    for i in range(len(bins) - 1):
        bin_code = i + offset
        mask = codes == bin_code
        n = mask.sum()
        if n > 0:
            draws = np.random.uniform(bins[i], bins[i + 1], n)
            scaler = 10 ** -stats["min_decimal"]
            draws = np.floor(draws * scaler) / scaler
            values.loc[mask] = draws

    return values.astype(dtype)


# ---------------------------------------------------------------------------
# Numeric — digit
# ---------------------------------------------------------------------------

def _decode_numeric_digit(sub_cols: dict[str, np.ndarray], stats: dict) -> pd.Series:
    max_decimal = stats["max_decimal"]
    min_decimal = stats["min_decimal"]
    dtype = "Float64" if min_decimal < 0 else "Int64"

    digit_sum = np.zeros(len(next(iter(sub_cols.values()))), dtype="float64")
    for d in np.arange(max_decimal, min_decimal - 1, -1):
        key = f"E{d}"
        digit_sum += (sub_cols[key] + stats["min_digits"][key]) * (10 ** int(d))

    values = pd.array(digit_sum, dtype=dtype)
    values = pd.Series(values)

    if "nan" in sub_cols:
        values[sub_cols["nan"] == 1] = pd.NA
    if "neg" in sub_cols:
        values[sub_cols["neg"] == 1] = -1 * values[sub_cols["neg"] == 1]

    if stats["min"] is not None and stats["max"] is not None:
        too_low = values.notna() & (values < stats["min"])
        too_high = values.notna() & (values > stats["max"])
        values.loc[too_low] = stats["min"]
        values.loc[too_high] = stats["max"]
    elif "nan" in sub_cols:
        values[sub_cols["nan"] == 0] = pd.NA

    values = np.round(values, -min_decimal)
    return values


# ---------------------------------------------------------------------------
# Datetime
# ---------------------------------------------------------------------------

def _decode_datetime(sub_cols: dict[str, np.ndarray], stats: dict) -> pd.Series:
    y = pd.Series(sub_cols["year"] + stats["min_values"]["year"])
    m = pd.Series(sub_cols["month"] + stats["min_values"]["month"])
    d = pd.Series(sub_cols["day"] + stats["min_values"]["day"])

    is_leap = y.apply(calendar.isleap)
    d[(is_leap) & (m == 2) & (d > 29)] = 29
    d[(~is_leap) & (m == 2) & (d > 28)] = 28
    d[((m == 4) | (m == 6) | (m == 9) | (m == 11)) & (d > 30)] = 30

    y_str = y.astype(str)
    m_str = m.astype(str).str.zfill(2)
    d_str = d.astype(str).str.zfill(2)
    values = y_str + "-" + m_str + "-" + d_str

    has_time = stats.get("has_time", False)
    has_ms = stats.get("has_ms", False)

    if has_time:
        hh = pd.Series(sub_cols["hour"] + stats["min_values"]["hour"]).astype(str).str.zfill(2)
        mm = pd.Series(sub_cols["minute"] + stats["min_values"]["minute"]).astype(str).str.zfill(2)
        ss = pd.Series(sub_cols["second"] + stats["min_values"]["second"]).astype(str).str.zfill(2)
        values = values + " " + hh + ":" + mm + ":" + ss

    if has_ms:
        ms2 = sub_cols["ms_E2"] + stats["min_values"]["ms_E2"]
        ms1 = sub_cols["ms_E1"] + stats["min_values"]["ms_E1"]
        ms0 = sub_cols["ms_E0"] + stats["min_values"]["ms_E0"]
        ms = pd.Series(100 * ms2 + 10 * ms1 + ms0).astype(str).str.zfill(3)
        values = values + "." + ms

    if "nan" in sub_cols:
        values[sub_cols["nan"] == 1] = pd.NA

    if stats.get("min") is not None and stats.get("max") is not None:
        dt_fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
        reduced_min = pd.to_datetime(stats["min"]).strftime(dt_fmt)
        reduced_max = pd.to_datetime(stats["max"]).strftime(dt_fmt)
        too_low = values.notna() & (values < reduced_min)
        too_high = values.notna() & (values > reduced_max)
        values.loc[too_low] = reduced_min
        values.loc[too_high] = reduced_max
    elif "nan" in sub_cols:
        values[sub_cols["nan"] == 0] = pd.NA

    values = pd.to_datetime(values)
    if not has_time:
        values = pd.to_datetime(values.dt.date)
    return values


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------

def _decode_character(sub_cols: dict[str, np.ndarray], stats: dict) -> pd.Series:
    codes_map = stats.get("codes", {})
    positions = [k for k in codes_map if k.startswith("P")]

    if not positions:
        return pd.Series(pd.NA, index=range(len(next(iter(sub_cols.values())))))

    df_decoded = pd.DataFrame({
        pos: pd.Series(
            pd.Categorical.from_codes(sub_cols[pos], categories=list(codes_map[pos].keys())),
            dtype="string",
        )
        for pos in positions
    })
    values = df_decoded.apply(
        lambda row: "".join(row), axis=1, result_type="reduce",
    ).astype(str)
    values = values.str.replace(CHARACTER_PAD, "", regex=False).str.rstrip()

    if stats.get("has_nan") and "nan" in sub_cols:
        values[sub_cols["nan"] == 1] = pd.NA

    return values


# ---------------------------------------------------------------------------
# Fixed probs — suppress UNKNOWN/RARE tokens
# ---------------------------------------------------------------------------

def compute_fixed_probs(
    column_stats: dict[str, dict],
) -> dict[str, dict[int, float]]:
    """
    Build fixed_probs dict to suppress UNKNOWN/RARE tokens during generation.

    Returns mapping: {wire_key: {token_code: 0.0}} — passed to server.generate().
    """
    fixed_probs: dict[str, dict[int, float]] = {}

    for feature, stats in column_stats.items():
        raw_et = stats.get("encoding_type", EncodingType.categorical)
        et = EncodingType(raw_et) if isinstance(raw_et, str) else raw_et
        codes = stats.get("codes", {})
        if not codes:
            continue

        if et == EncodingType.categorical:
            if RARE_TOKEN in codes:
                fixed_probs[wire_key(feature, "cat")] = {codes[RARE_TOKEN]: 0.0}

        elif et == EncodingType.numeric_binned:
            if BINNED_UNKNOWN in codes:
                fixed_probs[wire_key(feature, "bin")] = {codes[BINNED_UNKNOWN]: 0.0}

        elif et == EncodingType.numeric_discrete:
            if RARE_TOKEN in codes:
                fixed_probs[wire_key(feature, "cat")] = {codes[RARE_TOKEN]: 0.0}

    return fixed_probs


# ===================================================================
# Public API — dispatcher
# ===================================================================

def decode_columns(
    raw_codes: dict[str, list[int] | np.ndarray],
    features: list[str],
    column_stats: dict[str, dict],
) -> pd.DataFrame:
    """
    Decode raw integer codes back to original feature values.

    Inverse of encode_columns(). Uses column_stats to map each integer code
    back to its original value.

    Args:
        raw_codes: {wire_key: array_of_int_codes} from server generation
        features: Feature names in original order
        column_stats: Per-feature encoding statistics (from analyze())

    Returns:
        DataFrame with decoded feature values
    """
    decoded: dict[str, pd.Series] = {}

    for feature in features:
        if feature not in column_stats:
            continue

        stats = column_stats[feature]
        raw_et = stats.get("encoding_type", EncodingType.categorical)
        et = EncodingType(raw_et) if isinstance(raw_et, str) else raw_et

        if et == EncodingType.categorical:
            key = wire_key(feature, "cat")
            if key not in raw_codes:
                continue
            arr = np.asarray(raw_codes[key])
            decoded[feature] = _decode_categorical(arr, stats)

        elif et == EncodingType.numeric_discrete:
            key = wire_key(feature, "cat")
            if key not in raw_codes:
                continue
            arr = np.asarray(raw_codes[key])
            decoded[feature] = _decode_numeric_discrete(arr, stats)

        elif et == EncodingType.numeric_binned:
            key = wire_key(feature, "bin")
            if key not in raw_codes:
                continue
            arr = np.asarray(raw_codes[key])
            decoded[feature] = _decode_numeric_binned(arr, stats)

        elif et == EncodingType.numeric_digit:
            sub_cols = _collect_sub_cols(feature, stats, raw_codes)
            if not sub_cols:
                continue
            decoded[feature] = _decode_numeric_digit(sub_cols, stats)

        elif et == EncodingType.datetime:
            sub_cols = _collect_sub_cols(feature, stats, raw_codes)
            if not sub_cols:
                continue
            decoded[feature] = _decode_datetime(sub_cols, stats)

        elif et == EncodingType.character:
            sub_cols = _collect_sub_cols(feature, stats, raw_codes)
            if not sub_cols:
                continue
            decoded[feature] = _decode_character(sub_cols, stats)

        else:
            key = wire_key(feature, "cat")
            if key not in raw_codes:
                continue
            arr = np.asarray(raw_codes[key])
            decoded[feature] = _decode_categorical(arr, stats)

    return pd.DataFrame(decoded)


def _collect_sub_cols(
    feature: str,
    stats: dict,
    raw_codes: dict[str, list[int] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Gather all sub-column arrays for a multi-sub-column encoding type."""
    sub_cols: dict[str, np.ndarray] = {}
    for sub_col in stats.get("cardinalities", {}):
        key = wire_key(feature, sub_col)
        if key in raw_codes:
            sub_cols[sub_col] = np.asarray(raw_codes[key])
    return sub_cols
