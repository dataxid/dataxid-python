# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Decoding pipeline for the DataXID SDK.

Converts integer codes produced during generation back into the original
feature values using the per-column statistics computed during analysis.

Privacy boundary: only opaque integer codes cross the network. Decoding
happens client-side because only the client holds the per-column statistics
and value mappings that map codes back to real values.
"""

from __future__ import annotations

import calendar
import logging

import numpy as np
import pandas as pd

from dataxid.encoder._ports import (
    WIRE_COLUMN_SEP,
    WIRE_PREFIX,
    WIRE_TABLE_SEP,
    EncodingType,
    wire_key,
)
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
from dataxid.training._config import RareStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def _decode_categorical(
    codes: np.ndarray,
    stats: dict,
    rare_strategy: RareStrategy = "sample",
) -> pd.Series:
    """Decode categorical codes back to their string values.

    Args:
        codes: Integer codes produced during generation.
        stats: Column statistics (must contain ``codes`` mapping).
        rare_strategy: How the ``<protected>`` sentinel is rendered.
            - ``"sample"`` (default): replace each ``<protected>`` code with a
              random draw from the remaining frequent categories, so the
              output contains only real values.
            - ``"mask"``: keep the literal ``<protected>`` token in the output,
              preserving the training distribution including rare values.
    """
    categories = list(stats["codes"].keys())
    values = pd.Series(
        pd.Categorical.from_codes(codes, categories=categories),
        dtype="string",
    )
    values[values == NULL_TOKEN] = pd.NA

    if rare_strategy == "sample":
        is_rare = values == RARE_TOKEN
        cnt_rare = is_rare.sum()
        if cnt_rare > 0:
            valid = [k for k in categories if k not in (RARE_TOKEN, NULL_TOKEN)]
            if valid:
                values[is_rare] = np.random.choice(valid, size=cnt_rare)
            else:
                values[is_rare] = pd.NA

    values = unescape_tokens(values)
    return values


# ---------------------------------------------------------------------------
# Numeric — discrete
# ---------------------------------------------------------------------------

def _decode_numeric_discrete(codes: np.ndarray, stats: dict) -> pd.Series:
    dtype = "Float64" if stats["min_decimal"] < 0 else "Int64"
    values = _decode_categorical(codes, stats, rare_strategy="mask")

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
# Suppress protected-category and out-of-range tokens
# ---------------------------------------------------------------------------

def compute_fixed_probs(
    column_stats: dict[str, dict],
    mode: RareStrategy = "mask",
) -> dict[str, dict[int, float]]:
    """Build a ``{wire_key: {code: 0.0}}`` map that zeroes reserved tokens.

    The returned mapping is applied at generation time to prevent the
    ``<protected>`` sentinel (categorical, numeric discrete) and the
    out-of-range / unknown bin tokens (numeric binned) from being sampled
    where they would corrupt the output.

    Args:
        column_stats: Per-feature encoding statistics.
        mode: Controls ``<protected>`` handling for categorical columns.
            - ``"mask"``: suppress ``<protected>`` only in columns that had
              no rare values during training; columns with real rare values
              keep the sentinel sampleable (distribution fidelity).
            - ``"sample"``: always suppress ``<protected>`` in every
              categorical column.
            Numeric discrete/binned columns always suppress their reserved
            tokens regardless of mode — sampling them would produce
            invalid numeric output.
    """
    fixed_probs: dict[str, dict[int, float]] = {}

    for feature, stats in column_stats.items():
        raw_et = stats.get("encoding_type", EncodingType.categorical)
        et = EncodingType(raw_et) if isinstance(raw_et, str) else raw_et
        codes = stats.get("codes", {})
        if not codes:
            continue

        if et == EncodingType.categorical:
            if RARE_TOKEN not in codes:
                continue
            no_rare = stats.get("no_of_rare_categories", 0)
            if mode == "sample" or no_rare == 0:
                fixed_probs[wire_key(feature, "cat")] = {codes[RARE_TOKEN]: 0.0}

        elif et == EncodingType.numeric_binned:
            if BINNED_UNKNOWN in codes:
                fixed_probs[wire_key(feature, "bin")] = {codes[BINNED_UNKNOWN]: 0.0}

        elif et == EncodingType.numeric_discrete:
            if RARE_TOKEN in codes:
                fixed_probs[wire_key(feature, "cat")] = {codes[RARE_TOKEN]: 0.0}

    return fixed_probs


# ---------------------------------------------------------------------------
# Distribution probs — override category distribution
# ---------------------------------------------------------------------------

def compute_distribution_probs(
    column_stats: dict[str, dict],
    distribution: dict,
) -> dict[str, dict[int, float]]:
    """
    Build fixed_probs that shift a categorical column toward target probabilities.

    Args:
        column_stats: Column statistics dictionary.
        distribution: ``{"column": "gender", "probabilities": {"M": 0.7, "F": 0.3}}``

    Returns:
        Fixed-probability overrides for the distribution column.
    """
    column = distribution.get("column")
    probabilities = distribution.get("probabilities", {})
    if not column or not probabilities:
        return {}

    stats = column_stats.get(column)
    if stats is None:
        logger.warning("Distribution column '%s' not found in column_stats", column)
        return {}

    raw_et = stats.get("encoding_type", EncodingType.categorical)
    et = EncodingType(raw_et) if isinstance(raw_et, str) else raw_et
    if et != EncodingType.categorical:
        logger.warning(
            "Distribution column '%s' has encoding_type=%s, expected categorical",
            column, et,
        )
        return {}

    codes = stats.get("codes", {})
    if not codes:
        logger.warning("Distribution column '%s' has no codes mapping", column)
        return {}

    code_probs: dict[int, float] = {}
    for category, prob in probabilities.items():
        if category not in codes:
            logger.warning("Distribution category '%s' not found in codes for '%s'", category, column)
            continue
        code_probs[codes[category]] = max(0.0, prob)

    if not code_probs:
        return {}

    total = sum(code_probs.values())
    if total > 1.0:
        code_probs = {c: p / total for c, p in code_probs.items()}

    return {wire_key(column, "cat"): code_probs}


def distribution_column_order(
    features: list[str],
    distribution: dict,
) -> list[str] | None:
    """
    Build column generation order with the distribution column first.

    Args:
        features: Column names from the training data.
        distribution: ``{"column": "gender", "probabilities": {"M": 0.7, "F": 0.3}}``

    Returns:
        Ordered column list in wire-key format, or None if no reordering needed.
    """
    column = distribution.get("column")
    if not column or column not in features:
        if column and column not in features:
            logger.warning("Distribution column '%s' not in features", column)
        return None

    wire_col = f"{WIRE_PREFIX}{WIRE_TABLE_SEP}{WIRE_COLUMN_SEP}"
    dist_col = f"{wire_col}{column}"
    others = [f"{wire_col}{f}" for f in features if f != column]
    return [dist_col] + others


def conditional_column_order(
    features: list[str],
    conditions_columns: list[str],
    distribution: dict | None = None,
) -> list[str]:
    """Build column generation order: condition (fixed) columns first, then the rest.

    Priority (highest first):
    1. Condition columns — known values, generated first so later columns see them
    2. Distribution column (if not already in conditions)
    3. Remaining columns

    Args:
        features: All training feature names.
        conditions_columns: Column names present in the conditions DataFrame.
        distribution: Optional ``{"column": "...", "probabilities": {...}}``.

    Returns:
        Ordered column list in wire-key format.
    """
    wire_col = f"{WIRE_PREFIX}{WIRE_TABLE_SEP}{WIRE_COLUMN_SEP}"
    conditions_set = set(conditions_columns)

    conditions_wire = [f"{wire_col}{f}" for f in features if f in conditions_set]
    rest = [f for f in features if f not in conditions_set]

    dist_wire: list[str] = []
    if distribution:
        dc = distribution.get("column")
        if dc and dc in rest:
            dist_wire = [f"{wire_col}{dc}"]
            rest = [f for f in rest if f != dc]

    rest_wire = [f"{wire_col}{f}" for f in rest]
    return conditions_wire + dist_wire + rest_wire


def merge_fixed_probs(
    *probs_dicts: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    """Deep-merge multiple fixed_probs dicts (later dicts override earlier ones)."""
    merged: dict[str, dict[int, float]] = {}
    for d in probs_dicts:
        for sub_col, code_map in d.items():
            if sub_col in merged:
                merged[sub_col] = {**merged[sub_col], **code_map}
            else:
                merged[sub_col] = dict(code_map)
    return merged


# ---------------------------------------------------------------------------
# Imputation probs — suppress NULL tokens for imputed columns
# ---------------------------------------------------------------------------

def compute_imputation_probs(
    column_stats: dict[str, dict],
    columns: list[str],
) -> dict[str, dict[int, float]]:
    """Build fixed_probs that suppress NULL/NaN tokens for imputed columns.

    Two strategies depending on encoding type:

    1. Separate binary ``nan`` sub-column (numeric_digit, datetime, character,
       lat_long): code 1 → 0.0.
    2. NULL token in the main vocabulary (categorical, numeric_discrete,
       numeric_binned): NULL token code → 0.0. For categorical the RARE
       token is also suppressed.

    Args:
        column_stats: Per-feature encoding statistics.
        columns: Feature names to impute.

    Returns:
        ``{wire_key: {code: 0.0, ...}}``
    """
    fixed_probs: dict[str, dict[int, float]] = {}

    for col in columns:
        stats = column_stats.get(col)
        if stats is None:
            continue

        raw_et = stats.get("encoding_type", EncodingType.categorical)
        et = EncodingType(raw_et) if isinstance(raw_et, str) else raw_et

        # Strategy 1: separate binary nan sub-column
        if stats.get("has_nan"):
            fixed_probs[wire_key(col, "nan")] = {1: 0.0}
            continue

        # Strategy 2: NULL token in codes vocabulary
        codes = stats.get("codes", {})
        if not codes:
            continue

        if et == EncodingType.categorical:
            code_probs: dict[int, float] = {}
            if NULL_TOKEN in codes:
                code_probs[codes[NULL_TOKEN]] = 0.0
            if RARE_TOKEN in codes:
                code_probs[codes[RARE_TOKEN]] = 0.0
            if code_probs:
                fixed_probs[wire_key(col, "cat")] = code_probs

        elif et == EncodingType.numeric_discrete:
            if NULL_TOKEN in codes:
                fixed_probs[wire_key(col, "cat")] = {codes[NULL_TOKEN]: 0.0}

        elif et == EncodingType.numeric_binned:
            if BINNED_NULL in codes:
                fixed_probs[wire_key(col, "bin")] = {codes[BINNED_NULL]: 0.0}

    return fixed_probs


def imputation_column_order(
    features: list[str],
    imputed_columns: list[str],
    conditions_columns: list[str] | None = None,
    distribution: dict | None = None,
) -> list[str]:
    """Build column generation order with imputed columns last.

    Priority (highest first):
    1. Condition columns (known values from non-NULL cells)
    2. Distribution column
    3. Non-imputed remaining columns
    4. Imputed columns (predicted from known context)

    Args:
        features: All training feature names.
        imputed_columns: Feature names being imputed (moved to end).
        conditions_columns: Feature names with at least one known value in conditions.
        distribution: Optional ``{"column": "...", "probabilities": {...}}``.

    Returns:
        Ordered column list in wire-key format.
    """
    wire = f"{WIRE_PREFIX}{WIRE_TABLE_SEP}{WIRE_COLUMN_SEP}"
    imputed_set = set(imputed_columns)
    conditions_set = set(conditions_columns or [])

    conditions_wire = [f"{wire}{f}" for f in features if f in conditions_set and f not in imputed_set]
    imputed_wire = [f"{wire}{f}" for f in features if f in imputed_set]
    rest = [f for f in features if f not in conditions_set and f not in imputed_set]

    dist_wire: list[str] = []
    if distribution:
        dc = distribution.get("column")
        if dc and dc in rest:
            dist_wire = [f"{wire}{dc}"]
            rest = [f for f in rest if f != dc]

    rest_wire = [f"{wire}{f}" for f in rest]
    return conditions_wire + dist_wire + rest_wire + imputed_wire


def bias_column_order(
    features: list[str],
    bias: dict,
    conditions_columns: list[str] | None = None,
    distribution: dict | None = None,
    imputed_columns: list[str] | None = None,
) -> list[str]:
    """Build column generation order for bias-corrected synthesis.

    Earlier columns are generated first and condition the later ones. The
    target column is placed last so its distribution can be steered by the
    already-generated sensitive attributes.

    Priority (first to last):
        1. Condition columns (known values)
        2. Distribution column
        3. Sensitive columns (non-imputed)
        4. Sensitive columns (imputed)
        5. Remaining columns
        6. Imputed columns (non-sensitive)
        7. Bias target column

    Args:
        features: All training feature names.
        bias: ``{"target": "income", "sensitive": ["gender", ...]}``.
        conditions_columns: Feature names present in the conditions frame.
        distribution: Optional ``{"column": "...", "probabilities": {...}}``.
        imputed_columns: Feature names being imputed.

    Returns:
        Ordered column list in wire-key format.
    """
    wire = f"{WIRE_PREFIX}{WIRE_TABLE_SEP}{WIRE_COLUMN_SEP}"
    target = bias["target"]
    sensitive_set = set(bias["sensitive"])
    conditions_set = set(conditions_columns or [])
    imputed_set = set(imputed_columns or [])

    sensitive_non_imp = [f for f in features if f in sensitive_set and f not in imputed_set]
    sensitive_imp = [f for f in features if f in sensitive_set and f in imputed_set]
    imputed_non_sens = [
        f for f in features
        if f in imputed_set and f not in sensitive_set and f != target
    ]
    rest = [f for f in features
            if f not in sensitive_set and f not in imputed_set and f != target]

    conditions_wire = [f"{wire}{f}" for f in features if f in conditions_set
                       and f not in sensitive_set and f != target and f not in imputed_set]

    dist_wire: list[str] = []
    if distribution:
        dc = distribution.get("column")
        if dc and dc in rest:
            dist_wire = [f"{wire}{dc}"]
            rest = [f for f in rest if f != dc]

    return (
        conditions_wire + dist_wire
        + [f"{wire}{f}" for f in sensitive_non_imp]
        + [f"{wire}{f}" for f in sensitive_imp]
        + [f"{wire}{f}" for f in rest]
        + [f"{wire}{f}" for f in imputed_non_sens]
        + [f"{wire}{target}"]
    )


def resolve_bias_payload(
    column_stats: dict[str, dict],
    bias: dict,
) -> dict:
    """Validate a bias spec and resolve it into integer-coded form.

    Translates the target/sensitive column names into their sub-column
    identifiers and looks up the integer codes that make up each sensitive
    group, ready for consumption by downstream generation code.

    Args:
        column_stats: Column statistics produced by the analyze step.
        bias: ``{"target": "income", "sensitive": ["gender"]}``.

    Returns:
        Dict with keys ``target_sub_col``, ``sensitive_sub_cols``, and
        ``sensitive_groups`` (lists of integer codes per sensitive column).

    Raises:
        ValueError: If columns are missing, non-categorical, or if the
            target column also appears in the sensitive list.
    """
    target = bias.get("target")
    sensitive = bias.get("sensitive")
    if not target:
        raise ValueError("bias.target is required")
    if not sensitive:
        raise ValueError("bias.sensitive is required")
    if target in sensitive:
        raise ValueError("Bias.target cannot appear in sensitive")

    all_cols = [target] + list(sensitive)
    for col in all_cols:
        if col not in column_stats:
            raise ValueError(f"bias column '{col}' not found in training data")
        et = column_stats[col].get("encoding_type")
        if et != EncodingType.categorical:
            raise ValueError(
                f"bias column '{col}' has encoding_type={et}, "
                f"expected {EncodingType.categorical}"
            )

    target_sub_col = wire_key(target, "cat")
    sensitive_sub_cols = [wire_key(col, "cat") for col in sensitive]

    sensitive_groups: list[list[int]] = []
    for col in sensitive:
        codes = column_stats[col].get("codes", {})
        no_rare = column_stats[col].get("no_of_rare_categories", 0)
        code_values = list(codes.values())
        if no_rare == 0 and RARE_TOKEN in codes:
            code_values.remove(codes[RARE_TOKEN])
        sensitive_groups.append(code_values)

    return {
        "target_sub_col": target_sub_col,
        "sensitive_sub_cols": sensitive_sub_cols,
        "sensitive_groups": sensitive_groups,
    }


# ===================================================================
# Dispatcher
# ===================================================================

def decode_columns(
    raw_codes: dict[str, list[int] | np.ndarray],
    features: list[str],
    column_stats: dict[str, dict],
    rare_strategy: RareStrategy = "sample",
) -> pd.DataFrame:
    """Decode raw integer codes back to original feature values.

    Inverse of ``encode_columns``. Uses ``column_stats`` to map each integer
    code back to its original value.

    Args:
        raw_codes: ``{wire_key: int_codes}`` produced during generation.
        features: Feature names in original order.
        column_stats: Per-feature encoding statistics (from analyze).
        rare_strategy: How the ``<protected>`` sentinel is rendered in
            categorical columns.
            - ``"sample"`` (default): replace rare codes with a random draw
              from the frequent categories, so the output contains only
              real values.
            - ``"mask"``: keep the literal ``<protected>`` token in the
              output, preserving the training distribution including the
              presence of rare values.
            Numeric discrete/binned and all non-categorical columns always
            strip their reserved tokens regardless of this flag.

    Returns:
        DataFrame with decoded feature values.
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
            decoded[feature] = _decode_categorical(
                arr, stats, rare_strategy=rare_strategy,
            )

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
