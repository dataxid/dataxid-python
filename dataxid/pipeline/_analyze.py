# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Data analysis — produces column statistics, cardinalities, and value mappings
from raw tabular data.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from dataxid.encoder._ports import EncodingType, wire_key
from dataxid.pipeline._privacy import (
    log_histogram,
    noise_threshold,
    private_bounds,
    private_filter,
    quantile_bins,
)
from dataxid.pipeline._transform import (
    BINNED_MAX,
    BINNED_MIN,
    BINNED_NULL,
    BINNED_UNKNOWN,
    CHARACTER_PAD,
    DATETIME_PARTS,
    NULL_TOKEN,
    PRECISION_HI,
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
# Constants — analysis thresholds
# ---------------------------------------------------------------------------

_BOUNDS_SAMPLE_N = 1000
_BOUNDS_MIN_N = 20

_MAX_BINS = 100
_MAX_QUANTILES = 1001

_MAX_DISCRETE = 100
_DISCRETE_COVERAGE = 0.999
_MAX_POSITIONS = 3


# ===================================================================
# Per-type analyze (partition-level statistics)
# ===================================================================

def _analyze_categorical(values: pd.Series, root_keys: pd.Series) -> dict:
    values = escape_tokens(to_string(values))
    df = pd.concat([root_keys, values], axis=1)
    cnt_values = df.groupby(values.name)[root_keys.name].nunique().to_dict()
    return {"has_nan": sum(values.isna()) > 0, "cnt_values": cnt_values}


def _analyze_numeric(
    values: pd.Series, root_keys: pd.Series, encoding_type: EncodingType,
) -> dict:
    values = to_numeric(values)
    non_na = values.dropna()
    cnt_unique = non_na.nunique()
    log_hist = log_histogram(non_na.to_numpy())

    quantiles = None
    is_binned = encoding_type in (EncodingType.numeric_binned, EncodingType.numeric_auto)
    if is_binned and len(non_na) > 0:
        quantiles = np.quantile(
            non_na, np.linspace(0, 1, _MAX_QUANTILES), method="closest_observation",
        ).tolist()

    cnt_values = None
    if encoding_type == EncodingType.numeric_discrete or (
        encoding_type == EncodingType.numeric_auto and cnt_unique < _MAX_DISCRETE
    ):
        df = pd.concat([root_keys, values], axis=1)
        cnt_values = df.groupby(values.name)[root_keys.name].nunique().to_dict()

    df = pd.concat([root_keys, values], axis=1)
    min_vals = df.groupby(root_keys.name)[values.name].min().dropna()
    min_n = min_vals.sort_values(ascending=True).head(_BOUNDS_SAMPLE_N).astype("float").tolist()
    max_vals = df.groupby(root_keys.name)[values.name].max().dropna()
    max_n = max_vals.sort_values(ascending=False).head(_BOUNDS_SAMPLE_N).astype("float").tolist()

    df_split = digitize(values)
    is_not_nan = df_split["nan"] == 0
    has_nan = sum(df_split["nan"]) > 0
    has_neg = sum(df_split["neg"]) > 0

    if any(is_not_nan):
        min_digits = {k: int(df_split[k][is_not_nan].min()) for k in df_split if k.startswith("E")}
        max_digits = {k: int(df_split[k][is_not_nan].max()) for k in df_split if k.startswith("E")}
    else:
        min_digits = {k: 0 for k in df_split if k.startswith("E")}
        max_digits = {k: 0 for k in df_split if k.startswith("E")}

    return {
        "has_nan": has_nan, "has_neg": has_neg,
        "min_digits": min_digits, "max_digits": max_digits,
        "min_n": min_n, "max_n": max_n,
        "cnt_values": cnt_values, "quantiles": quantiles,
        "log_hist": log_hist,
    }


def _analyze_datetime(values: pd.Series, root_keys: pd.Series) -> dict:
    values = to_datetime(values)
    log_hist = log_histogram(values.dropna().astype("int64").to_numpy())
    df = pd.concat([root_keys, values], axis=1)
    min_dates = df.groupby(root_keys.name)[values.name].min().dropna()
    min_n = min_dates.sort_values(ascending=True).head(_BOUNDS_SAMPLE_N).astype(str).tolist()
    max_dates = df.groupby(root_keys.name)[values.name].max().dropna()
    max_n = max_dates.sort_values(ascending=False).head(_BOUNDS_SAMPLE_N).astype(str).tolist()
    df_split = split_datetime(values)
    is_not_nan = df_split["nan"] == 0
    has_nan = any(df_split["nan"] == 1)
    if any(is_not_nan):
        min_values = {k: int(df_split[k][is_not_nan].min()) for k in DATETIME_PARTS}
        max_values = {k: int(df_split[k][is_not_nan].max()) for k in DATETIME_PARTS}
    else:
        defaults = {"year": 2022, "month": 1, "day": 1}
        min_values = {k: 0 for k in DATETIME_PARTS} | defaults
        max_values = {k: 0 for k in DATETIME_PARTS} | defaults
    return {
        "has_nan": has_nan, "min_values": min_values, "max_values": max_values,
        "min_n": min_n, "max_n": max_n, "log_hist": log_hist,
    }


def _analyze_character(values: pd.Series, root_keys: pd.Series) -> dict:
    values = to_string(values)
    df_split = split_chars(values)
    has_nan = sum(df_split["nan"]) > 0
    df = pd.concat([root_keys, df_split], axis=1)
    characters = {
        sc: df.groupby(sc)[root_keys.name].nunique().to_dict()
        for sc in df_split.columns if sc.startswith("P")
    }
    return {"max_string_length": len(characters), "has_nan": has_nan, "characters": characters}


# ===================================================================
# Per-type reduce (merge partition stats → final column_stats)
# ===================================================================

def _reduce_categorical(
    stats_list: list[dict],
    protect_rare: bool = True,
    epsilon: float | None = None,
) -> dict:
    cnt_values: dict[str, int] = {}
    for item in stats_list:
        for value, count in item["cnt_values"].items():
            cnt_values[value] = cnt_values.get(value, 0) + count
    cnt_values = dict(sorted(cnt_values.items()))
    known = list(cnt_values.keys())

    if protect_rare:
        if epsilon is not None:
            categories, _ = private_filter(cnt_values, epsilon, threshold=5)
        else:
            threshold = noise_threshold(min_threshold=5)
            categories = [k for k in known if cnt_values[k] >= threshold]
    else:
        categories = known

    if any(s["has_nan"] for s in stats_list):
        categories = [NULL_TOKEN] + categories
    categories = [RARE_TOKEN] + categories

    return {
        "no_of_rare_categories": len(known) - len(
            [c for c in categories if c not in (RARE_TOKEN, NULL_TOKEN)]
        ),
        "codes": {categories[i]: i for i in range(len(categories))},
        "cardinalities": {"cat": len(categories)},
    }


def _reduce_numeric(
    stats_list: list[dict],
    protect_rare: bool = True,
    epsilon: float | None = None,
    encoding_type: EncodingType | None = EncodingType.numeric_auto,
) -> dict:
    has_nan = any(s["has_nan"] for s in stats_list)
    has_neg = any(s["has_neg"] for s in stats_list)
    keys = stats_list[0]["max_digits"].keys()
    min_digits = {k: min(s["min_digits"][k] for s in stats_list) for k in keys}
    max_digits = {k: max(s["max_digits"][k] for s in stats_list) for k in keys}
    non_zero_prec = [k for k in keys if max_digits[k] > 0 and k.startswith("E")]
    min_decimal = min(int(k[1:]) for k in non_zero_prec) if non_zero_prec else 0

    reduced_min_n = sorted([v for s in stats_list for v in s["min_n"]])
    reduced_max_n = sorted([v for s in stats_list for v in s["max_n"]], reverse=True)

    if protect_rare:
        if len(reduced_min_n) < _BOUNDS_MIN_N or len(reduced_max_n) < _BOUNDS_MIN_N:
            reduced_min = reduced_max = None
        else:
            if epsilon is not None:
                log_hist = [
                    sum(b) for b in zip(*(s["log_hist"] for s in stats_list), strict=False)
                ]
                reduced_min, reduced_max = private_bounds(log_hist, epsilon)
            else:
                reduced_min = reduced_min_n[noise_threshold(min_threshold=5)]
                reduced_max = reduced_max_n[noise_threshold(min_threshold=5)]
    else:
        reduced_min = reduced_min_n[0] if reduced_min_n else None
        reduced_max = reduced_max_n[0] if reduced_max_n else None

    if reduced_min is not None or reduced_max is not None:
        max_abs = np.max(np.abs(np.array([reduced_min, reduced_max])))
        max_decimal = int(np.floor(np.log10(max_abs))) if max_abs >= 10 else 0
    else:
        max_decimal = 0
    decimal_cap_str = [d[1:] for d in keys]
    first_cap = next(iter(decimal_cap_str), None)
    decimal_cap = (
        int(first_cap) if first_cap and first_cap.isnumeric() else PRECISION_HI
    )
    max_decimal = min(max(min_decimal, max_decimal), decimal_cap)

    has_cnt = all(s["cnt_values"] for s in stats_list)
    if has_cnt:
        cnt_values: dict[str, int] = {}
        for item in stats_list:
            for value, count in item["cnt_values"].items():
                cnt_values[value] = cnt_values.get(value, 0) + count
        cnt_total = sum(cnt_values.values())
        if protect_rare:
            if epsilon is not None:
                categories, non_rare_ratio = private_filter(cnt_values, epsilon, threshold=5)
            else:
                threshold = noise_threshold(min_threshold=5)
                cnt_values = {c: v for c, v in cnt_values.items() if v >= threshold}
                categories = list(cnt_values.keys())
                non_rare_ratio = sum(cnt_values.values()) / cnt_total
        else:
            categories = list(cnt_values.keys())
            non_rare_ratio = 1.0
    else:
        categories = []
        non_rare_ratio = 0.0

    if encoding_type == EncodingType.numeric_auto:
        if non_rare_ratio > _DISCRETE_COVERAGE:
            encoding_type = EncodingType.numeric_discrete
        elif len(non_zero_prec) <= _MAX_POSITIONS:
            encoding_type = EncodingType.numeric_digit
        else:
            encoding_type = EncodingType.numeric_binned

    if encoding_type == EncodingType.numeric_discrete:
        if min_decimal >= 0:
            categories = [str(cat).split(".")[0] for cat in categories]
        if has_nan:
            categories = [NULL_TOKEN] + categories
        categories = [RARE_TOKEN] + categories
        return {
            "encoding_type": EncodingType.numeric_discrete.value,
            "cardinalities": {"cat": len(categories)},
            "codes": {categories[i]: i for i in range(len(categories))},
            "min_decimal": min_decimal,
        }

    elif encoding_type == EncodingType.numeric_digit:
        cardinalities: dict[str, int] = {}
        if has_nan:
            cardinalities["nan"] = 2
        if has_neg:
            cardinalities["neg"] = 2
        for d in np.arange(max_decimal, min_decimal - 1, -1):
            cardinalities[f"E{d}"] = max_digits[f"E{d}"] + 1 - min_digits[f"E{d}"]
        return {
            "encoding_type": EncodingType.numeric_digit.value,
            "cardinalities": cardinalities,
            "has_nan": has_nan, "has_neg": has_neg,
            "min_digits": min_digits, "max_digits": max_digits,
            "max_decimal": max_decimal, "min_decimal": min_decimal,
            "min": reduced_min, "max": reduced_max,
        }

    elif encoding_type == EncodingType.numeric_binned:
        if reduced_min is None or reduced_max is None:
            bins = [0]
            min_decimal = 0
        else:
            if epsilon is None:
                quantiles = np.concatenate([s["quantiles"] for s in stats_list if s["quantiles"]])
                quantiles = list(np.clip(quantiles, reduced_min, reduced_max))
                bins = quantile_bins(quantiles, _MAX_BINS)
            else:
                bins = list(np.linspace(reduced_min, reduced_max, _MAX_BINS + 1))
        cats = [BINNED_UNKNOWN]
        if has_nan:
            cats += [BINNED_NULL]
        if reduced_min is not None:
            cats += [BINNED_MIN]
        if reduced_max is not None:
            cats += [BINNED_MAX]
        return {
            "encoding_type": EncodingType.numeric_binned.value,
            "cardinalities": {"bin": len(cats) + len(bins) - 1},
            "codes": {cats[i]: i for i in range(len(cats))},
            "min_decimal": min_decimal, "bins": bins,
        }
    else:
        raise ValueError(f"Unknown numeric encoding type: {encoding_type}")


def _reduce_datetime(
    stats_list: list[dict],
    protect_rare: bool = True,
    epsilon: float | None = None,
) -> dict:
    has_nan = any(s["has_nan"] for s in stats_list)
    keys = stats_list[0]["min_values"].keys()
    min_values = {k: min(s["min_values"][k] for s in stats_list) for k in keys}
    max_values = {k: max(s["max_values"][k] for s in stats_list) for k in keys}
    has_time = max_values["hour"] > 0 or max_values["minute"] > 0 or max_values["second"] > 0
    has_ms = has_time and (
        max_values["ms_E2"] > 0 or max_values["ms_E1"] > 0 or max_values["ms_E0"] > 0
    )

    reduced_min_n = sorted([v for s in stats_list for v in s["min_n"]])
    reduced_max_n = sorted([v for s in stats_list for v in s["max_n"]], reverse=True)

    if protect_rare:
        if len(reduced_min_n) < _BOUNDS_MIN_N or len(reduced_max_n) < _BOUNDS_MIN_N:
            reduced_min = reduced_max = None
            has_time = has_ms = False
        else:
            if epsilon is not None:
                has_long = any(len(v) > 10 for v in reduced_min_n + reduced_max_n)
                dt_fmt = "%Y-%m-%d %H:%M:%S" if has_long else "%Y-%m-%d"
                log_hist = [
                    sum(b) for b in zip(*(s["log_hist"] for s in stats_list), strict=False)
                ]
                reduced_min, reduced_max = private_bounds(log_hist, epsilon)
                if reduced_min is not None and reduced_max is not None:
                    reduced_min = pd.to_datetime(int(reduced_min), unit="us").strftime(dt_fmt)
                    reduced_max = pd.to_datetime(int(reduced_max), unit="us").strftime(dt_fmt)
            else:
                reduced_min = str(reduced_min_n[noise_threshold(min_threshold=5)])
                reduced_max = str(reduced_max_n[noise_threshold(min_threshold=5)])
            if reduced_min is not None and reduced_max is not None:
                max_values["year"] = int(reduced_max[0:4])
                min_values["year"] = int(reduced_min[0:4])
    else:
        reduced_min = str(reduced_min_n[0]) if reduced_min_n else None
        reduced_max = str(reduced_max_n[0]) if reduced_max_n else None

    cardinalities: dict[str, int] = {}
    if has_nan:
        cardinalities["nan"] = 2
    cardinalities["year"] = max_values["year"] + 1 - min_values["year"]
    cardinalities["month"] = max_values["month"] + 1 - min_values["month"]
    cardinalities["day"] = max_values["day"] + 1 - min_values["day"]
    if has_time:
        cardinalities["hour"] = max_values["hour"] + 1 - min_values["hour"]
        cardinalities["minute"] = max_values["minute"] + 1 - min_values["minute"]
        cardinalities["second"] = max_values["second"] + 1 - min_values["second"]
    if has_ms:
        cardinalities["ms_E2"] = max_values["ms_E2"] + 1 - min_values["ms_E2"]
        cardinalities["ms_E1"] = max_values["ms_E1"] + 1 - min_values["ms_E1"]
        cardinalities["ms_E0"] = max_values["ms_E0"] + 1 - min_values["ms_E0"]
    return {
        "cardinalities": cardinalities,
        "has_nan": has_nan, "has_time": has_time, "has_ms": has_ms,
        "min_values": min_values, "max_values": max_values,
        "min": reduced_min, "max": reduced_max,
    }


def _reduce_character(
    stats_list: list[dict],
    protect_rare: bool = True,
    epsilon: float | None = None,
) -> dict:
    max_len = max(s["max_string_length"] for s in stats_list)
    positions = [f"P{i}" for i in range(max_len)]
    codes: dict[str, dict[str, int]] = {pos: {} for pos in positions}
    for pos in positions:
        cnt: dict[str, int] = {}
        for item in stats_list:
            for value, count in item["characters"].get(pos, {}).items():
                cnt[value] = cnt.get(value, 0) + count
        cnt = dict(sorted(cnt.items()))
        known = list(cnt.keys())
        if protect_rare:
            if epsilon is not None:
                cats, _ = private_filter(cnt, epsilon, threshold=5)
            else:
                threshold = noise_threshold(min_threshold=5)
                cats = [k for k in known if cnt[k] >= threshold]
        else:
            cats = known
        cats = [CHARACTER_PAD] + [c for c in cats if c != CHARACTER_PAD]
        codes[pos] = {cats[i]: i for i in range(len(cats))}
    cardinalities: dict[str, int] = {}
    has_nan = any(s["has_nan"] for s in stats_list)
    if has_nan:
        cardinalities["nan"] = 2
    for sc, sc_codes in codes.items():
        cardinalities[sc] = len(sc_codes)
    return {
        "has_nan": has_nan, "max_string_length": max_len,
        "codes": codes, "cardinalities": cardinalities,
    }


# ===================================================================
# Semantic type inference
# ===================================================================

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^uuid|^guid|^sku|^code$|_code$|^ref$|_ref$|^key$|_key$)",
    re.IGNORECASE,
)

_LATLONG_PATTERNS = re.compile(
    r"(lat|lon|latitude|longitude|geo_?lat|geo_?lon|coord)",
    re.IGNORECASE,
)

_DATETIME_NAME_PATTERNS = re.compile(
    r"(date|time|timestamp|datetime|created|updated|_at$|_dt$|_ts$)",
    re.IGNORECASE,
)

_DATETIME_PARSE_THRESHOLD = 0.8


def _looks_like_datetime(series: pd.Series, column_name: str) -> bool:
    """Heuristic: does a string column likely contain datetime values?"""
    if _DATETIME_NAME_PATTERNS.search(column_name):
        sample = series.dropna().head(200)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() >= _DATETIME_PARSE_THRESHOLD
    return False


def detect_encoding(series: pd.Series, column_name: str = "") -> EncodingType:
    """Infer the best encoding type from data characteristics and column semantics."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return EncodingType.datetime

    name = column_name.lower()

    if _LATLONG_PATTERNS.search(name) and pd.api.types.is_numeric_dtype(series):
        return EncodingType.lat_long

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if pd.api.types.is_integer_dtype(series) and n_unique <= 100:
            return EncodingType.numeric_discrete
        elif n_unique <= 20:
            return EncodingType.categorical
        else:
            return EncodingType.numeric_binned

    if series.dtype == object:
        if _looks_like_datetime(series, column_name):
            return EncodingType.datetime
        sample = series.dropna().head(100)
        if len(sample) > 0:
            avg_len = sample.astype(str).str.len().mean()
            n_unique = series.nunique()
            if _ID_PATTERNS.search(name):
                return EncodingType.character
            if n_unique > 100 and avg_len < 20:
                return EncodingType.character

    return EncodingType.categorical


# ===================================================================
# Public API
# ===================================================================

def compute_stats(
    df: pd.DataFrame,
    features: list[str],
    protect_rare: bool = True,
    encoding_types: dict[str, str | EncodingType] | None = None,
) -> dict[str, dict]:
    """Analyze raw features and produce column_stats for encoding."""
    encoding_types = encoding_types or {}
    column_stats: dict[str, dict] = {}
    root_keys = pd.Series(range(len(df)), name="__root_key__")

    for feature in features:
        if feature not in df.columns:
            logger.warning("Feature %s not found, skipping", feature)
            continue

        series = df[feature]
        if feature in encoding_types:
            et = encoding_types[feature]
            enc_type = EncodingType(et) if isinstance(et, str) else et
        else:
            enc_type = detect_encoding(series, feature)

        logger.info("  Analyzing %s: %s", feature, enc_type.value)

        if enc_type == EncodingType.categorical:
            partition = _analyze_categorical(series, root_keys)
            stats = _reduce_categorical([partition], protect_rare=protect_rare)

        elif enc_type in (
            EncodingType.numeric_auto, EncodingType.numeric_binned,
            EncodingType.numeric_discrete, EncodingType.numeric_digit,
        ):
            partition = _analyze_numeric(series, root_keys, enc_type)
            stats = _reduce_numeric(
                [partition], protect_rare=protect_rare, encoding_type=enc_type,
            )

        elif enc_type == EncodingType.datetime:
            partition = _analyze_datetime(series, root_keys)
            stats = _reduce_datetime([partition], protect_rare=protect_rare)

        elif enc_type == EncodingType.character:
            partition = _analyze_character(series, root_keys)
            stats = _reduce_character([partition], protect_rare=protect_rare)

        else:
            logger.warning(
                "Unsupported encoding type %s for %s, falling back to categorical",
                enc_type, feature,
            )
            partition = _analyze_categorical(series, root_keys)
            stats = _reduce_categorical([partition], protect_rare=protect_rare)
            enc_type = EncodingType.categorical

        if "encoding_type" not in stats:
            stats["encoding_type"] = enc_type
        column_stats[feature] = stats

    return column_stats


def unpack_stats(
    column_stats: dict[str, dict],
) -> tuple[dict[str, int], dict[str, str], dict[str, dict]]:
    """Extract cardinalities, encoding_map, and value_mappings from column_stats."""
    cardinalities: dict[str, int] = {}
    encoding_map: dict[str, str] = {}
    value_mappings: dict[str, dict] = {}

    for feature, stats in column_stats.items():
        et = stats.get("encoding_type", EncodingType.categorical)
        encoding_map[feature] = et.value if hasattr(et, "value") else et
        for sub_col, card in stats.get("cardinalities", {}).items():
            cardinalities[wire_key(feature, sub_col)] = card
        codes = stats.get("codes", {})
        if codes:
            value_mappings[feature] = codes

    return cardinalities, encoding_map, value_mappings
