# Copyright (c) 2026 DataXID Teknoloji ve Ticaret A.Ş.
# SPDX-License-Identifier: Apache-2.0
"""
Statistical privacy helpers for the analysis pipeline.

Provides noise injection, bounds estimation, and filtering used during
the reduce step to protect individual values from leaking through
column statistics.
"""

from __future__ import annotations

import numpy as np


def noise_threshold(min_threshold: int = 5, noise_scale: float = 3) -> int:
    return min_threshold + int(noise_scale * np.random.uniform())


def log_histogram_bounds(idx: int, bins: int = 64) -> tuple[float, float]:
    if idx == bins:
        return (0.0, 1.0)
    elif idx > bins:
        return (2.0 ** (idx - bins - 1), 2.0 ** (idx - bins))
    elif idx == bins - 1:
        return (-1.0, -0.0)
    else:
        return (-1 * 2.0 ** np.abs(bins - idx - 1), -1 * 2.0 ** np.abs(bins - idx - 2))


def log_histogram(values: np.ndarray, bins: int = 64) -> list[int]:
    values = np.array(values, dtype=np.float64)
    values = values[~np.isinf(values) & ~np.isnan(values)]
    edges = [log_histogram_bounds(i, bins) for i in range(bins * 2)]
    bin_edges = np.array([lo for lo, _ in edges] + [edges[-1][1]])
    values = np.clip(values, bin_edges[0], bin_edges[-1])
    hist, _ = np.histogram(values, bins=bin_edges)
    return hist.tolist()


def private_bounds(hist: list[int], epsilon: float) -> tuple[float | None, float | None]:
    n_bins = len(hist)
    noise = np.random.laplace(loc=0.0, scale=1 / epsilon, size=n_bins)
    noisy = [v + n for v, n in zip(hist, noise, strict=False)]
    failure_prob = 10e-9
    max_prob = 1 / (n_bins * 2)
    exceeds: list[int] = []
    while len(exceeds) < 1 and failure_prob <= max_prob:
        p = 1 - failure_prob
        K = -np.log(2 - 2 * p ** (1 / (n_bins - 1))) / epsilon
        exceeds = [i for i, v in enumerate(noisy) if v > K]
        failure_prob *= 10
    if not exceeds:
        return (None, None)
    lo, _ = log_histogram_bounds(min(exceeds))
    _, hi = log_histogram_bounds(max(exceeds))
    return (float(lo), float(hi))


def private_filter(
    value_counts: dict[str, int], epsilon: float, threshold: int = 5,
) -> tuple[list[str], float]:
    noise = np.random.laplace(loc=0.0, scale=1 / epsilon, size=len(value_counts))
    noisy = np.clip(np.array(list(value_counts.values())) + noise, 0, None).astype(int)
    counts = dict(value_counts)
    for i, cat in enumerate(counts):
        counts[cat] = noisy[i]
    total = sum(counts.values())
    selected = {c: n for c, n in counts.items() if n >= threshold}
    ratio = sum(selected.values()) / total if total > 0 else 0
    return list(selected.keys()), ratio


def quantile_bins(x: list, n: int, n_max: int = 1_000) -> list:
    if len(x) <= n or len(set(x)) <= n:
        return list(sorted(set(x)))
    n_quantiles = n
    qs = None
    while n_quantiles <= n_max:
        qs = np.quantile(x, np.linspace(0, 1, n_quantiles + 1), method="closest_observation")
        n_distinct = len(set(qs))
        if n_distinct >= n + 1:
            bins = list(sorted(set(qs)))
            if len(bins) > n + 1:
                bins = bins[: (n // 2) + 1] + bins[-(n // 2):]
            return bins
        n_quantiles += 1 + max(0, n - n_distinct)
    return list(sorted(set(qs))) if qs is not None else list(sorted(set(x)))
