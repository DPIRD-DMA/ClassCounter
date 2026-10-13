from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(parallel=True)
def bincount_numba(flat: NDArray[np.integer], num_bins: int) -> NDArray[np.int64]:
    """Count occurrences of each value in a flat non-negative integer array."""
    n_threads = numba.get_num_threads()
    chunk_size = (flat.size + n_threads - 1) // n_threads
    local_counts = np.zeros((n_threads, num_bins), dtype=np.int64)

    for t in numba.prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, flat.size)
        for i in range(start, end):
            local_counts[t, flat[i]] += 1

    counts = np.zeros(num_bins, dtype=np.int64)
    for t in range(n_threads):
        for j in range(num_bins):
            counts[j] += local_counts[t, j]
    return counts


@numba.njit(parallel=True)
def bincount_signed_numba(
    flat: NDArray[np.integer], max_bins: int
) -> tuple[NDArray[np.int64], int, int]:
    """Single-pass histogram + min/max for signed integer arrays.

    Returns (counts, min_val, max_val). Caller must check:
      * if min_val < 0: counts are invalid (fall back to np.unique)
      * if max_val >= max_bins: counts is missing entries (rerun bigger)

    For typical class-counting workloads (small positive max), this replaces
    the separate min/max scan + histogram with a single memory pass.
    """
    n_threads = numba.get_num_threads()
    chunk_size = (flat.size + n_threads - 1) // n_threads
    local_counts = np.zeros((n_threads, max_bins), dtype=np.int64)
    local_min = np.empty(n_threads, dtype=flat.dtype)
    local_max = np.empty(n_threads, dtype=flat.dtype)

    for t in numba.prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, flat.size)
        if start >= end:
            local_min[t] = flat[0]
            local_max[t] = flat[0]
            continue
        mn = flat[start]
        mx = flat[start]
        for i in range(start, end):
            v = flat[i]
            if v < mn:
                mn = v
            elif v > mx:
                mx = v
            if 0 <= v < max_bins:
                local_counts[t, v] += 1
        local_min[t] = mn
        local_max[t] = mx

    counts = np.zeros(max_bins, dtype=np.int64)
    for t in range(n_threads):
        for j in range(max_bins):
            counts[j] += local_counts[t, j]

    mn = local_min[0]
    mx = local_max[0]
    for t in range(1, n_threads):
        if local_min[t] < mn:
            mn = local_min[t]
        if local_max[t] > mx:
            mx = local_max[t]
    return counts, mn, mx


@numba.njit(parallel=True)
def minmax_numba(flat: NDArray[np.integer]) -> tuple[int, int]:
    """Single parallel pass to compute (min, max) of an integer array.

    Replaces calling `flat.min()` and `flat.max()` separately, which costs
    two full memory-bandwidth-bound scans over the data. Precondition:
    flat.size > 0.
    """
    n_threads = numba.get_num_threads()
    chunk_size = (flat.size + n_threads - 1) // n_threads
    local_min = np.empty(n_threads, dtype=flat.dtype)
    local_max = np.empty(n_threads, dtype=flat.dtype)

    for t in numba.prange(n_threads):
        start = t * chunk_size
        end = min(start + chunk_size, flat.size)
        if start >= end:
            local_min[t] = flat[0]
            local_max[t] = flat[0]
            continue
        mn = flat[start]
        mx = flat[start]
        for i in range(start + 1, end):
            v = flat[i]
            if v < mn:
                mn = v
            elif v > mx:
                mx = v
        local_min[t] = mn
        local_max[t] = mx

    mn = local_min[0]
    mx = local_max[0]
    for t in range(1, n_threads):
        if local_min[t] < mn:
            mn = local_min[t]
        if local_max[t] > mx:
            mx = local_max[t]
    return mn, mx
