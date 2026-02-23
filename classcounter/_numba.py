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
