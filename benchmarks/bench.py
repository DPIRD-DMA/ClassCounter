"""Benchmark harness for ClassCounter backends.

Run with: uv run python benchmarks/bench.py
"""

from __future__ import annotations

import time

import numpy as np

from classcounter.core import _count_numba, _count_numpy, _count_torch

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False


# Sizes chosen to cover small / medium / very-large workloads
SIZES = [
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
]
NUM_CLASSES = 20
REPEATS = 80
WARMUP = 5


def warmup_numba_signatures() -> None:
    """Force numba to compile every dtype kernel before timing starts."""
    rng = np.random.default_rng(0)
    for dtype in (np.uint8, np.uint16, np.int32, np.int64):
        arr = rng.integers(0, 10, size=1000, dtype=dtype)
        _count_numba(arr)


def bench(fn, *, sync: bool = False, repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    """Median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def make_inputs(size: int, dtype: np.dtype) -> dict:
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        hi = min(NUM_CLASSES, info.max + 1)
        arr = rng.integers(0, hi, size=size, dtype=dtype)
    else:
        arr = rng.integers(0, NUM_CLASSES, size=size).astype(dtype)
    out = {"numpy": arr}
    if torch is not None:
        out["tensor_cpu"] = torch.from_numpy(arr.copy())
        if HAS_CUDA:
            out["tensor_gpu"] = out["tensor_cpu"].cuda()
            torch.cuda.synchronize()
    return out


def run() -> None:
    print(f"CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Warming up numba kernels...")
    warmup_numba_signatures()
    print()

    for dtype in (np.uint8, np.int32):
        print(f"=== dtype={np.dtype(dtype).name} ===")
        header = f"{'size':>14} | {'numpy':>9} | {'numba':>9} | {'torch_cpu':>10}"
        if HAS_CUDA:
            header += f" | {'torch_gpu':>10}"
        print(header)
        print("-" * len(header))

        for size in SIZES:
            data = make_inputs(size, np.dtype(dtype))

            t_numpy = bench(lambda: _count_numpy(data["numpy"]))
            t_numba = bench(lambda: _count_numba(data["numpy"]))
            line = f"{size:>14,} | {t_numpy:>7.2f}ms | {t_numba:>7.2f}ms"

            if torch is not None:
                t_torch_cpu = bench(lambda: _count_torch(data["tensor_cpu"]))
                line += f" | {t_torch_cpu:>8.2f}ms"
            if HAS_CUDA:
                t_torch_gpu = bench(lambda: _count_torch(data["tensor_gpu"]), sync=True)
                line += f" | {t_torch_gpu:>8.2f}ms"

            print(line)
        print()


if __name__ == "__main__":
    run()
