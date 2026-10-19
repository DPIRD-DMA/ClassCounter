from __future__ import annotations

import numpy as np
import pytest

# tests/install/ holds standalone scripts that test_install.py runs in isolated
# environments with `uv run --script`. Collecting them here would run them in
# this environment, where their imports (torch, rasterio, ...) may be missing.
collect_ignore = ["install"]

# ---------------------------------------------------------------------------
# Skip helpers for optional dependencies
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import rasterio  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
skip_no_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
skip_no_rasterio = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")


# ---------------------------------------------------------------------------
# Reusable array fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_array() -> np.ndarray:
    """Small 1D integer array with known distribution. Class 2 is absent."""
    return np.array([0, 0, 0, 1, 1, 3], dtype=np.int32)


@pytest.fixture
def simple_2d_array() -> np.ndarray:
    return np.array([0, 0, 0, 1, 1, 3], dtype=np.int32).reshape(2, 3)


@pytest.fixture
def simple_3d_array() -> np.ndarray:
    return np.array([0, 0, 0, 1, 1, 3], dtype=np.int32).reshape(1, 2, 3)


@pytest.fixture
def empty_array() -> np.ndarray:
    return np.array([], dtype=np.int32)


@pytest.fixture
def single_element_array() -> np.ndarray:
    return np.array([42], dtype=np.int32)


@pytest.fixture
def large_array() -> np.ndarray:
    """Large array for backend consistency checks."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 10, size=1_000_000, dtype=np.int32)


@pytest.fixture
def negative_array() -> np.ndarray:
    """Array with negative values — forces np.unique fallback."""
    return np.array([-2, -1, 0, 1, 1, 2, 2, 2], dtype=np.int32)


@pytest.fixture
def float_array() -> np.ndarray:
    """Float array — forces np.unique fallback."""
    return np.array([1.0, 1.0, 2.5, 2.5, 2.5], dtype=np.float64)


@pytest.fixture
def names_mapping() -> dict[int, str]:
    """Standard names dict covering classes 0-4."""
    return {0: "water", 1: "forest", 2: "urban", 3: "crop", 4: "bare"}
