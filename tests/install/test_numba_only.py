# /// script
# requires-python = ">=3.10"
# dependencies = ["classcounter[numba]", "pytest"]
#
# [tool.uv.sources]
# classcounter = { path = "../.." }
# ///
"""Test that classcounter[numba] works (numba available, torch/rasterio not required)."""

import sys

import numpy as np
import pytest

from classcounter import count_classes
from classcounter._numba import bincount_numba


def test_numba_importable():
    """numba must be available with the numba extra."""
    import numba

    assert hasattr(numba, "njit")


def test_numba_bincount_direct():
    """The numba bincount function should work directly."""
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = bincount_numba(arr, 3)
    expected = np.array([2, 3, 1])
    np.testing.assert_array_equal(result, expected)


def test_count_classes_uses_numba():
    """count_classes should use numba backend when available."""
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr)
    assert result == {0: 2, 1: 3, 2: 1}


def test_count_classes_2d():
    arr = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    result = count_classes(arr)
    assert result == {0: 2, 1: 2, 2: 2}


def test_count_classes_with_names():
    arr = np.array([0, 0, 1], dtype=np.int32)
    result = count_classes(arr, names={0: "a", 1: "b"})
    assert result == {"a": 2, "b": 1}


def test_count_classes_percent():
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr, percent=True)
    assert abs(result[1] - 50.0) < 0.01


def test_negative_values_fallback():
    """Negative values should still work (falls back to np.unique)."""
    arr = np.array([-1, -1, 0, 1, 1], dtype=np.int32)
    result = count_classes(arr)
    assert result == {-1: 2, 0: 1, 1: 2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
