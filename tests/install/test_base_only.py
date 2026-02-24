# /// script
# requires-python = ">=3.10"
# dependencies = ["classcounter", "pytest"]
#
# [tool.uv.sources]
# classcounter = { path = "../.." }
# ///
"""Test that classcounter works with only the base (numpy) install."""

import sys

import numpy as np
import pytest

from classcounter import count_classes


def test_numpy_counting():
    """Core counting must work with just numpy."""
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr)
    assert result == {0: 2, 1: 3, 2: 1}, f"Expected {{0: 2, 1: 3, 2: 1}}, got {result}"


def test_percent_mode():
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr, percent=True)
    assert abs(result[1] - 50.0) < 0.01, f"Expected ~50.0 for class 1, got {result[1]}"


def test_names_mapping():
    arr = np.array([0, 0, 1], dtype=np.int32)
    result = count_classes(arr, names={0: "a", 1: "b"})
    assert result == {"a": 2, "b": 1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
