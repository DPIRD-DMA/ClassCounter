"""Tests for the numpy counting backend."""

from __future__ import annotations

import numpy as np
import pytest

from classcounter.core import _count_numpy


class TestNumpyBackend:

    def test_small_array_bincount_path(self, simple_array):
        result = _count_numpy(simple_array)
        assert result == {0: 3, 1: 2, 3: 1}

    def test_negative_values_unique_path(self, negative_array):
        result = _count_numpy(negative_array)
        assert result == {-2: 1, -1: 1, 0: 1, 1: 2, 2: 3}

    def test_float_values_unique_path(self, float_array):
        result = _count_numpy(float_array)
        assert result == {1.0: 2, 2.5: 3}

    def test_empty_array(self, empty_array):
        assert _count_numpy(empty_array) == {}

    def test_single_element(self, single_element_array):
        assert _count_numpy(single_element_array) == {42: 1}

    @pytest.mark.parametrize("shape", [(6,), (2, 3), (1, 2, 3)])
    def test_shape_agnostic(self, shape):
        arr = np.array([0, 0, 0, 1, 1, 3], dtype=np.int32).reshape(shape)
        assert _count_numpy(arr) == {0: 3, 1: 2, 3: 1}

    def test_uint8_dtype(self):
        arr = np.array([0, 1, 1, 255], dtype=np.uint8)
        assert _count_numpy(arr) == {0: 1, 1: 2, 255: 1}

    def test_int64_dtype(self):
        arr = np.array([0, 100, 100, 200], dtype=np.int64)
        assert _count_numpy(arr) == {0: 1, 100: 2, 200: 1}

    def test_all_same_class(self):
        arr = np.full(100, 5, dtype=np.int32)
        assert _count_numpy(arr) == {5: 100}

    def test_consecutive_classes(self):
        arr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        assert _count_numpy(arr) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
