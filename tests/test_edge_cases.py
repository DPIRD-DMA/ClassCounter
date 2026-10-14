"""Tests for edge cases and unusual inputs."""

from __future__ import annotations

import numpy as np
import pytest

from classcounter import count_classes


class TestEmptyArrays:
    def test_empty_1d(self, empty_array):
        assert count_classes(empty_array) == {}

    def test_empty_2d(self):
        arr = np.array([], dtype=np.int32).reshape(0, 5)
        assert count_classes(arr) == {}

    def test_empty_3d(self):
        arr = np.array([], dtype=np.int32).reshape(1, 0, 5)
        assert count_classes(arr) == {}


class TestSingleElement:
    def test_single_zero(self):
        assert count_classes(np.array([0], dtype=np.int32)) == {0: 1}

    def test_single_large_value(self):
        assert count_classes(np.array([9999], dtype=np.int32)) == {9999: 1}


class TestShapeHandling:
    @pytest.mark.parametrize(
        "shape",
        [(12,), (3, 4), (2, 2, 3), (1, 12), (12, 1), (1, 1, 12)],
        ids=["1d", "2d", "3d", "row", "col", "band-row-col"],
    )
    def test_shape_agnostic_same_result(self, shape):
        base = np.arange(12, dtype=np.int32) % 4
        arr = base.reshape(shape)
        assert count_classes(arr) == {0: 3, 1: 3, 2: 3, 3: 3}


class TestNegativeValues:
    def test_all_negative(self):
        arr = np.array([-3, -3, -1, -1, -1], dtype=np.int32)
        assert count_classes(arr) == {-3: 2, -1: 3}

    def test_mixed_sign(self, negative_array):
        assert count_classes(negative_array) == {-2: 1, -1: 1, 0: 1, 1: 2, 2: 3}


class TestFloatValues:
    def test_float32(self):
        arr = np.array([0.0, 0.0, 1.5, 1.5, 1.5], dtype=np.float32)
        assert count_classes(arr) == {0.0: 2, 1.5: 3}

    def test_float64(self, float_array):
        assert count_classes(float_array) == {1.0: 2, 2.5: 3}


class TestLargeValues:
    def test_high_class_values(self):
        arr = np.array([0, 1_000_000], dtype=np.int64)
        assert count_classes(arr) == {0: 1, 1_000_000: 1}

    def test_bool_array(self):
        arr = np.array([True, True, False, True], dtype=bool)
        result = count_classes(arr)
        assert sum(result.values()) == 4
