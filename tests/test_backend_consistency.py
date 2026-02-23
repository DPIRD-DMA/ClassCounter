"""Tests that all three backends produce identical results for the same input."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from classcounter.core import _count_numba, _count_numpy, _count_torch


def _run_all_backends(data: np.ndarray) -> list[dict]:
    """Run the same data through numpy, torch, and numba backends."""
    tensor = torch.from_numpy(data)
    return [
        _count_numpy(data),
        _count_torch(tensor),
        _count_numba(data),
    ]


class TestBackendConsistency:

    def test_simple_positive_integers(self, simple_array):
        results = _run_all_backends(simple_array)
        assert results[0] == results[1] == results[2]

    def test_2d_array(self, simple_2d_array):
        results = _run_all_backends(simple_2d_array)
        assert results[0] == results[1] == results[2]

    def test_3d_array(self, simple_3d_array):
        results = _run_all_backends(simple_3d_array)
        assert results[0] == results[1] == results[2]

    def test_empty_array(self, empty_array):
        results = _run_all_backends(empty_array)
        assert results[0] == results[1] == results[2] == {}

    def test_single_element(self, single_element_array):
        results = _run_all_backends(single_element_array)
        assert results[0] == results[1] == results[2]

    def test_negative_values(self, negative_array):
        results = _run_all_backends(negative_array)
        assert results[0] == results[1] == results[2]

    def test_float_values(self, float_array):
        results = _run_all_backends(float_array)
        assert results[0] == results[1] == results[2]

    def test_large_values(self):
        arr = np.array([0, 100_000, 100_000, 999_999], dtype=np.int64)
        results = _run_all_backends(arr)
        assert results[0] == results[1] == results[2]

    def test_bool_array(self):
        arr = np.array([True, True, False, True], dtype=bool)
        results = _run_all_backends(arr)
        assert results[0] == results[1] == results[2]

    def test_all_same_class(self):
        arr = np.full(100, 7, dtype=np.int32)
        results = _run_all_backends(arr)
        assert results[0] == results[1] == results[2] == {7: 100}

    @pytest.mark.slow
    def test_large_array(self, large_array):
        results = _run_all_backends(large_array)
        assert results[0] == results[1] == results[2]
