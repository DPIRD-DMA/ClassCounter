"""Tests for the percent parameter."""

from __future__ import annotations

import numpy as np
import pytest

from classcounter import count_classes


class TestPercent:
    def test_percent_basic(self):
        arr = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        result = count_classes(arr, percent=True)
        assert result == {0: 40.0, 1: 60.0}

    def test_percent_sums_to_100(self, simple_array):
        result = count_classes(simple_array, percent=True)
        assert sum(result.values()) == pytest.approx(100.0)

    def test_percent_false_returns_counts(self, simple_array):
        result = count_classes(simple_array, percent=False)
        assert result == {0: 3, 1: 2, 3: 1}

    def test_percent_with_names(self, simple_array, names_mapping):
        result = count_classes(simple_array, names=names_mapping, percent=True)
        assert result["water"] == pytest.approx(50.0)
        assert result["urban"] == 0.0
        assert result["bare"] == 0.0

    def test_percent_empty_array(self, empty_array):
        result = count_classes(empty_array, percent=True)
        assert result == {}

    def test_percent_single_element(self):
        arr = np.array([5], dtype=np.int32)
        result = count_classes(arr, percent=True)
        assert result == {5: 100.0}

    def test_percent_rounding(self):
        """3 classes with 1/3 each should round to 33.33."""
        arr = np.array([0, 1, 2], dtype=np.int32)
        result = count_classes(arr, percent=True)
        for v in result.values():
            assert v == pytest.approx(33.33, abs=0.01)
