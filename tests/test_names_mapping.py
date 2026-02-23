"""Tests for the names mapping parameter."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from classcounter import count_classes


class TestNamesMapping:

    def test_names_maps_keys(self, simple_array, names_mapping):
        result = count_classes(simple_array, names=names_mapping)
        assert result["water"] == 3
        assert result["forest"] == 2
        assert result["crop"] == 1

    def test_zero_count_for_absent_classes(self, simple_array, names_mapping):
        """Classes 2 and 4 are NOT in data but ARE in names — they get count 0."""
        result = count_classes(simple_array, names=names_mapping)
        assert result["urban"] == 0
        assert result["bare"] == 0

    def test_all_classes_named_no_warning(self):
        """No warning when every class in data has a name."""
        arr = np.array([0, 1, 1], dtype=np.int32)
        names = {0: "water", 1: "forest"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = count_classes(arr, names=names)
        assert result == {"water": 1, "forest": 2}

    def test_names_none_returns_int_keys(self, simple_array):
        result = count_classes(simple_array, names=None)
        assert all(isinstance(k, int) for k in result)

    def test_empty_names_dict(self, simple_array):
        """Empty names dict — all classes are unmapped, each emits a warning."""
        with pytest.warns(UserWarning):
            result = count_classes(simple_array, names={})
        assert result == {0: 3, 1: 2, 3: 1}

    def test_names_ordering(self, simple_array, names_mapping):
        """Named entries appear first (in names dict order), then unmapped."""
        keys = list(count_classes(simple_array, names=names_mapping).keys())
        assert keys[:5] == ["water", "forest", "urban", "crop", "bare"]

    def test_names_with_all_classes_absent(self):
        """All data classes are unmapped — warnings emitted, int keys used."""
        arr = np.array([10, 20, 20], dtype=np.int32)
        names = {0: "water", 1: "forest"}
        with pytest.warns(UserWarning) as record:
            result = count_classes(arr, names=names)
        assert result["water"] == 0
        assert result["forest"] == 0
        assert result[10] == 1
        assert result[20] == 2
        assert len(record) == 2  # one warning per unmapped class



class TestUnmappedClassWarning:

    def test_warning_emitted_for_unmapped_class(self):
        arr = np.array([0, 0, 99], dtype=np.int32)
        names = {0: "water"}
        with pytest.warns(UserWarning, match="Class 99 found in data but missing from names"):
            result = count_classes(arr, names=names)
        assert result["water"] == 2
        assert result[99] == 1

    def test_multiple_unmapped_classes_emit_multiple_warnings(self):
        arr = np.array([0, 5, 10], dtype=np.int32)
        names = {0: "water"}
        with pytest.warns(UserWarning) as record:
            count_classes(arr, names=names)
        assert len(record) == 2
        messages = {str(w.message) for w in record}
        assert any("Class 5" in m for m in messages)
        assert any("Class 10" in m for m in messages)

    def test_no_warning_when_names_is_none(self, simple_array):
        """No warning at all when names is not provided."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            count_classes(simple_array)

    def test_unmapped_class_value_used_as_key(self):
        arr = np.array([0, 7], dtype=np.int32)
        names = {0: "water"}
        with pytest.warns(UserWarning):
            result = count_classes(arr, names=names)
        assert 7 in result
        assert isinstance(list(result.keys())[-1], int)
