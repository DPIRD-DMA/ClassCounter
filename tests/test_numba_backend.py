"""Tests for the numba counting backend."""

from __future__ import annotations

import pytest

from tests.conftest import skip_no_numba

pytestmark = [pytest.mark.numba, skip_no_numba]


class TestNumbaBackend:

    def test_basic_count(self, simple_array):
        from classcounter.core import _count_numba

        result = _count_numba(simple_array)
        assert result == {0: 3, 1: 2, 3: 1}

    def test_negative_falls_back_to_unique(self, negative_array):
        from classcounter.core import _count_numba

        result = _count_numba(negative_array)
        assert result == {-2: 1, -1: 1, 0: 1, 1: 2, 2: 3}

    def test_float_falls_back_to_unique(self, float_array):
        from classcounter.core import _count_numba

        result = _count_numba(float_array)
        assert result == {1.0: 2, 2.5: 3}

    def test_empty_array(self, empty_array):
        from classcounter.core import _count_numba

        assert _count_numba(empty_array) == {}

    def test_single_element(self, single_element_array):
        from classcounter.core import _count_numba

        assert _count_numba(single_element_array) == {42: 1}

    def test_shape_agnostic_3d(self, simple_3d_array):
        from classcounter.core import _count_numba

        result = _count_numba(simple_3d_array)
        assert result == {0: 3, 1: 2, 3: 1}
