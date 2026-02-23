"""Integration tests for the public count_classes API."""

from __future__ import annotations

from classcounter import count_classes


class TestCountClassesBasic:

    def test_basic_count(self, simple_array):
        result = count_classes(simple_array)
        assert result == {0: 3, 1: 2, 3: 1}

    def test_returns_dict(self, simple_array):
        assert isinstance(count_classes(simple_array), dict)

    def test_values_are_python_int(self, simple_array):
        result = count_classes(simple_array)
        for k, v in result.items():
            assert isinstance(k, int)
            assert isinstance(v, int)

    def test_auto_backend_numpy_array(self, simple_array):
        result = count_classes(simple_array)
        assert result == {0: 3, 1: 2, 3: 1}
