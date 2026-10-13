"""Tests for the torch counting backend."""

from __future__ import annotations

import pytest

from tests.conftest import skip_no_torch

pytestmark = [pytest.mark.torch, skip_no_torch]


@pytest.fixture
def torch_module():
    import torch

    return torch


class TestTorchBackend:

    def test_tensor_input_auto_detects_torch(self, torch_module, simple_array):
        from classcounter import count_classes

        tensor = torch_module.from_numpy(simple_array)
        result = count_classes(tensor)
        assert result == {0: 3, 1: 2, 3: 1}

    def test_negative_values(self, torch_module):
        from classcounter.core import _count_torch

        tensor = torch_module.tensor([-1, 0, 0, 1], dtype=torch_module.int32)
        result = _count_torch(tensor)
        assert result == {-1: 1, 0: 2, 1: 1}

    def test_float_tensor(self, torch_module):
        from classcounter.core import _count_torch

        tensor = torch_module.tensor([1.0, 1.0, 2.5, 2.5])
        result = _count_torch(tensor)
        assert result == {1.0: 2, 2.5: 2}

    @pytest.mark.parametrize("dtype_name", ["uint8", "uint16", "uint32", "uint64"])
    def test_unsigned_dtypes(self, torch_module, dtype_name):
        from classcounter.core import _count_torch

        dtype = getattr(torch_module, dtype_name, None)
        if dtype is None:
            pytest.skip(f"torch has no {dtype_name}")
        tensor = torch_module.tensor([0, 1, 1, 5], dtype=dtype)
        result = _count_torch(tensor)
        assert result == {0: 1, 1: 2, 5: 1}

    def test_empty_tensor(self, torch_module):
        from classcounter.core import _count_torch

        tensor = torch_module.tensor([], dtype=torch_module.int32)
        result = _count_torch(tensor)
        assert result == {}

    def test_2d_tensor(self, torch_module):
        from classcounter.core import _count_torch

        tensor = torch_module.tensor([[0, 1, 1], [2, 2, 0]], dtype=torch_module.int32)
        result = _count_torch(tensor)
        assert result == {0: 2, 1: 2, 2: 2}

    def test_values_are_python_int(self, torch_module, simple_array):
        from classcounter import count_classes

        tensor = torch_module.from_numpy(simple_array)
        result = count_classes(tensor)
        for v in result.values():
            assert isinstance(v, int)
