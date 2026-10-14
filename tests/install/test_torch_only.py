# /// script
# requires-python = ">=3.10"
# dependencies = ["classcounter[torch]", "pytest"]
#
# [tool.uv.sources]
# classcounter = { path = "../.." }
# # CPU-only torch: these tests only check the install works, and the default
# # Linux wheel pulls ~3 GB of CUDA libraries (on every CI run).
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Test that classcounter[torch] works (torch available, rasterio/numba not required)."""

import sys

import numpy as np
import pytest
import torch

from classcounter import count_classes


def test_torch_importable():
    """torch must be available with the torch extra."""
    assert hasattr(torch, "tensor")


def test_count_torch_tensor():
    """count_classes should handle a torch tensor."""
    t = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
    result = count_classes(t)
    assert result == {0: 2, 1: 3, 2: 1}


def test_count_torch_2d():
    t = torch.tensor([[0, 1, 1], [2, 2, 0]], dtype=torch.int32)
    result = count_classes(t)
    assert result == {0: 2, 1: 2, 2: 2}


def test_count_torch_bool():
    t = torch.tensor([True, False, True, True])
    result = count_classes(t)
    assert result == {0: 1, 1: 3}


def test_count_torch_with_names():
    t = torch.tensor([0, 0, 1], dtype=torch.int32)
    result = count_classes(t, names={0: "a", 1: "b"})
    assert result == {"a": 2, "b": 1}


def test_count_torch_percent():
    t = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
    result = count_classes(t, percent=True)
    assert abs(result[1] - 50.0) < 0.01


def test_numpy_still_works():
    """Base numpy counting must still work with torch extra installed."""
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr)
    assert result == {0: 2, 1: 3, 2: 1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
