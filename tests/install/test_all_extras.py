# /// script
# requires-python = ">=3.10"
# dependencies = ["classcounter[all]", "pytest"]
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
"""Test that classcounter[all] works with every backend available."""

import sys

import numpy as np
import pytest

from classcounter import count_classes


def test_all_optional_deps_importable():
    """All optional dependencies must be available with the all extra."""
    import numba
    import rasterio
    import torch

    assert hasattr(torch, "tensor")
    assert hasattr(numba, "njit")
    assert hasattr(rasterio, "open")


def test_numpy_backend():
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr)
    assert result == {0: 2, 1: 3, 2: 1}


def test_torch_backend():
    import torch

    t = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
    result = count_classes(t)
    assert result == {0: 2, 1: 3, 2: 1}


def test_geo_backend(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    data = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    path = tmp_path / "test.tif"
    transform = from_bounds(0, 0, 3, 2, 3, 2)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(data, 1)

    result = count_classes(str(path))
    assert result == {0: 2, 1: 2, 2: 2}


def test_numba_bincount():
    from classcounter._numba import bincount_numba

    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = bincount_numba(arr, 3)
    expected = np.array([2, 3, 1])
    np.testing.assert_array_equal(result, expected)


def test_percent_mode():
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr, percent=True)
    assert abs(result[1] - 50.0) < 0.01


def test_names_mapping():
    arr = np.array([0, 0, 1], dtype=np.int32)
    result = count_classes(arr, names={0: "a", 1: "b"})
    assert result == {"a": 2, "b": 1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
