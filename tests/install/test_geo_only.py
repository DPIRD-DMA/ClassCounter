# /// script
# requires-python = ">=3.10"
# dependencies = ["classcounter[geo]", "pytest"]
#
# [tool.uv.sources]
# classcounter = { path = "../.." }
# ///
"""Test that classcounter[geo] works (rasterio available, torch/numba not required)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from classcounter import count_classes, save_counts_to_raster


@pytest.fixture
def tiny_geotiff(tmp_path):
    """Create a minimal single-band GeoTIFF with known class values."""
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
    return path


def test_rasterio_importable():
    """rasterio must be available with the geo extra."""
    import rasterio  # noqa: F811

    assert hasattr(rasterio, "open")


def test_count_classes_from_file(tiny_geotiff):
    """count_classes should accept a file path and return correct counts."""
    result = count_classes(str(tiny_geotiff))
    assert result == {0: 2, 1: 2, 2: 2}


def test_count_classes_from_pathlib(tiny_geotiff):
    result = count_classes(Path(tiny_geotiff))
    assert result == {0: 2, 1: 2, 2: 2}


def test_count_classes_with_names(tiny_geotiff):
    names = {0: "water", 1: "forest", 2: "urban"}
    result = count_classes(str(tiny_geotiff), names=names)
    assert result == {"water": 2, "forest": 2, "urban": 2}


def test_save_counts_to_raster(tiny_geotiff):
    counts = {0: 2, 1: 2, 2: 2}
    save_counts_to_raster(tiny_geotiff, counts)

    with rasterio.open(tiny_geotiff) as src:
        tags = src.tags()
    assert tags["CLASS_COUNT_0"] == "2"
    assert tags["CLASS_COUNT_1"] == "2"
    assert tags["CLASS_COUNT_2"] == "2"


def test_save_metadata_flag(tiny_geotiff):
    result = count_classes(str(tiny_geotiff), save_metadata=True)
    assert result == {0: 2, 1: 2, 2: 2}

    with rasterio.open(tiny_geotiff) as src:
        tags = src.tags()
    assert tags["CLASS_COUNT_0"] == "2"


def test_numpy_still_works():
    """Base numpy counting must still work with geo extra installed."""
    arr = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    result = count_classes(arr)
    assert result == {0: 2, 1: 3, 2: 1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
