"""Tests for raster file loading via rasterio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import skip_no_rasterio

pytestmark = [pytest.mark.geo, skip_no_rasterio]


@pytest.fixture
def tiny_geotiff(tmp_path):
    """Create a minimal single-band GeoTIFF with known class values."""
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
    return path


@pytest.fixture
def multiband_geotiff(tmp_path):
    """Create a 3-band GeoTIFF."""
    import rasterio
    from rasterio.transform import from_bounds

    rng = np.random.default_rng(0)
    data = rng.integers(0, 5, size=(3, 4, 4), dtype=np.uint8)
    path = tmp_path / "multi.tif"
    transform = from_bounds(0, 0, 4, 4, 4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="uint8",
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        for band in range(3):
            dst.write(data[band], band + 1)
    return path


class TestRasterLoading:

    def test_load_geotiff_string_path(self, tiny_geotiff):
        from classcounter import count_classes

        result = count_classes(str(tiny_geotiff))
        assert result == {0: 2, 1: 2, 2: 2}

    def test_load_geotiff_pathlib(self, tiny_geotiff):
        from classcounter import count_classes

        result = count_classes(Path(tiny_geotiff))
        assert result == {0: 2, 1: 2, 2: 2}

    def test_multiband_raster(self, multiband_geotiff):
        from classcounter import count_classes

        result = count_classes(multiband_geotiff)
        total = sum(result.values())
        assert total == 3 * 4 * 4  # 3 bands * 4 rows * 4 cols

    def test_raster_with_names(self, tiny_geotiff):
        from classcounter import count_classes

        names = {0: "water", 1: "forest", 2: "urban"}
        result = count_classes(str(tiny_geotiff), names=names)
        assert result == {"water": 2, "forest": 2, "urban": 2}
