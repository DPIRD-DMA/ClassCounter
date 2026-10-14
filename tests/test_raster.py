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


class TestSaveCountsToRaster:
    def test_saves_tags(self, tiny_geotiff):
        import rasterio

        from classcounter import save_counts_to_raster

        counts = {0: 2, 1: 2, 2: 2}
        save_counts_to_raster(tiny_geotiff, counts)

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert tags["CLASS_COUNT_0"] == "2"
        assert tags["CLASS_COUNT_1"] == "2"
        assert tags["CLASS_COUNT_2"] == "2"

    def test_saves_named_keys(self, tiny_geotiff):
        import rasterio

        from classcounter import save_counts_to_raster

        counts = {"water": 2, "forest": 2, "urban": 2}
        save_counts_to_raster(tiny_geotiff, counts)

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert tags["CLASS_COUNT_water"] == "2"
        assert tags["CLASS_COUNT_forest"] == "2"
        assert tags["CLASS_COUNT_urban"] == "2"

    def test_clears_stale_tags(self, tiny_geotiff):
        """Old CLASS_COUNT_ tags are removed when writing new counts."""
        import rasterio

        from classcounter import save_counts_to_raster

        save_counts_to_raster(tiny_geotiff, {0: 10, 1: 20, 99: 5})
        save_counts_to_raster(tiny_geotiff, {0: 2, 1: 2})

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert tags["CLASS_COUNT_0"] == "2"
        assert tags["CLASS_COUNT_1"] == "2"
        assert "CLASS_COUNT_99" not in tags

    def test_preserves_raster_data(self, tiny_geotiff):
        import rasterio

        from classcounter import save_counts_to_raster

        with rasterio.open(tiny_geotiff) as src:
            original_data = src.read()

        save_counts_to_raster(tiny_geotiff, {0: 2, 1: 2, 2: 2})

        with rasterio.open(tiny_geotiff) as src:
            after_data = src.read()
        np.testing.assert_array_equal(original_data, after_data)

    def test_file_not_found(self, tmp_path):
        from classcounter import save_counts_to_raster

        with pytest.raises(FileNotFoundError):
            save_counts_to_raster(tmp_path / "nonexistent.tif", {0: 1})

    def test_count_classes_save_metadata(self, tiny_geotiff):
        import rasterio

        from classcounter import count_classes

        result = count_classes(str(tiny_geotiff), save_metadata=True)
        assert result == {0: 2, 1: 2, 2: 2}

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert tags["CLASS_COUNT_0"] == "2"
        assert tags["CLASS_COUNT_1"] == "2"
        assert tags["CLASS_COUNT_2"] == "2"

    def test_save_metadata_with_non_path_raises(self):
        from classcounter import count_classes

        with pytest.raises(ValueError, match="save_metadata=True requires data to be a file path"):
            count_classes(np.array([0, 1, 2]), save_metadata=True)

    def test_custom_prefix(self, tiny_geotiff):
        import rasterio

        from classcounter import save_counts_to_raster

        save_counts_to_raster(tiny_geotiff, {0: 2, 1: 2, 2: 2}, prefix="LAND_")

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert tags["LAND_0"] == "2"
        assert tags["LAND_1"] == "2"
        assert tags["LAND_2"] == "2"

    def test_custom_prefix_clears_only_matching(self, tiny_geotiff):
        """Custom prefix only clears tags with that same prefix."""
        import rasterio

        from classcounter import save_counts_to_raster

        save_counts_to_raster(tiny_geotiff, {0: 10}, prefix="LAND_")
        save_counts_to_raster(tiny_geotiff, {0: 2}, prefix="OTHER_")

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert tags["LAND_0"] == "10"
        assert tags["OTHER_0"] == "2"

    def test_percent_auto_prefix(self, tiny_geotiff):
        import rasterio

        from classcounter import count_classes

        count_classes(str(tiny_geotiff), percent=True, save_metadata=True)

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert "CLASS_PERCENT_0" in tags
        assert "CLASS_PERCENT_1" in tags
        assert "CLASS_PERCENT_2" in tags
        assert "CLASS_COUNT_0" not in tags

    def test_metadata_prefix_overrides_auto(self, tiny_geotiff):
        import rasterio

        from classcounter import count_classes

        count_classes(str(tiny_geotiff), percent=True, save_metadata=True, metadata_prefix="MY_")

        with rasterio.open(tiny_geotiff) as src:
            tags = src.tags()
        assert "MY_0" in tags
        assert "CLASS_PERCENT_0" not in tags
