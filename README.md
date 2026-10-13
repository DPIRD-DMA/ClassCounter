[![PyPI Version](https://img.shields.io/pypi/v/classcounter)](https://pypi.org/project/classcounter/)
[![Python Versions](https://img.shields.io/pypi/pyversions/classcounter)](https://pypi.org/project/classcounter/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# ClassCounter

Fast class counting for NumPy arrays, PyTorch tensors, and GeoTIFFs.

## Installation

```bash
pip install classcounter
```

Optional backends:

```bash
pip install classcounter[torch]   # PyTorch support
pip install classcounter[geo]     # GeoTIFF support via rasterio
pip install classcounter[numba]   # Numba-accelerated backend
pip install classcounter[all]     # Everything
```

Requires Python 3.10+.

## Usage

```python
import numpy as np
from classcounter import count_classes

arr = np.random.default_rng(0).integers(0, 5, size=(100, 100), dtype=np.int32)

count_classes(arr)
# {0: 2017, 1: 1960, 2: 2050, 3: 1932, 4: 2041}

# Map class IDs to names
count_classes(arr, names={0: "water", 1: "forest", 2: "urban", 3: "crop", 4: "bare"})
# {'water': 2017, 'forest': 1960, 'urban': 2050, 'crop': 1932, 'bare': 2041}

# Get percentages instead of counts
count_classes(arr, percent=True)
# {0: 20.17, 1: 19.6, 2: 20.5, 3: 19.32, 4: 20.41}
```

Input arrays can be any shape — they are flattened internally. Negative integers and floats are supported via a `np.unique` fallback path.

### Counting only some elements

Pass a boolean `mask` to count only the elements where it is `True`, e.g. only valid (non-nodata) pixels:

```python
count_classes(pred, mask=valid)
count_classes(pred, mask=valid, percent=True)  # percentages of the valid pixels
```

The mask must have the same shape as the data or be broadcastable to it. A `(rows, cols)` mask covers every band of a `(bands, rows, cols)` raster. This gives the same result as `count_classes(pred[valid])` without copying the selected elements. On GPU tensors it also avoids the host sync that boolean indexing forces.

### GeoTIFF files

```python
count_classes("land_cover.tif")
```

Requires the `geo` extra.

### Saving results to GeoTIFF metadata

Write class counts back into the raster's GDAL metadata tags:

```python
# One-liner: count and save in one step
count_classes("land_cover.tif", save_metadata=True)
# Writes tags: CLASS_COUNT_0=2017, CLASS_COUNT_1=1960, ...

# With percentages — automatically uses CLASS_PERCENT_ prefix
count_classes("land_cover.tif", percent=True, save_metadata=True)
# Writes tags: CLASS_PERCENT_0=20.17, CLASS_PERCENT_1=19.6, ...

# Custom prefix
count_classes("land_cover.tif", save_metadata=True, metadata_prefix="LAND_")
# Writes tags: LAND_0=2017, LAND_1=1960, ...
```

Or use the standalone function for more control:

```python
from classcounter import save_counts_to_raster

counts = count_classes("land_cover.tif", names={0: "water", 1: "forest"})
save_counts_to_raster("land_cover.tif", counts)
# Writes tags: CLASS_COUNT_water=2017, CLASS_COUNT_forest=1960
```

Stale tags from previous runs are automatically cleared before writing.

### PyTorch tensors

```python
import torch

tensor = torch.randint(0, 5, (100, 100))
count_classes(tensor)

tensor = tensor.to("cuda")  # GPU — counting happens on-device
count_classes(tensor)
```

### Backend selection

By default (`backend="auto"`) the backend is chosen based on the input type:

- **NumPy arrays** → Numba (if installed), otherwise NumPy
- **PyTorch tensors** → PyTorch (runs on-device, including CUDA)
- **File paths** → loaded via rasterio, then counted with Numba/NumPy

Pass `backend="numpy"`, `"numba"` or `"torch"` to force one, so behaviour doesn't depend on what else is installed. An error is raised if that backend isn't installed or doesn't accept the input (`"torch"` takes tensors; `"numpy"` and `"numba"` take arrays and file paths).

Numba compiles its kernels on first use and caches them on disk (in `__pycache__` next to the package, or in `NUMBA_CACHE_DIR` if set), so only the first process pays the compile cost. If the package directory isn't writable, set `NUMBA_CACHE_DIR` to a writable location.

## API

### `count_classes(data, names=None, percent=False, save_metadata=False, metadata_prefix=None, mask=None, backend="auto")`

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `ndarray`, `Tensor`, `str`, or `Path` | Input array, tensor, or path to a raster file |
| `names` | `dict[int, str]` or `None` | Optional mapping of class IDs to human-readable names |
| `percent` | `bool` | Return percentages (0–100) instead of raw counts. Default `False` |
| `save_metadata` | `bool` | Write results as GDAL tags in the source GeoTIFF. Only valid when `data` is a file path. Default `False` |
| `metadata_prefix` | `str` or `None` | Custom tag prefix. Defaults to `CLASS_COUNT_` or `CLASS_PERCENT_` (when `percent=True`) |
| `mask` | bool `ndarray`, bool `Tensor`, or `None` | Count only elements where the mask is `True`. Same shape as the data, or broadcastable to it. Default `None` (count everything) |
| `backend` | `str` | `"auto"`, `"numpy"`, `"numba"` or `"torch"`. Default `"auto"` |

**Returns:** `dict[int | str, int]` mapping class values (or names) to counts, or `dict[int | str, float]` when `percent=True`.

When `names` is provided, classes present in the data but missing from the mapping use their integer key (with a warning). Classes in the mapping but absent from the data receive a count of 0.

### `save_counts_to_raster(path, counts, *, prefix=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` or `Path` | Path to an existing GeoTIFF file |
| `counts` | `dict` | Dict of class counts as returned by `count_classes` |
| `prefix` | `str` or `None` | Tag name prefix. Defaults to `CLASS_COUNT_` |

Writes each entry as a GDAL metadata tag (e.g. `CLASS_COUNT_0=1234`). Existing tags matching the prefix are cleared before writing.

## Performance

Benchmarks on a Ryzen 9 5950X with RTX 4090, 100M-element uint8 arrays:

| Backend                     | Time (ms) |
|-----------------------------|-----------|
| NumPy (CPU)                 | 178       |
| Numba (CPU, multithreaded)  | 17        |
| PyTorch CPU                 | 38        |
| PyTorch GPU (CUDA)          | 2         |

The Numba backend runs on the CPU only. For GPU counting, pass a CUDA tensor to use the PyTorch backend.

![Backend comparison](https://raw.githubusercontent.com/DPIRD-DMA/ClassCounter/main/images/Backend%20comparison.png)

Run the included [benchmark notebook](https://github.com/DPIRD-DMA/ClassCounter/blob/main/benchmark_backends.ipynb) to compare backends on your hardware.

See [Examples.ipynb](https://github.com/DPIRD-DMA/ClassCounter/blob/main/Examples.ipynb) for a walkthrough of all features including name mapping, percentages, PyTorch tensors, and GPU acceleration.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests: `uv run pytest`
4. Submit a pull request

## License

MIT
