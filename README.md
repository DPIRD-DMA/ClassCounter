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

### GeoTIFF files

```python
count_classes("land_cover.tif")
```

Requires the `geo` extra.

### PyTorch tensors

```python
import torch

tensor = torch.randint(0, 5, (100, 100))
count_classes(tensor)
```

CUDA tensors are supported — counting happens on-device.

### Backend selection

The backend is chosen automatically based on the input type:

- **NumPy arrays** → Numba (if installed), otherwise NumPy
- **PyTorch tensors** → PyTorch (runs on-device, including CUDA)
- **File paths** → loaded via rasterio, then counted with Numba/NumPy

To use the PyTorch backend, pass a torch tensor:

```python
import torch

tensor = torch.from_numpy(arr)       # CPU
count_classes(tensor)

tensor = tensor.to("cuda")           # GPU — counting happens on-device
count_classes(tensor)
```

## API

### `count_classes(data, names=None, percent=False)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `ndarray`, `Tensor`, `str`, or `Path` | Input array, tensor, or path to a raster file |
| `names` | `dict[int, str]` or `None` | Optional mapping of class IDs to human-readable names |
| `percent` | `bool` | Return percentages (0–100) instead of raw counts. Default `False` |

**Returns:** `dict[int | str, int]` mapping class values (or names) to counts, or `dict[int | str, float]` when `percent=True`.

When `names` is provided, classes present in the data but missing from the mapping use their integer key (with a warning). Classes in the mapping but absent from the data receive a count of 0.

## Performance

Benchmarks on an Ryzen 9 5950X and RTX 4090 with 100M-element arrays:

| Backend       | Time (ms) |
|---------------|-----------|
| NumPy         | 176     |
| Numba         | 12      |
| PyTorch CPU   | 37      |
| PyTorch GPU   | 2       |

![Backend comparison](https://raw.githubusercontent.com/DPIRD-DMA/ClassCounter/main/images/Backend%20comparison.png)

Run the included [benchmark notebook](https://github.com/DPIRD-DMA/ClassCounter/blob/main/Benchmark%20backends.ipynb) to compare backends on your hardware.

See [Examples.ipynb](https://github.com/DPIRD-DMA/ClassCounter/blob/main/Examples.ipynb) for a walkthrough of all features including name mapping, percentages, PyTorch tensors, and GPU acceleration.

## License

MIT
