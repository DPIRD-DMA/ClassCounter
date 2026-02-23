from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

_torch_module: Any = None
_torch_checked = False
_has_numba_cached: bool | None = None


def _is_nonneg_int(flat: np.ndarray) -> bool:
    """Check if a flat array is non-negative integer (skips min() for unsigned dtypes)."""
    if not np.issubdtype(flat.dtype, np.integer):
        return False
    if np.issubdtype(flat.dtype, np.unsignedinteger):
        return True
    return flat.min() >= 0


def _count_bool(flat: np.ndarray) -> dict[int, int]:
    """Fast path for boolean arrays using count_nonzero."""
    n_true = int(np.count_nonzero(flat))
    n_false = flat.size - n_true
    result: dict[int, int] = {}
    if n_false > 0:
        result[0] = n_false
    if n_true > 0:
        result[1] = n_true
    return result


def _has_numba() -> bool:
    """Check if numba is available. Result is cached."""
    global _has_numba_cached  # noqa: PLW0603
    if _has_numba_cached is None:
        try:
            from classcounter._numba import bincount_numba  # noqa: F401

            _has_numba_cached = True
        except ImportError:
            _has_numba_cached = False
    return _has_numba_cached


def count_classes(
    data: str | Path | np.ndarray | Any,
    names: dict[int, str] | None = None,
    percent: bool = False,
) -> dict[int | str, int] | dict[int | str, float]:
    """Count occurrences of each unique class value.

    Args:
        data: A numpy array, torch tensor, or path to a raster file (GeoTIFF).
        names: Optional mapping of integer class values to string names.
            When provided, returned dict keys are the mapped names.
        percent: If True, return percentages (0-100) instead of raw counts.

    Returns:
        Dict mapping class values (or names) to their counts or percentages.
    """
    arr, use_torch = _resolve_input(data)

    if use_torch:
        counts = _count_torch(arr)
    elif _has_numba():
        counts = _count_numba(arr)
    else:
        counts = _count_numpy(arr)

    result: dict[int | str, int] | dict[int | str, float]
    if names is not None:
        result = {name: counts.get(k, 0) for k, name in names.items()}
        for k, v in counts.items():
            if k not in names:
                warnings.warn(
                    f"Class {k} found in data but missing from names mapping. "
                    f"Using integer value as key.",
                    UserWarning,
                    stacklevel=2,
                )
                result[k] = v
    else:
        result = counts  # type: ignore[assignment]

    if percent:
        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total * 100, 2) for k, v in result.items()}

    return result


def _resolve_input(
    data: str | Path | np.ndarray | Any,
) -> tuple[np.ndarray | Any, bool]:
    """Load data if needed and determine whether to use torch.

    Returns:
        Tuple of (array_or_tensor, use_torch).
    """
    torch_module = _try_import_torch()

    if isinstance(data, str | Path):
        return _load_raster(Path(data)), False
    if isinstance(data, np.ndarray):
        return data, False
    if torch_module is not None and isinstance(data, torch_module.Tensor):
        return data, True

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def _count_numpy(arr: np.ndarray) -> dict[int, int]:
    """Count unique values using numpy.bincount."""
    flat = arr.ravel()
    if flat.dtype == np.bool_:
        return _count_bool(flat)
    if flat.size > 0 and _is_nonneg_int(flat):
        bin_counts = np.bincount(flat)
        nonzero = np.nonzero(bin_counts)[0]
        return dict(zip(nonzero.tolist(), bin_counts[nonzero].tolist(), strict=True))
    values, counts = np.unique(flat, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def _count_torch(tensor: Any) -> dict[int, int]:
    """Count unique values using torch.bincount (O(n), no sort)."""
    torch_module = _try_import_torch()
    if torch_module is None:
        msg = (
            "torch is required to count torch tensors. "
            "Install with: pip install classcounter[torch]"
        )
        raise ImportError(msg)
    flat = tensor.ravel()
    if flat.dtype == torch_module.bool:
        flat = flat.to(torch_module.int8)
    if flat.numel() > 0 and not flat.is_floating_point() and flat.min() >= 0:
        counts_list = torch_module.bincount(flat).tolist()
        return {i: c for i, c in enumerate(counts_list) if c > 0}
    values, counts = torch_module.unique(flat, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def _count_numba(arr: np.ndarray) -> dict[int, int]:
    """Count unique values using a numba-jitted loop."""
    try:
        from classcounter._numba import bincount_numba
    except ImportError:
        msg = (
            "numba is required for the numba backend. "
            "Install with: pip install classcounter[numba]"
        )
        raise ImportError(msg) from None

    flat = arr.ravel()
    if flat.dtype == np.bool_:
        return _count_bool(flat)
    if flat.size > 0 and _is_nonneg_int(flat):
        max_val = int(flat.max())
        bin_counts = bincount_numba(flat, max_val + 1)
        nonzero = np.nonzero(bin_counts)[0]
        return dict(zip(nonzero.tolist(), bin_counts[nonzero].tolist(), strict=True))
    values, counts = np.unique(flat, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def _load_raster(path: Path) -> np.ndarray:
    """Load a raster file into a numpy array via rasterio."""
    try:
        import rasterio
    except ImportError:
        msg = (
            "rasterio is required to read raster files. "
            "Install with: pip install classcounter[geo]"
        )
        raise ImportError(msg) from None

    with rasterio.open(path) as src:
        return src.read()


def _try_import_torch() -> Any:
    """Try to import torch, return module or None. Result is cached."""
    global _torch_module, _torch_checked  # noqa: PLW0603
    if not _torch_checked:
        try:
            import torch

            _torch_module = torch
        except ImportError:
            _torch_module = None
        _torch_checked = True
    return _torch_module
