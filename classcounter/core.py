from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_torch_module: Any = None
_torch_checked = False
_has_numba_cached: bool | None = None

# Unsigned dtypes whose entire value range fits in a small bin array — we can
# skip the `flat.max()` pass and pre-allocate `dtype.max + 1` bins instead.
_FIXED_BIN_DTYPES: dict[Any, int] = {
    np.dtype(np.uint8): 256,
    np.dtype(np.uint16): 65536,
}

# Upper bound for the single-pass signed-int histogram. 65536 keeps the
# per-thread local-counts row at 512 KB (fits in L2). Values above this
# trigger a one-time fall back to the two-pass max+bincount path.
_SIGNED_BIN_LIMIT = 65536

# Byte size above which the single-pass signed-int kernel beats the
# minmax+bincount approach. Below ~L3-cache size, the second scan in the
# two-pass approach hits cache and is essentially free, so the extra
# per-element work of the single-pass kernel makes it slower.
_SIGNED_SINGLE_PASS_BYTES = 50 * 1024 * 1024

_BACKENDS = ("auto", "numpy", "numba", "torch")


def _is_nonneg_int(flat: np.ndarray) -> bool:
    """Check if a flat array is non-negative integer (skips min() for unsigned dtypes).

    Precondition: flat.size > 0.
    """
    if not np.issubdtype(flat.dtype, np.integer):
        return False
    if np.issubdtype(flat.dtype, np.unsignedinteger):
        return True
    return flat.size > 0 and flat.min() >= 0


def _count_bool(flat: np.ndarray, mask: np.ndarray | None = None) -> dict[int, int]:
    """Fast path for boolean arrays using count_nonzero."""
    if mask is None:
        n_true = int(np.count_nonzero(flat))
        n_false = flat.size - n_true
    else:
        n_true = int(np.count_nonzero(flat & mask))
        n_false = int(np.count_nonzero(mask)) - n_true
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
    save_metadata: bool = False,
    metadata_prefix: str | None = None,
    mask: np.ndarray | Any | None = None,
    backend: str = "auto",
) -> dict[int | str, int] | dict[int | str, float]:
    """Count occurrences of each unique class value.

    Automatically selects the optimal backend (NumPy, Numba, or PyTorch)
    based on input type.

    Args:
        data: A numpy array, torch tensor, or path to a raster file (GeoTIFF).
        names: Optional mapping of integer class values to string names.
            When provided, returned dict keys are the mapped names. Classes in
            the mapping but absent from the data receive a count of 0.
        percent: If True, return percentages (0-100) instead of raw counts.
            Values are rounded to 2 decimal places.
        save_metadata: If True, write the results as GDAL metadata tags in the
            source GeoTIFF. Only valid when *data* is a file path.
        metadata_prefix: Custom prefix for metadata tag names. When *None*,
            defaults to ``CLASS_COUNT_`` for raw counts or ``CLASS_PERCENT_``
            when *percent* is True.
        mask: Optional boolean array or tensor, the same shape as the data or
            broadcastable to it. Only elements where *mask* is True are
            counted, and percentages are of the masked total. For a file
            path, the mask applies to the loaded ``(bands, rows, cols)``
            raster, so a ``(rows, cols)`` mask covers every band.
        backend: ``"auto"`` (default) picks PyTorch for tensors, otherwise
            Numba if it is installed, otherwise NumPy. ``"numpy"``,
            ``"numba"`` or ``"torch"`` forces that backend, so results and
            first-call cost don't depend on what else happens to be
            installed. ``"numpy"`` and ``"numba"`` take arrays or file paths;
            ``"torch"`` takes tensors.

    Returns:
        Dict mapping class values (or names) to their counts or percentages.

    Raises:
        TypeError: If data is not an ndarray, Tensor, str, or Path, if
            *mask* is not boolean, or if *backend* does not support the
            input type (e.g. ``"numba"`` with a torch tensor).
        ValueError: If *mask* cannot be broadcast to the data's shape, or
            *backend* is not a recognised name.
        ImportError: If a torch tensor is given but torch is not installed,
            a file path is given but rasterio is not installed, or
            ``backend="numba"`` is requested without numba installed.

    Examples:
        >>> import numpy as np
        >>> count_classes(np.array([0, 0, 1, 1, 1, 2]))
        {0: 2, 1: 3, 2: 1}

        >>> count_classes(np.array([0, 0, 1]), names={0: "a", 1: "b"})
        {'a': 2, 'b': 1}

        >>> count_classes(np.array([0, 0, 1, 1, 1, 2]), percent=True)
        {0: 33.33, 1: 50.0, 2: 16.67}
    """
    if backend not in _BACKENDS:
        msg = f"backend must be one of {_BACKENDS}, got {backend!r}."
        raise ValueError(msg)

    arr, is_tensor = _resolve_input(data)

    if backend == "auto":
        backend = "torch" if is_tensor else "numba" if _has_numba() else "numpy"
    elif backend == "torch" and not is_tensor:
        msg = (
            "backend='torch' requires a torch tensor. Convert with "
            "torch.from_numpy(), or use backend='numpy' or 'numba'."
        )
        raise TypeError(msg)
    elif backend != "torch" and is_tensor:
        msg = (
            f"backend={backend!r} does not accept torch tensors. Use "
            "backend='torch', or convert with tensor.cpu().numpy()."
        )
        raise TypeError(msg)

    if backend == "torch":
        counts = _count_torch(arr, mask)
    elif backend == "numba":
        counts = _count_numba(arr, mask)
    else:
        counts = _count_numpy(arr, mask)

    result: dict[int | str, int] | dict[int | str, float]
    if names is not None:
        named: dict[int | str, int] = {name: counts.get(k, 0) for k, name in names.items()}
        for k, v in counts.items():
            if k not in names:
                warnings.warn(
                    f"Class {k} found in data but missing from names mapping. "
                    f"Using integer value as key.",
                    UserWarning,
                    stacklevel=2,
                )
                named[k] = v
        result = named
    else:
        result = counts  # type: ignore[assignment]

    if percent:
        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total * 100, 2) for k, v in result.items()}

    if save_metadata:
        if not isinstance(data, str | Path):
            msg = "save_metadata=True requires data to be a file path, not an array or tensor."
            raise ValueError(msg)
        if metadata_prefix is None:
            metadata_prefix = "CLASS_PERCENT_" if percent else "CLASS_COUNT_"
        save_counts_to_raster(data, result, prefix=metadata_prefix)

    return result


def _resolve_input(
    data: str | Path | np.ndarray | Any,
) -> tuple[np.ndarray | Any, bool]:
    """Load data if needed and determine whether to use torch.

    Returns:
        Tuple of (array_or_tensor, use_torch).
    """
    if isinstance(data, str | Path):
        return _load_raster(Path(data)), False
    if isinstance(data, np.ndarray):
        return data, False
    # A tensor can only exist if torch is already imported, so look it up
    # rather than importing it: `import torch` takes ~1s and would otherwise
    # be paid by every process that only ever counts numpy arrays.
    torch_module = sys.modules.get("torch")
    if torch_module is not None and isinstance(data, torch_module.Tensor):
        return data, True

    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def _flat_mask_numpy(mask: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Validate *mask* and return it as a flat bool array matching ``arr.ravel()``."""
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        msg = f"mask must be a boolean array, got dtype {m.dtype}."
        raise TypeError(msg)
    try:
        m = np.broadcast_to(m, shape)
    except ValueError:
        msg = f"mask shape {m.shape} cannot be broadcast to data shape {shape}."
        raise ValueError(msg) from None
    return m.ravel()


def _count_numpy(arr: np.ndarray, mask: Any = None) -> dict[int, int]:
    """Count unique values using numpy.bincount."""
    flat = arr.ravel()
    if mask is not None:
        # Fallback backend: a plain boolean-index copy keeps this simple.
        flat = flat[_flat_mask_numpy(mask, arr.shape)]
    if flat.dtype == np.bool_:
        return _count_bool(flat)
    if flat.size > 0 and _is_nonneg_int(flat):
        bin_counts = np.bincount(flat)
        nonzero = np.nonzero(bin_counts)[0]
        return dict(zip(nonzero.tolist(), bin_counts[nonzero].tolist(), strict=True))
    values, counts = np.unique(flat, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def _flat_mask_torch(torch_module: Any, mask: Any, tensor: Any) -> Any:
    """Validate *mask* and return it as a flat bool tensor on *tensor*'s device."""
    m = mask if isinstance(mask, torch_module.Tensor) else torch_module.as_tensor(np.asarray(mask))
    if m.dtype != torch_module.bool:
        msg = f"mask must be a boolean array or tensor, got dtype {m.dtype}."
        raise TypeError(msg)
    try:
        m = torch_module.broadcast_to(m, tensor.shape)
    except RuntimeError:
        msg = (
            f"mask shape {tuple(m.shape)} cannot be broadcast to data shape {tuple(tensor.shape)}."
        )
        raise ValueError(msg) from None
    return m.to(tensor.device).reshape(-1)


def _count_torch(tensor: Any, mask: Any = None) -> dict[int, int]:
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
    if flat.numel() == 0:
        return {}

    # Unsigned values are known non-negative, so bincount can skip the
    # full-pass min() check (and on GPU, the implicit sync that .min() forces).
    nonneg = flat.dtype in _torch_unsigned_dtypes(torch_module)
    m = None if mask is None else _flat_mask_torch(torch_module, mask, tensor)
    is_uint64 = flat.dtype == getattr(torch_module, "uint64", None)
    if nonneg and flat.dtype != torch_module.uint8:
        # bincount, where and indexing only implement uint8 among unsigned
        # dtypes. uint16/uint32 widen losslessly to int64. uint64 is
        # reinterpreted as int64 (same bits, no copy); values >= 2**63 show
        # up negative and are mapped back below.
        if is_uint64:
            flat = flat.view(torch_module.int64)
            nonneg = False
        else:
            flat = flat.to(torch_module.int64)

    n_excluded = None
    if m is not None:
        # Boolean indexing (flat[m]) has a data-dependent shape and forces a
        # host sync. Instead, overwrite excluded elements with 0, count as
        # usual, and subtract the excluded total from the 0 bin on-device.
        zero = torch_module.zeros((), dtype=flat.dtype, device=flat.device)
        flat = torch_module.where(m, flat, zero)
        n_excluded = m.numel() - m.sum()

    if not flat.is_floating_point() and (nonneg or bool(flat.min() >= 0)):
        counts = torch_module.bincount(flat)
        if n_excluded is not None:
            counts[0] -= n_excluded
        return {i: c for i, c in enumerate(counts.tolist()) if c > 0}

    values, counts = torch_module.unique(flat, return_counts=True)
    if n_excluded is not None:
        counts = counts - (values == 0) * n_excluded
    result = {v: c for v, c in zip(values.tolist(), counts.tolist(), strict=True) if c > 0}
    if is_uint64:
        result = {(v + 2**64 if v < 0 else v): c for v, c in result.items()}
    return result


_torch_unsigned_dtypes_cache: tuple[Any, ...] | None = None


def _torch_unsigned_dtypes(torch_module: Any) -> tuple[Any, ...]:
    """Return the set of unsigned integer dtypes available in this torch build."""
    global _torch_unsigned_dtypes_cache  # noqa: PLW0603
    if _torch_unsigned_dtypes_cache is None:
        names = ("uint8", "uint16", "uint32", "uint64")
        _torch_unsigned_dtypes_cache = tuple(
            getattr(torch_module, n) for n in names if hasattr(torch_module, n)
        )
    return _torch_unsigned_dtypes_cache


def _count_numba(arr: np.ndarray, mask: Any = None) -> dict[int, int]:
    """Count unique values using a numba-jitted loop."""
    try:
        from classcounter._numba import (
            bincount_numba,
            bincount_signed_numba,
            minmax_numba,
        )
    except ImportError:
        msg = (
            "numba is required for the numba backend. Install with: pip install classcounter[numba]"
        )
        raise ImportError(msg) from None

    flat = arr.ravel()
    if mask is not None:
        return _count_numba_masked(flat, _flat_mask_numpy(mask, arr.shape))
    if flat.dtype == np.bool_:
        return _count_bool(flat)
    if flat.size == 0:
        return {}

    # uint8 / uint16: dtype max is small enough to allocate bins directly,
    # so we skip the entire max() pass over the data.
    fixed_bins = _FIXED_BIN_DTYPES.get(flat.dtype)
    if fixed_bins is not None:
        bin_counts = bincount_numba(flat, fixed_bins)
        nonzero = np.nonzero(bin_counts)[0]
        return dict(zip(nonzero.tolist(), bin_counts[nonzero].tolist(), strict=True))

    if np.issubdtype(flat.dtype, np.integer):
        if np.issubdtype(flat.dtype, np.unsignedinteger):
            max_val = int(flat.max())
            bin_counts = bincount_numba(flat, max_val + 1)
        elif flat.nbytes >= _SIGNED_SINGLE_PASS_BYTES:
            # Large signed array: single-pass histogram that also tracks
            # min/max so we skip a separate scan for the max.
            bin_counts, min_val, max_val = bincount_signed_numba(flat, _SIGNED_BIN_LIMIT)
            if min_val < 0:
                values, counts = np.unique(flat, return_counts=True)
                return dict(zip(values.tolist(), counts.tolist(), strict=True))
            if max_val >= _SIGNED_BIN_LIMIT:
                bin_counts = bincount_numba(flat, int(max_val) + 1)
        else:
            # Small signed array: cheap separate min/max scan avoids the
            # ~0.5 ms cost of allocating the large single-pass workspace.
            min_val, max_val = minmax_numba(flat)
            if min_val < 0:
                values, counts = np.unique(flat, return_counts=True)
                return dict(zip(values.tolist(), counts.tolist(), strict=True))
            bin_counts = bincount_numba(flat, int(max_val) + 1)
        nonzero = np.nonzero(bin_counts)[0]
        return dict(zip(nonzero.tolist(), bin_counts[nonzero].tolist(), strict=True))

    values, counts = np.unique(flat, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist(), strict=True))


def _count_numba_masked(flat: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    """Masked variant of :func:`_count_numba`; skips excluded elements in-kernel."""
    from classcounter._numba import bincount_masked_numba, minmax_masked_numba

    if flat.dtype == np.bool_:
        return _count_bool(flat, mask)
    if flat.size == 0:
        return {}

    fixed_bins = _FIXED_BIN_DTYPES.get(flat.dtype)
    if fixed_bins is not None:
        bin_counts = bincount_masked_numba(flat, mask, fixed_bins)
    elif np.issubdtype(flat.dtype, np.integer):
        min_val, max_val, found = minmax_masked_numba(flat, mask)
        if not found:
            return {}
        if min_val < 0:
            values, counts = np.unique(flat[mask], return_counts=True)
            return dict(zip(values.tolist(), counts.tolist(), strict=True))
        bin_counts = bincount_masked_numba(flat, mask, int(max_val) + 1)
    else:
        values, counts = np.unique(flat[mask], return_counts=True)
        return dict(zip(values.tolist(), counts.tolist(), strict=True))

    nonzero = np.nonzero(bin_counts)[0]
    return dict(zip(nonzero.tolist(), bin_counts[nonzero].tolist(), strict=True))


def save_counts_to_raster(
    path: str | Path,
    counts: dict[int | str, int] | dict[int | str, float],
    *,
    prefix: str | None = None,
) -> None:
    """Save class counts as GDAL metadata tags in a GeoTIFF file.

    Each entry is written as a tag named ``{prefix}{key}`` with the
    count (or percentage) as the value. Existing tags matching the prefix
    are cleared before writing so stale entries don't persist.

    Args:
        path: Path to an existing GeoTIFF file.
        counts: Dict of class counts as returned by :func:`count_classes`.
        prefix: Tag name prefix. Defaults to ``CLASS_COUNT_`` for raw counts
            or ``CLASS_PERCENT_`` for percentage values (auto-detected from
            the ``count_classes`` convenience parameter).

    Raises:
        ImportError: If rasterio is not installed.
        FileNotFoundError: If *path* does not exist.
    """
    if prefix is None:
        prefix = "CLASS_COUNT_"

    try:
        import rasterio
    except ImportError:
        msg = (
            "rasterio is required to write raster metadata. "
            "Install with: pip install classcounter[geo]"
        )
        raise ImportError(msg) from None

    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    with rasterio.open(path, "r+") as dst:
        existing = dst.tags()
        merged = {k: "" for k in existing if k.startswith(prefix)}
        merged.update({f"{prefix}{k}": str(v) for k, v in counts.items()})
        dst.update_tags(**merged)


def _load_raster(path: Path) -> np.ndarray:
    """Load a raster file into a numpy array via rasterio."""
    try:
        import rasterio
    except ImportError:
        msg = (
            "rasterio is required to read raster files. Install with: pip install classcounter[geo]"
        )
        raise ImportError(msg) from None

    with rasterio.open(path) as src:
        data: np.ndarray = src.read()
    return data


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
