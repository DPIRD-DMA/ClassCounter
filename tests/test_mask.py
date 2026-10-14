"""Tests for the ``mask`` parameter (count only selected elements)."""

from __future__ import annotations

import numpy as np
import pytest

from classcounter import count_classes
from classcounter.core import _count_numpy
from tests.conftest import HAS_NUMBA, HAS_TORCH, skip_no_rasterio, skip_no_torch

DTYPES = [
    np.bool_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
]


def _make_data(dtype, size=10_007, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == np.bool_:
        return rng.random(size) < 0.3
    if np.issubdtype(dtype, np.floating):
        return rng.integers(0, 6, size).astype(dtype) / 2
    lo = -3 if np.issubdtype(dtype, np.signedinteger) else 0
    return rng.integers(lo, 9, size).astype(dtype)


def _make_mask(size, seed=1):
    return np.random.default_rng(seed).random(size) < 0.6


def _backends():
    """(id, count_fn) pairs; count_fn takes numpy data and a numpy mask."""
    backends = [("numpy", _count_numpy)]
    if HAS_NUMBA:
        from classcounter.core import _count_numba

        backends.append(("numba", _count_numba))

        def numba_parallel(data, mask):
            # Test arrays are small enough to take the single-threaded
            # kernels; force the parallel ones too.
            import classcounter._numba as kernels

            saved = kernels._SERIAL_MAX_ELEMENTS
            kernels._SERIAL_MAX_ELEMENTS = -1
            try:
                return _count_numba(data, mask)
            finally:
                kernels._SERIAL_MAX_ELEMENTS = saved

        backends.append(("numba-parallel", numba_parallel))
    if HAS_TORCH:
        import torch

        from classcounter.core import _count_torch

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:

            def fn(data, mask, dev=dev):
                t = torch.from_numpy(data).to(dev)
                m = None if mask is None else torch.from_numpy(mask).to(dev)
                return _count_torch(t, m)

            backends.append((f"torch-{dev}", fn))
    return backends


BACKENDS = _backends()
BACKEND_IDS = [b[0] for b in BACKENDS]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
class TestMaskMatchesIndexing:
    def test_matches_boolean_index(self, backend, dtype):
        _, fn = backend
        data = _make_data(dtype)
        mask = _make_mask(data.size)
        assert fn(data, mask) == _count_numpy(data[mask])

    def test_all_true_matches_no_mask(self, backend, dtype):
        _, fn = backend
        data = _make_data(dtype)
        assert fn(data, np.ones(data.size, dtype=bool)) == fn(data, None)

    def test_all_false_is_empty(self, backend, dtype):
        _, fn = backend
        data = _make_data(dtype)
        assert fn(data, np.zeros(data.size, dtype=bool)) == {}


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
class TestMaskedOutValuesIgnored:
    def test_negative_values_masked_out(self, backend):
        # Negatives only in the excluded region must not affect the result.
        _, fn = backend
        data = np.array([-5, 1, 2, -7, 2], dtype=np.int32)
        mask = np.array([False, True, True, False, True])
        assert fn(data, mask) == {1: 1, 2: 2}

    def test_huge_values_masked_out(self, backend):
        # A large excluded value must not size the histogram or appear.
        _, fn = backend
        data = np.array([3, 3, 2_000_000_000, 1], dtype=np.int64)
        mask = np.array([True, True, False, True])
        assert fn(data, mask) == {1: 1, 3: 2}

    def test_zero_class_kept_when_selected(self, backend):
        # Excluded elements are counted as 0 internally on torch; real zeros
        # must survive the correction.
        _, fn = backend
        data = np.array([0, 0, 1, 0, 1], dtype=np.uint8)
        mask = np.array([True, False, True, True, False])
        assert fn(data, mask) == {0: 2, 1: 1}

    def test_2d_data_and_mask(self, backend):
        _, fn = backend
        data = np.array([[0, 1, 2], [2, 2, 1]], dtype=np.uint8)
        mask = np.array([[True, False, True], [True, True, False]])
        assert fn(data, mask) == {0: 1, 2: 3}


class TestMaskCountClasses:
    def test_names_absent_class_gets_zero(self):
        data = np.array([0, 1, 1, 2], dtype=np.uint8)
        mask = np.array([True, True, True, False])
        result = count_classes(data, names={0: "a", 1: "b", 2: "c"}, mask=mask)
        assert result == {"a": 1, "b": 2, "c": 0}

    def test_all_false_with_names_gives_zeros(self):
        data = np.array([0, 1], dtype=np.uint8)
        result = count_classes(data, names={0: "a", 1: "b"}, mask=np.zeros(2, dtype=bool))
        assert result == {"a": 0, "b": 0}

    def test_percent_of_masked_total(self):
        data = np.array([0, 0, 1, 1, 1, 1], dtype=np.uint8)
        mask = np.array([True, True, True, False, False, False])
        assert count_classes(data, percent=True, mask=mask) == {0: 66.67, 1: 33.33}

    def test_broadcast_mask(self):
        data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.uint8)
        mask = np.array([[True, False], [False, True]])  # applies to each band
        assert count_classes(data, mask=mask) == {0: 1, 3: 1, 4: 1, 7: 1}

    def test_shape_mismatch_raises(self):
        data = np.zeros((2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="cannot be broadcast"):
            count_classes(data, mask=np.ones((3, 2), dtype=bool))

    def test_non_bool_mask_raises(self):
        data = np.zeros(4, dtype=np.uint8)
        with pytest.raises(TypeError, match="mask must be a boolean"):
            count_classes(data, mask=np.ones(4, dtype=np.uint8))

    def test_none_mask_counts_everything(self):
        data = np.array([0, 1, 1], dtype=np.uint8)
        assert count_classes(data, mask=None) == count_classes(data)


@skip_no_torch
class TestMaskTorchInputs:
    def test_numpy_mask_with_tensor(self):
        import torch

        data = torch.tensor([0, 1, 1, 2], dtype=torch.uint8)
        mask = np.array([True, False, True, True])
        assert count_classes(data, mask=mask) == {0: 1, 1: 1, 2: 1}

    def test_shape_mismatch_raises(self):
        import torch

        data = torch.zeros((2, 3), dtype=torch.uint8)
        with pytest.raises(ValueError, match="cannot be broadcast"):
            count_classes(data, mask=torch.ones((3, 2), dtype=torch.bool))

    def test_non_bool_mask_raises(self):
        import torch

        data = torch.zeros(4, dtype=torch.uint8)
        with pytest.raises(TypeError, match="mask must be a boolean"):
            count_classes(data, mask=torch.ones(4, dtype=torch.uint8))


@skip_no_rasterio
def test_mask_with_file_path(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    data = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8)
    path = tmp_path / "test.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        transform=from_bounds(0, 0, 3, 2, 3, 2),
        crs="EPSG:4326",
    ) as dst:
        dst.write(data, 1)

    mask = np.array([[True, True, False], [True, False, False]])  # (rows, cols)
    assert count_classes(path, mask=mask) == {0: 1, 1: 1, 2: 1}


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
class TestMaskSparseAndLayout:
    @pytest.mark.parametrize("pos", [0, 5_000, -1])
    def test_single_selected_element(self, backend, pos):
        # Most per-thread chunks select nothing; exercises the found/merge
        # logic in the masked min/max kernel.
        _, fn = backend
        data = np.arange(10_007, dtype=np.int32) % 7
        mask = np.zeros(data.size, dtype=bool)
        mask[pos] = True
        assert fn(data, mask) == {int(data[pos]): 1}

    def test_only_negative_selected(self, backend):
        _, fn = backend
        data = np.full(10_007, 4, dtype=np.int16)
        data[-1] = -9
        mask = np.zeros(data.size, dtype=bool)
        mask[-1] = True
        assert fn(data, mask) == {-9: 1}

    def test_non_contiguous_data_and_mask(self, backend):
        _, fn = backend
        data = _make_data(np.uint8, size=60 * 70).reshape(60, 70).T
        mask = _make_mask(data.size).reshape(60, 70).T
        assert not data.flags.c_contiguous
        assert fn(data, mask) == _count_numpy(data[mask])


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("threads", [1, 3])
def test_numba_respects_set_num_threads(threads, monkeypatch):
    import numba

    import classcounter._numba as kernels

    monkeypatch.setattr(kernels, "_SERIAL_MAX_ELEMENTS", -1)

    from classcounter.core import _count_numba

    data = _make_data(np.int32)
    mask = _make_mask(data.size)
    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        assert _count_numba(data) == _count_numpy(data)
        assert _count_numba(data, mask) == _count_numpy(data[mask])
    finally:
        numba.set_num_threads(previous)


def _cuda_available():
    if not HAS_TORCH:
        return False
    import torch

    return torch.cuda.is_available()


skip_no_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


@skip_no_cuda
class TestMaskCuda:
    @pytest.mark.parametrize("mask_kind", ["numpy", "torch-cpu"])
    def test_cpu_mask_with_cuda_tensor(self, mask_kind):
        import torch

        data = _make_data(np.uint8)
        mask = _make_mask(data.size)
        m = mask if mask_kind == "numpy" else torch.from_numpy(mask)
        tensor = torch.from_numpy(data).cuda()
        assert count_classes(tensor, mask=m) == _count_numpy(data[mask])

    def test_mask_adds_no_gpu_syncs(self):
        import warnings

        import torch

        from classcounter.core import _count_torch

        tensor = torch.from_numpy(_make_data(np.uint8)).cuda()
        mask = torch.from_numpy(_make_mask(tensor.numel())).cuda()
        _count_torch(tensor, mask)  # warm up
        torch.cuda.synchronize()

        def syncs(fn):
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fn()
            finally:
                torch.cuda.set_sync_debug_mode(0)
            return sum("synchroniz" in str(w.message).lower() for w in caught)

        assert syncs(lambda: _count_torch(tensor, mask)) <= syncs(lambda: _count_torch(tensor))
