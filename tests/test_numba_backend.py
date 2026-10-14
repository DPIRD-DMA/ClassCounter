"""Tests for the numba counting backend."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import skip_no_numba

pytestmark = [pytest.mark.numba, skip_no_numba]


@pytest.fixture(autouse=True, params=["serial", "parallel"])
def kernel_mode(request, monkeypatch):
    """Run every test with both the single-threaded and parallel kernels."""
    import classcounter._numba as kernels

    limit = 2**62 if request.param == "serial" else -1
    monkeypatch.setattr(kernels, "_SERIAL_MAX_ELEMENTS", limit)
    return request.param


class TestNumbaBackend:
    def test_basic_count(self, simple_array):
        from classcounter.core import _count_numba

        result = _count_numba(simple_array)
        assert result == {0: 3, 1: 2, 3: 1}

    def test_negative_falls_back_to_unique(self, negative_array):
        from classcounter.core import _count_numba

        result = _count_numba(negative_array)
        assert result == {-2: 1, -1: 1, 0: 1, 1: 2, 2: 3}

    def test_float_falls_back_to_unique(self, float_array):
        from classcounter.core import _count_numba

        result = _count_numba(float_array)
        assert result == {1.0: 2, 2.5: 3}

    def test_empty_array(self, empty_array):
        from classcounter.core import _count_numba

        assert _count_numba(empty_array) == {}

    def test_single_element(self, single_element_array):
        from classcounter.core import _count_numba

        assert _count_numba(single_element_array) == {42: 1}

    def test_shape_agnostic_3d(self, simple_3d_array):
        from classcounter.core import _count_numba

        result = _count_numba(simple_3d_array)
        assert result == {0: 3, 1: 2, 3: 1}


class TestNumbaDtypePaths:
    @pytest.mark.parametrize("fixed_path", [True, False], ids=["fixed-bins", "max-pass"])
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_small_unsigned_full_range(self, dtype, fixed_path, monkeypatch):
        import classcounter.core as core

        bins = np.iinfo(dtype).max + 1
        # Force each path regardless of the array-size cutoff.
        min_size = 0 if fixed_path else 2**62
        monkeypatch.setitem(core._FIXED_BIN_DTYPES, np.dtype(dtype), (bins, min_size))
        top = np.iinfo(dtype).max
        arr = np.array([0, 3, 3, top, top, top], dtype=dtype)
        assert core._count_numba(arr) == {0: 1, 3: 2, int(top): 3}
        mask = np.array([True, False, True, True, False, True])
        assert core._count_numba(arr, mask) == {0: 1, 3: 1, int(top): 2}

    def test_uint16_fixed_bins_only_for_large_arrays(self):
        from classcounter.core import _fixed_bins

        assert _fixed_bins(np.empty(10, dtype=np.uint8)) == 256
        assert _fixed_bins(np.empty(10, dtype=np.uint16)) is None
        assert _fixed_bins(np.empty(1 << 24, dtype=np.uint16)) == 65536
        assert _fixed_bins(np.empty(10, dtype=np.int32)) is None

    @pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
    def test_wide_unsigned_dtypes(self, dtype):
        from classcounter.core import _count_numba

        arr = np.array([0, 7, 7, 100_000], dtype=dtype)
        assert _count_numba(arr) == {0: 1, 7: 2, 100_000: 1}

    @pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
    def test_signed_small_path(self, dtype):
        from classcounter.core import _count_numba

        arr = np.array([5, 0, 5, 2], dtype=dtype)
        assert _count_numba(arr) == {0: 1, 2: 1, 5: 2}


class TestNumbaSignedSinglePass:
    """Force the large-array single-pass kernel by lowering the size threshold."""

    @pytest.fixture(autouse=True)
    def _force_single_pass(self, monkeypatch):
        import classcounter.core as core

        monkeypatch.setattr(core, "_SIGNED_SINGLE_PASS_BYTES", 0)

    @pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
    def test_matches_numpy(self, dtype):
        from classcounter.core import _count_numba, _count_numpy

        rng = np.random.default_rng(0)
        arr = rng.integers(0, 20, size=100_003, dtype=dtype)
        assert _count_numba(arr) == _count_numpy(arr)

    def test_negative_falls_back_to_unique(self):
        from classcounter.core import _count_numba

        # Negative value deep in the array, not in the first chunk.
        arr = np.zeros(10_000, dtype=np.int32)
        arr[-1] = -4
        assert _count_numba(arr) == {-4: 1, 0: 9_999}

    def test_max_above_bin_limit_reruns(self):
        from classcounter.core import _SIGNED_BIN_LIMIT, _count_numba

        big = _SIGNED_BIN_LIMIT + 10
        arr = np.array([1, 1, big, 3], dtype=np.int64)
        assert _count_numba(arr) == {1: 2, 3: 1, big: 1}

    def test_single_element(self):
        from classcounter.core import _count_numba

        assert _count_numba(np.array([9], dtype=np.int32)) == {9: 1}
