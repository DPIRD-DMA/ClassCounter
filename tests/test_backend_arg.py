"""Tests for the ``backend`` parameter and numba on-disk kernel caching."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from classcounter import count_classes
from tests.conftest import skip_no_numba, skip_no_torch


@pytest.fixture
def data():
    return np.array([0, 0, 1, 2, 2, 2], dtype=np.uint8)


class TestBackendArg:

    @pytest.mark.parametrize(
        "backend",
        [
            "auto",
            "numpy",
            pytest.param("numba", marks=skip_no_numba),
        ],
    )
    def test_array_backends_agree(self, data, backend):
        assert count_classes(data, backend=backend) == {0: 2, 1: 1, 2: 3}

    def test_numpy_backend_does_not_use_numba(self, data, monkeypatch):
        import classcounter.core as core

        def boom(*args, **kwargs):
            raise AssertionError("numba backend should not be called")

        monkeypatch.setattr(core, "_count_numba", boom)
        assert count_classes(data, backend="numpy") == {0: 2, 1: 1, 2: 3}

    def test_unknown_backend_raises(self, data):
        with pytest.raises(ValueError, match="backend must be one of"):
            count_classes(data, backend="cupy")

    def test_numba_missing_raises_import_error(self, data, monkeypatch):
        # A None entry in sys.modules makes `import classcounter._numba` fail.
        monkeypatch.setitem(sys.modules, "classcounter._numba", None)
        with pytest.raises(ImportError, match="pip install classcounter\\[numba\\]"):
            count_classes(data, backend="numba")

    def test_torch_backend_rejects_array(self, data):
        with pytest.raises(TypeError, match="backend='torch' requires a torch tensor"):
            count_classes(data, backend="torch")


@skip_no_torch
class TestBackendArgTensors:

    def test_torch_backend_with_tensor(self, data):
        import torch

        tensor = torch.from_numpy(data)
        assert count_classes(tensor, backend="torch") == {0: 2, 1: 1, 2: 3}
        assert count_classes(tensor, backend="auto") == {0: 2, 1: 1, 2: 3}

    @pytest.mark.parametrize("backend", ["numpy", "numba"])
    def test_array_backends_reject_tensor(self, data, backend):
        import torch

        with pytest.raises(TypeError, match="does not accept torch tensors"):
            count_classes(torch.from_numpy(data), backend=backend)


_CACHE_PROBE = """
import numpy as np
from numba.core.registry import CPUDispatcher

from classcounter import _numba as kernels
from classcounter import count_classes

for n in (1000, 3_000_000):  # below and above the serial-kernel threshold
    count_classes(np.zeros(n, dtype=np.uint8), backend="numba")
    count_classes(np.ones(n, dtype=np.int32), backend="numba")
    count_classes(np.zeros(n, dtype=np.uint8), backend="numba", mask=np.ones(n, dtype=bool))
used = [k for k in vars(kernels).values() if isinstance(k, CPUDispatcher)]
print(sum(sum(k.stats.cache_hits.values()) for k in used),
      sum(sum(k.stats.cache_misses.values()) for k in used))
"""


@skip_no_numba
def test_numba_kernels_cached_across_processes(tmp_path):
    """A second process loads compiled kernels from disk instead of recompiling."""
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}

    def run() -> tuple[int, int]:
        out = subprocess.run(
            [sys.executable, "-c", _CACHE_PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        hits, misses = out.stdout.split()
        return int(hits), int(misses)

    cold_hits, compiled = run()
    assert cold_hits == 0
    assert compiled > 0
    # Warm cache: everything compiled last time now loads from disk.
    assert run() == (compiled, 0)


def test_counting_arrays_does_not_import_torch():
    code = (
        "import sys, numpy as np\n"
        "from classcounter import count_classes\n"
        "count_classes(np.array([0, 1, 1], dtype=np.uint8))\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


@pytest.mark.geo
class TestBackendArgFilePath:

    @pytest.fixture
    def tif(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds

        path = tmp_path / "t.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=3,
            count=1,
            dtype="uint8",
            transform=from_bounds(0, 0, 3, 1, 3, 1),
            crs="EPSG:4326",
        ) as dst:
            dst.write(np.array([[0, 2, 2]], dtype=np.uint8), 1)
        return path

    @pytest.mark.parametrize(
        "backend", ["numpy", pytest.param("numba", marks=skip_no_numba)]
    )
    def test_array_backends_accept_path(self, tif, backend):
        assert count_classes(tif, backend=backend) == {0: 1, 2: 2}

    def test_torch_backend_rejects_path(self, tif):
        with pytest.raises(TypeError, match="backend='torch' requires a torch tensor"):
            count_classes(tif, backend="torch")
