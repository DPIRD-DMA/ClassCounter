"""Tests for error handling: bad types, missing dependencies."""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

import classcounter.core as _core
from classcounter import count_classes


class TestTypeErrors:

    def test_unsupported_type_list(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            count_classes([1, 2, 3])

    def test_unsupported_type_int(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            count_classes(42)

    def test_unsupported_type_dict(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            count_classes({"a": 1})

    def test_unsupported_type_none(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            count_classes(None)

    def test_string_path_raises_without_rasterio_or_file(self):
        with pytest.raises((ImportError, Exception)):
            count_classes("nonexistent_file.tif")


class TestImportErrors:

    def test_torch_import_cache_miss(self):
        """Exercise the except branch inside _try_import_torch."""
        saved_checked = _core._torch_checked
        saved_module = _core._torch_module
        try:
            _core._torch_checked = False
            _core._torch_module = None
            with patch.dict(sys.modules, {"torch": None}):
                result = _core._try_import_torch()
                assert result is None
        finally:
            _core._torch_checked = saved_checked
            _core._torch_module = saved_module

    def test_count_torch_guard_without_torch(self):
        """Exercise the ImportError guard inside _count_torch directly."""
        import torch

        tensor = torch.tensor([1, 2, 3])
        with patch("classcounter.core._try_import_torch", return_value=None):
            with pytest.raises(ImportError, match="torch is required"):
                _core._count_torch(tensor)

    def test_numba_backend_without_numba(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        with patch.dict(sys.modules, {"classcounter._numba": None}):
            with pytest.raises(ImportError, match="numba is required"):
                _core._count_numba(arr)

    def test_raster_without_rasterio(self):
        with patch.dict(sys.modules, {"rasterio": None}):
            with pytest.raises(ImportError, match="rasterio is required"):
                _core._load_raster(pytest.importorskip("pathlib").Path("some_file.tif"))
