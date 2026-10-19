# Changelog

All notable changes to ClassCounter are documented in this file.

## [0.2.1] - 2026-10-19

### Changed
- The sdist leaves out notebooks, benchmarks, images, CI config and other
  dev-only files (200 KB to 28 KB); it still includes the package,
  tests, LICENSE, README and CHANGELOG
- A plain `pytest` run no longer runs the isolated install tests, which need
  network access, so the suite works offline (e.g. conda-forge builds). Run
  them with `pytest -m install`

## [0.2.0] - 2026-10-19

### Added
- `mask=` parameter to count only selected elements; accepts a boolean array
  or tensor broadcastable to the data shape (for files, a `(rows, cols)` mask
  covers every band)
- `backend=` parameter (`"auto"`, `"numpy"`, `"numba"`, `"torch"`) to force a
  backend, with clear errors when it is missing or can't take the input
- Python 3.14 support

### Changed
- Faster Numba counting: fixed-size bins for uint8 (and uint16 from 16M
  elements), single-pass vectorised min/max for signed ints, and a fused
  histogram + min/max kernel for large arrays. At 100M uint8 elements Numba
  drops from 17 ms to 2.8 ms
- Numba uses single-threaded kernels below 1M elements, avoiding 20-100 ms
  stalls waiting for pool threads on a busy machine
- Numba kernels are cached on disk, so later processes skip compilation
- Torch is no longer imported when counting NumPy arrays or files (saves
  ~1 s per process)
- Torch skips the min() check for unsigned dtypes, avoiding a GPU sync

### Fixed
- Torch uint16 and uint32 tensors crashed in `bincount`; uint64 tensors now
  work too

## [0.1.0] - 2026-02-23

### Added
- Initial public release
- NumPy, Numba, and PyTorch backends for fast class counting
- GeoTIFF support via rasterio
- Name mapping for readable output
- Percentage calculation mode
- Automatic backend selection based on input type
- GPU acceleration (CUDA) via PyTorch
- Full test suite