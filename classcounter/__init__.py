from importlib.metadata import version

from classcounter.core import count_classes, save_counts_to_raster

__all__ = ["count_classes", "save_counts_to_raster", "__version__"]
__version__ = version("classcounter")
