"""RDDAC — Real Deep Drawing and Cutting Dataset.

Python interface for the RDDAC dataset (experimental measurements of sheet
metal forming; the physical counterpart to the DDACS simulations). Built on a
Croissant 1.1 manifest: `rddac.load()` returns an `mlcroissant.Dataset` whose
`records(view)` streams the data; `add_view`, `open_h5`, `inspect_h5` are
convenience helpers around the same manifest.

The public surface mirrors the `ddacs` package, so DDACS code ports by
swapping the import.

Examples:
    >>> import rddac
    >>> ds = rddac.load(data_dir="./data")
    >>> for record in rddac.streaming.iter_view("force-curve", data_dir="./data"):
    ...     ...

    >>> with rddac.open_h5(42, data_dir="./data") as f:
    ...     rddac.inspect_h5(f)

Note: prefer ``rddac.streaming.iter_view`` (or ``RDDACDataset``) over
``ds.records(view)`` for the h5-backed views — mlcroissant's own records()
walks the full multi-GB zips per view and is impractically slow there.
"""

__version__ = "1.1.0"

from . import streaming
from .croissant import add_view, load
from .h5_tools import inspect_h5, open_h5
from .spec import RDDAC_SPEC, DatasetSpec
from .visualization import (
    plot_force,
    plot_point_cloud,
    plot_scan,
    plot_traverse,
    scan_to_pointcloud,
)

try:
    from .pytorch import RDDACDataset
except ImportError:
    pass

__all__ = [
    "__version__",
    # Dataset identity (consumed by the ddacs machinery via spec=)
    "RDDAC_SPEC",
    "DatasetSpec",
    # Croissant entry point + helpers
    "load",
    "add_view",
    # HDF5 helpers
    "open_h5",
    "inspect_h5",
    # Streaming pipeline (offline iteration + numpy export)
    "streaming",
    # PyTorch (optional — only available if torch is installed)
    "RDDACDataset",
    # Visualization
    "plot_scan",
    "plot_point_cloud",
    "plot_force",
    "plot_traverse",
    "scan_to_pointcloud",
]
