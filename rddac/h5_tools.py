"""HDF5 access helpers for RDDAC — thin wrappers over :mod:`ddacs.h5_tools`.

`open_h5` resolves the RDDAC manifest, locates the zip member matching the
requested experiment (zero-padded: ``42`` -> ``0042.h5``) and returns an
`h5py.File`. `inspect_h5` pretty-prints any `h5py.File` or path.

Both are re-exported as `rddac.open_h5` and `rddac.inspect_h5`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
from ddacs import h5_tools as _ddacs_h5_tools
from ddacs.h5_tools import inspect_h5  # noqa: F401  — dataset-agnostic

from .spec import RDDAC_SPEC

DEFAULT_DATA_DIR = RDDAC_SPEC.default_data_dir


def open_h5(
    experiment_id: int,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
) -> h5py.File:
    """Return an `h5py.File` for the requested RDDAC experiment.

    Looks the manifest up, walks the locally mapped zips and reads the h5
    member matching the zero-padded ``<experiment_id>.h5`` (e.g. ``0042.h5``)
    into a `BytesIO`. The returned object is read-only, supports the `with`
    idiom and can be indexed like any other `h5py.File`.

    Args:
        experiment_id: The experiment index (matches the h5 filename inside
            the zip; ``42`` -> ``0042.h5``).
        source: Override the Croissant manifest URL / path.
        data_dir: Directory searched for already-downloaded zips. Pass `None`
            to skip the local lookup entirely.
        dataset: A pre-loaded `mlcroissant.Dataset` (e.g. from `rddac.load`).
            When given, `source` and `data_dir` are ignored.

    Raises:
        FileNotFoundError: No locally mapped zip contained the requested h5.
    """
    return _ddacs_h5_tools.open_h5(experiment_id, source=source, data_dir=data_dir, dataset=dataset, spec=RDDAC_SPEC)
