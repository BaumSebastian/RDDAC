"""Streaming iteration and numpy export for RDDAC — thin wrappers over :mod:`ddacs.streaming`.

`iter_view` is the plain-Python counterpart to `RDDACDataset.__iter__`: it
yields one ``dict[str, numpy.ndarray]`` per experiment with no torch
dependency. `export_to_numpy` materializes a view as flat ``.npy`` memmaps
(fixed shapes); `export_to_numpy_per_sim` writes one ``.npz`` per experiment
(ragged shapes, e.g. the raw force tables); `load_export` reads an export back.

The ``sim_ids=`` keyword names are kept identical to ``ddacs`` so DDACS code
ports by swapping the import. Requested ids that cannot be served locally
raise a suppressible :class:`MissingDataWarning`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ddacs import streaming as _ddacs_streaming
from ddacs.streaming import (  # noqa: F401  — dataset-agnostic API + internals
    MissingDataWarning,
    _apply_transforms,
    _as_array,
    _build_field_specs,
    _build_unified_index,
    _extract_record,
    _LoadedExport,
    _parse_jsonpath,
    _progress_iter,
    _resolve_sim_ids,
    _warn_missing,
    load_export,
)

from .spec import RDDAC_SPEC

DEFAULT_DATA_DIR = RDDAC_SPEC.default_data_dir

__all__ = [
    "iter_view",
    "export_to_numpy",
    "export_to_numpy_per_sim",
    "load_export",
    "MissingDataWarning",
]


def iter_view(
    view: str,
    *,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
    sim_ids: list[int] | None = None,
    where: Callable[[pd.Series], bool] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield one record per RDDAC experiment for a Croissant view.

    Args:
        view: Name of the RecordSet to stream (published, e.g. ``force-curve``,
            or added via :func:`rddac.add_view`).
        source: Override the Croissant manifest URL/path.
        data_dir: Directory holding ``metadata.json``, ``process_parameters.csv``
            and either loose ``h5/<id>.h5`` files or the dataset zips.
        dataset: A pre-loaded ``mlcroissant.Dataset`` (carries ``add_view``
            mutations).
        sim_ids: Optional allowlist of experiment ids (name kept for drop-in
            DDACS compatibility). Requested ids that cannot be served warn via
            :class:`MissingDataWarning`.
        where: Predicate applied to each ``process_parameters.csv`` row before
            any HDF5 file is touched.

    Yields:
        A ``dict[str, np.ndarray]`` per experiment, keyed by view-field aliases
        (plus the private ``_sim_id`` scratch key).
    """
    return _ddacs_streaming.iter_view(
        view,
        source=source,
        data_dir=data_dir,
        dataset=dataset,
        sim_ids=sim_ids,
        where=where,
        spec=RDDAC_SPEC,
    )


def export_to_numpy(
    view: str,
    out_dir: str | Path,
    *,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
    sim_ids: list[int] | None = None,
    where: Callable[[pd.Series], bool] | None = None,
    transforms: dict[str, Callable[[Any], Any]] | None = None,
    record_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    show_progress: bool = False,
) -> dict[str, Path]:
    """Materialize a Croissant view as flat ``.npy`` memmaps on disk.

    Requires every record to share the same shape per field — raw RDDAC force
    and traverse tables vary per experiment, so either slice/resample them via
    ``record_transform`` or use :func:`export_to_numpy_per_sim`. See
    :func:`ddacs.streaming.export_to_numpy` for the full parameter reference.
    """
    return _ddacs_streaming.export_to_numpy(
        view,
        out_dir,
        source=source,
        data_dir=data_dir,
        dataset=dataset,
        sim_ids=sim_ids,
        where=where,
        transforms=transforms,
        record_transform=record_transform,
        show_progress=show_progress,
        spec=RDDAC_SPEC,
    )


def export_to_numpy_per_sim(
    view: str,
    out_dir: str | Path,
    *,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
    sim_ids: list[int] | None = None,
    where: Callable[[pd.Series], bool] | None = None,
    transforms: dict[str, Callable[[Any], Any]] | None = None,
    record_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    compressed: bool = False,
    show_progress: bool = False,
) -> Path:
    """Write one ``<experiment_id>.npz`` per experiment under ``out_dir``.

    Same pipeline as :func:`export_to_numpy` but fields may have
    experiment-dependent shapes (the natural fit for RDDAC's raw tables). See
    :func:`ddacs.streaming.export_to_numpy_per_sim` for details.
    """
    return _ddacs_streaming.export_to_numpy_per_sim(
        view,
        out_dir,
        source=source,
        data_dir=data_dir,
        dataset=dataset,
        sim_ids=sim_ids,
        where=where,
        transforms=transforms,
        record_transform=record_transform,
        compressed=compressed,
        show_progress=show_progress,
        spec=RDDAC_SPEC,
    )
