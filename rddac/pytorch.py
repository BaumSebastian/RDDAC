"""PyTorch IterableDataset adapter for RDDAC — a thin subclass of :class:`ddacs.pytorch.DDACSDataset`.

All machinery (view/field resolution, zip streaming, DataLoader-worker and DDP
sharding, seeded shuffle, `MissingDataWarning`) is inherited; the subclass only
injects :data:`rddac.spec.RDDAC_SPEC` so member names are zero-padded and the
manifest resolves to the RDDAC dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

try:
    from ddacs.pytorch import DDACSDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for RDDACDataset. Install with `pip install rddac[torch]` "
        "or install a flavour from https://pytorch.org/get-started/locally/."
    ) from exc

from .spec import RDDAC_SPEC

DEFAULT_DATA_DIR = RDDAC_SPEC.default_data_dir


class RDDACDataset(DDACSDataset):
    """Streaming PyTorch dataset for a single RDDAC Croissant view.

    Yields a `dict[str, numpy.ndarray]` per experiment. Field selection is
    derived from the Croissant view + field-map; sharding across DataLoader
    workers and DDP ranks is decided inside `__iter__`, so the same instance
    works under `num_workers=0`, `num_workers=N` and DDP.

    Views must source `field-map` (HDF5) fields only; for views that include
    `process-parameters` metadata columns use `rddac.streaming.iter_view`, or
    build the view with `with_metadata=False`.

    Args:
        view: Name of the RecordSet to stream (e.g. "force-curve").
        source: Override the manifest URL / path.
        data_dir: Local data directory (default "./data"). Pass `None` to skip
            local-file discovery.
        dataset: A pre-loaded `mlcroissant.Dataset` (e.g. one mutated by
            `rddac.add_view`) — the way to stream a custom view.
        sim_ids: Explicit allowlist of experiment ids (name kept for drop-in
            DDACS compatibility). Requested ids that cannot be served warn via
            `rddac.streaming.MissingDataWarning`.
        where: Predicate applied to each `process_parameters.csv` row before
            any zip is opened.
        shuffle: Per-shard seeded shuffle; call `set_epoch` between epochs.
        seed: Base seed for the per-shard shuffle.
    """

    def __init__(
        self,
        view: str,
        source: str | Path | None = None,
        data_dir: str | Path | None = DEFAULT_DATA_DIR,
        dataset=None,
        sim_ids: list[int] | None = None,
        where: Callable[[pd.Series], bool] | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        super().__init__(
            view,
            source=source,
            data_dir=data_dir,
            dataset=dataset,
            sim_ids=sim_ids,
            where=where,
            shuffle=shuffle,
            seed=seed,
            spec=RDDAC_SPEC,
        )
