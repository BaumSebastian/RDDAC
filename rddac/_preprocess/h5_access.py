"""Raw-experiment access shared by the runner and the pointcloud training."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import h5py

from ..h5_tools import open_h5


def open_raw(exp_id: int, data_dir: str | Path) -> h5py.File:
    """Open one raw experiment read-only: loose file first, zips via ``open_h5``."""
    loose = os.path.join(str(data_dir), f"{exp_id:04d}.h5")
    if os.path.isfile(loose):
        return h5py.File(loose, "r")
    return open_h5(exp_id, data_dir=data_dir)


def available_ids(data_dir: str | Path) -> set[int] | None:
    """Experiment ids that exist locally: loose ``<id>.h5`` files plus zip members.

    Returns ``None`` when nothing is found (unknown layout — let the caller
    try every id). Listing zip members is cheap and avoids one manifest load
    per missing experiment when the selection (e.g. the full
    ``process_parameters.csv``) is larger than the local data, as with the
    small bundle.
    """
    root = Path(data_dir)
    if not root.is_dir():
        return None
    ids: set[int] = set()
    for path in root.rglob("*"):
        if path.suffix == ".h5" and path.stem.isdigit():
            ids.add(int(path.stem))
        elif path.suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    names = zf.namelist()
            except zipfile.BadZipFile:
                continue
            ids.update(int(Path(n).stem) for n in names if n.endswith(".h5") and Path(n).stem.isdigit())
    return ids or None
