"""Raw-experiment access shared by the runner and the pointcloud training."""

from __future__ import annotations

import os
from pathlib import Path

import h5py

from ..h5_tools import open_h5


def open_raw(exp_id: int, data_dir: str | Path) -> h5py.File:
    """Open one raw experiment read-only: loose file first, zips via ``open_h5``."""
    loose = os.path.join(str(data_dir), f"{exp_id:04d}.h5")
    if os.path.isfile(loose):
        return h5py.File(loose, "r")
    return open_h5(exp_id, data_dir=data_dir)
