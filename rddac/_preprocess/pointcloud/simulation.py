"""Simulation matching and mirrored point-cloud access for the pointcloud stage.

Each experiment is aligned (ICP) against its best-matching DDACS simulation.
The hard constraints (geometry, blankholder force) plus the soft ones (sheet
thickness, friction coefficient derived from the oil film) select one of the
396 RDDAC-matched simulations. The packaged ``sim_params.csv`` is the
``rddac == True`` subset of the public DDACS ``process_parameters.csv``, so
matching needs no external parameter table; the simulation HDF5 files come
from ``rddac download`` (skip ``--no-sim``) and are read via ``ddacs``.
"""

from __future__ import annotations

import io
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from ddacs import h5_tools as _ddacs_h5_tools

from ...spec import SIM_SUBDIR

#: Linear oil-film -> friction-coefficient mapping (clamped): 0.8 g/m^2 -> 0.15, 1.6 g/m^2 -> 0.05.
OIL_MIN_GM2, OIL_MAX_GM2 = 0.8, 1.6
FC_MAX, FC_MIN = 0.15, 0.05


def oil_to_friction(oil_gm2: float) -> float:
    """Friction coefficient for an oil-film density (linear, clamped)."""
    oil = float(np.clip(oil_gm2, OIL_MIN_GM2, OIL_MAX_GM2))
    return FC_MAX - (oil - OIL_MIN_GM2) * (FC_MAX - FC_MIN) / (OIL_MAX_GM2 - OIL_MIN_GM2)


def _mirror(points: np.ndarray) -> np.ndarray:
    """Mirror a quarter part into the full part (4 quadrants)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.column_stack(
        [
            np.concatenate([x, -x, x, -x]),
            np.concatenate([y, y, -y, -y]),
            np.concatenate([z, z, z, z]),
        ]
    )


class SimContext:
    """Per-process simulation access: parameter table + mirrored point-cloud cache.

    Args:
        data_dir: The RDDAC data directory; simulations live in its
            ``simulation/`` subdirectory (from ``rddac download``).
    """

    def __init__(self, data_dir: str | Path):
        self.sim_dir = Path(data_dir) / SIM_SUBDIR
        table_bytes = (resources.files("rddac._preprocess") / "sim_params.csv").read_bytes()
        self.params = pd.read_csv(io.BytesIO(table_bytes))
        self._cache: dict[tuple[int, str], np.ndarray] = {}

    def available(self) -> bool:
        """Whether the simulation directory exists and is non-empty."""
        return self.sim_dir.is_dir() and any(self.sim_dir.iterdir())

    def match(
        self, geometry: str, blankholder_force_kn: int, sheet_thickness_um: float, oil_thickness_gm2: float
    ) -> dict:
        """Best-matching simulation for one experiment.

        Args:
            geometry: ``"concave"`` or ``"convex"``.
            blankholder_force_kn: Blankholder force in kN (100/300/500).
            sheet_thickness_um: Mean measured sheet thickness in um.
            oil_thickness_gm2: Mean measured oil film in g/m^2.

        Returns:
            Match info: ``simulation_id``, matched/target sheet thickness and
            friction coefficient, and the individual absolute errors.
        """
        candidates = self.params[
            (self.params["geometry"] == geometry) & (self.params["blankholder_force"] == blankholder_force_kn * 1000.0)
        ]
        if len(candidates) == 0:
            raise ValueError(f"no simulations for geometry={geometry!r}, blankholder_force={blankholder_force_kn}")
        shtk_target = sheet_thickness_um / 1000.0
        fc_target = oil_to_friction(oil_thickness_gm2)
        err_shtk = np.abs(candidates["sheet_metal_thickness"].to_numpy() - shtk_target)
        err_fc = np.abs(candidates["friction_coefficient"].to_numpy() - fc_target)
        best = int(np.argmin(err_shtk + err_fc))
        row = candidates.iloc[best]
        return {
            "simulation_id": int(row["index"]),
            "matched_shtk": float(row["sheet_metal_thickness"]),
            "matched_fc": float(row["friction_coefficient"]),
            "target_shtk": shtk_target,
            "target_fc": fc_target,
            "error_shtk": float(err_shtk[best]),
            "error_fc": float(err_fc[best]),
        }

    def points(self, simulation_id: int, op: str) -> np.ndarray:
        """Mirrored ``(4N, 3)`` simulation point cloud for one operation.

        Args:
            simulation_id: DDACS simulation id (``index`` column).
            op: ``"op10"`` or ``"op20"``.

        Returns:
            The final-timestep blank nodes of the quarter part, mirrored to
            the full part. Cached in memory per context.
        """
        key = (simulation_id, op.lower())
        if key not in self._cache:
            with _ddacs_h5_tools.open_h5(simulation_id, data_dir=self.sim_dir) as f:
                quarter = f[f"{op.upper()}/blank/node_displacement"][-1]
            self._cache[key] = _mirror(np.asarray(quarter, dtype=np.float64))
        return self._cache[key]
