"""Deterministic retraining of the RF fin classifier from the bundled labels.

Training needs, for every labeled (experiment, op) task, the aligned z grid,
the kd sim-distance grid and the luminescence grid — all derivable from data
the user already has locally (raw experiments + simulations). Prepared grids
are cached as npz under the model cache directory and deleted after a
successful retrain unless ``keep_prepared`` is set.

Everything is seeded and iterates in sorted order, so a retrain on the same
inputs reproduces the same models bit-for-bit (per scikit-learn version).
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

from .. import defaults as d
from ..h5_access import open_raw
from . import classifier, features, geometry
from . import registration as reg
from .simulation import SimContext

GROUPS = ("concave_op10", "concave_op20", "convex_op10", "convex_op20")
_TASK_RE = re.compile(r"^(\d{4})_(op10|op20)\.npz$")


def labeled_tasks() -> list[tuple[str, int, str]]:
    """The bundled label tasks as ``(name, experiment_id, op)``, sorted."""
    root = resources.files("rddac._preprocess") / "labels"
    tasks = []
    for entry in root.iterdir():
        match = _TASK_RE.match(entry.name)
        if match:
            tasks.append((entry.name[:-4], int(match.group(1)), match.group(2)))
    return sorted(tasks)


def load_label(name: str) -> np.ndarray:
    """The bundled outlier mask for one task name (``(H, W) bool``)."""
    data = (resources.files("rddac._preprocess") / "labels" / f"{name}.npz").read_bytes()
    import io

    return np.load(io.BytesIO(data))["outlier_mask"].astype(bool)


def _experiment_table(data_dir: str | Path) -> pd.DataFrame:
    csv = Path(data_dir) / "process_parameters.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} is required for retraining (it maps experiments to geometry)")
    return pd.read_csv(csv).set_index("index")


def _sheet_oil_means(raw: h5py.File) -> tuple[float, float]:
    """Mean sheet thickness (um) and oil film (g/m^2) computed from raw data."""
    from .. import oil as oil_mod
    from .. import sheet as sheet_mod

    sheet_mean, oil_mean = float("nan"), float("nan")
    if "sheet_thickness" in raw:
        processed, _ = sheet_mod.process(raw["sheet_thickness/data"][:])
        sheet_mean = float(np.nanmean(processed[:, 1]))
    if "oil_thickness" in raw:
        processed, _ = oil_mod.process(raw["oil_thickness/data"][:])
        oil_mean = float(np.nanmean(processed[:, 1]))
    return sheet_mean, oil_mean


def prepare_task(
    name: str, exp_id: int, op: str, data_dir: str | Path, sim_ctx: SimContext, out_path: Path, cfg: dict
) -> bool:
    """Prepare one task's grids (aligned z, kd sim distance, lumi, mask) into ``out_path``.

    Returns False when the raw experiment is not available locally.
    """
    if out_path.exists():
        return True
    try:
        raw = open_raw(exp_id, data_dir)
    except FileNotFoundError:
        return False
    with raw:
        table = _experiment_table(data_dir)
        row = table.loc[exp_id]
        sheet_mean, oil_mean = _sheet_oil_means(raw)
        match = sim_ctx.match(str(row["geometry"]), int(row["blankholder_force"]), sheet_mean, oil_mean)
        sim_pts = sim_ctx.points(match["simulation_id"], op)

        grp = raw[f"pointcloud/{op}"]
        shape = (int(grp.attrs["y_shape"]), int(grp.attrs["x_shape"]))
        z_2d = grp["z"][:].reshape(shape)
        lumi_2d = grp["luminescence"][:].reshape(shape)

    calib = geometry.load_calibration()
    valid = geometry.lumi_valid_mask(lumi_2d, cfg.get("lumi_min_patch_size", d.PC_LUMI_MIN_PATCH_SIZE)) & (z_2d > 0)
    y_mm_per_px = geometry.y_calibration(valid, calib["x_mm_per_pixel"])
    points = geometry.extract_points(z_2d, valid, calib["x_mm_per_pixel"], y_mm_per_px, calib["z_mm_per_unit"])
    rotation, translation, _ = geometry.run_icp(
        points,
        sim_pts,
        max_iterations=cfg.get("icp_max_iterations", d.PC_ICP_MAX_ITERATIONS),
        n_sample=cfg.get("icp_sample_size", d.PC_ICP_SAMPLE_SIZE),
        seed=d.PC_SEED,
    )
    aligned = points @ rotation.T + translation
    aligned[:, 2] -= geometry.z_at_center(aligned) - geometry.z_at_center(sim_pts)

    z_grid = np.full(z_2d.shape, np.nan, dtype=np.float32)
    z_grid[valid] = aligned[:, 2].astype(np.float32)
    sim_grid = np.full(z_2d.shape, np.nan, dtype=np.float32)
    sim_grid[valid] = cKDTree(sim_pts).query(aligned)[0].astype(np.float32)
    lumi_grid = np.full(z_2d.shape, np.nan, dtype=np.float32)
    lumi_values = lumi_2d[valid].astype(np.float64)
    lo, hi = lumi_values.min(), lumi_values.max()
    lumi_grid[valid] = ((lumi_values - lo) / (hi - lo) * 255.0 if hi > lo else lumi_values).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, z_mm=z_grid, valid_mask=valid, sim_distance=sim_grid, lumi=lumi_grid)
    return True


def _group_of(name: str, geometry_by_id: dict[int, str]) -> str:
    exp_id, op = int(name.split("_")[0]), name.split("_")[1]
    return f"{geometry_by_id[exp_id]}_{op}"


def train_all(data_dir: str | Path, cfg: dict, console=None, rebuild: bool = False) -> dict[str, dict]:
    """Train-or-load all group bundles; returns mapping group -> bundle.

    Args:
        data_dir: The RDDAC data directory (raw experiments + simulations).
        cfg: The ``[pointcloud]`` parameter table.
        console: Optional rich console for progress messages.
        rebuild: Force retraining even when the cache fingerprint matches.

    Returns:
        Mapping group name to trained bundle (see :mod:`.classifier`).
    """
    log = console.print if console is not None else (lambda *a, **k: None)
    fp = classifier.fingerprint(
        cfg.get("rf_n_estimators", d.PC_RF_N_ESTIMATORS), cfg.get("rf_max_depth", d.PC_RF_MAX_DEPTH)
    )
    if not rebuild:
        bundles = classifier.load_bundles(data_dir, fp)
        if bundles is not None:
            log(f"[dim]fin classifier: loaded {len(bundles)} cached models[/dim]")
            return bundles

    log("[bold]fin classifier: retraining from bundled labels (one-time, ~30-90 min)[/bold]")
    sim_ctx = SimContext(data_dir)
    table = _experiment_table(data_dir)
    geometry_by_id = {int(i): str(g) for i, g in table["geometry"].items()}
    prepared_dir = classifier.cache_dir(data_dir) / "prepared"

    tasks = labeled_tasks()
    usable: dict[str, list[str]] = {g: [] for g in GROUPS}
    for name, exp_id, op in tasks:
        out_path = prepared_dir / f"{name}.npz"
        if prepare_task(name, exp_id, op, data_dir, sim_ctx, out_path, cfg):
            usable[_group_of(name, geometry_by_id)].append(name)
        else:
            log(f"[yellow]labeled experiment {exp_id:04d} not available locally — skipped[/yellow]")

    bundles: dict[str, dict] = {}
    labels_used: dict[str, list[str]] = {}
    for group in GROUPS:
        members = sorted(usable[group])
        if len(members) < 3:
            raise RuntimeError(
                f"group {group}: only {len(members)} of its labeled experiments are available locally — "
                "download the full dataset (rddac download) to retrain the fin classifier"
            )
        log(f"  {group}: training on {len(members)} labeled samples")
        bundles[group] = _train_group(group, members, prepared_dir, cfg)
        labels_used[group] = members

    classifier.save_bundles(data_dir, bundles, {"labels_used": labels_used}, fp)
    if not cfg.get("keep_prepared", d.PC_KEEP_PREPARED):
        for path in prepared_dir.glob("*.npz"):
            os.remove(path)
    log("[dim]fin classifier: models cached[/dim]")
    return bundles


def _load_prepared(prepared_dir: Path, name: str) -> dict:
    data = np.load(prepared_dir / f"{name}.npz")
    return {key: data[key] for key in ("z_mm", "valid_mask", "sim_distance", "lumi")}


def _train_group(group: str, members: list[str], prepared_dir: Path, cfg: dict) -> dict:
    """Build prior/consensus + balanced matrix for one group and fit the RF."""
    ref_ds = _load_prepared(prepared_dir, members[0])["valid_mask"][::4, ::4]

    transforms: dict[str, tuple[int, float, float]] = {}
    osum = vsum = zsum = zweight = None
    labels: dict[str, np.ndarray] = {}
    for name in members:
        sample = _load_prepared(prepared_dir, name)
        valid = sample["valid_mask"]
        outlier = load_label(name) & valid
        labels[name] = outlier
        xf = reg.register(valid, ref_ds)
        transforms[name] = xf
        wall = valid & ~outlier
        z_filled = np.nan_to_num(sample["z_mm"].astype(np.float64), nan=0.0)
        of = reg.to_reference(outlier, *xf)
        vf = reg.to_reference(valid, *xf)
        zf = reg.to_reference(np.where(wall, z_filled, 0.0), *xf, order=1)
        wf = reg.to_reference(wall.astype(float), *xf, order=1)
        osum = of if osum is None else osum + of
        vsum = vf if vsum is None else vsum + vf
        zsum = zf if zsum is None else zsum + zf
        zweight = wf if zweight is None else zweight + wf
    prior_reg = np.clip(osum, 0, None) / np.clip(vsum, 1, None)
    expected_reg = zsum / np.clip(zweight, 1e-3, None)

    # Two-pass balanced matrix: choose rows from labels only, then stream
    # each sample's features once into a preallocated array (memory-lean).
    rng = np.random.RandomState(d.PC_SEED)
    y_parts = [labels[name][_load_prepared(prepared_dir, name)["valid_mask"]].astype(np.int8) for name in members]
    counts = np.array([len(y) for y in y_parts])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    y_all = np.concatenate(y_parts)
    outlier_rows = np.where(y_all == 1)[0]
    inlier_rows = np.where(y_all == 0)[0]
    if len(outlier_rows) == 0:
        selected = np.arange(len(y_all))
    else:
        n_inlier = min(len(inlier_rows), int(len(outlier_rows) * d.PC_BALANCE_RATIO))
        selected = np.concatenate([rng.choice(inlier_rows, n_inlier, replace=False), outlier_rows])
        if len(selected) > d.PC_MAX_TRAIN_ROWS:
            selected = rng.choice(selected, d.PC_MAX_TRAIN_ROWS, replace=False)
    selected.sort()

    x_train = None
    feature_names: list[str] = []
    for k, name in enumerate(members):
        local = selected[(selected >= offsets[k]) & (selected < offsets[k + 1])]
        if len(local) == 0:
            continue
        sample = _load_prepared(prepared_dir, name)
        valid = sample["valid_mask"]
        base = features.compute_features(
            sample["z_mm"].astype(np.float64), sample["sim_distance"], sample["lumi"], valid
        )
        x_base, base_names = features.features_to_array(base, valid)
        extra = features.registered_columns(
            sample["z_mm"], valid, prior_reg, expected_reg, transforms[name], d.PC_DX_MM, d.PC_DY_MM
        )
        if x_train is None:
            feature_names = base_names + list(features.REGISTERED_FEATURES)
            x_train = np.empty((len(selected), len(feature_names)), dtype=np.float32)
        rows = np.hstack([x_base, extra])
        x_train[np.searchsorted(selected, local)] = rows[local - offsets[k]]

    model = RandomForestClassifier(
        n_estimators=cfg.get("rf_n_estimators", d.PC_RF_N_ESTIMATORS),
        max_depth=cfg.get("rf_max_depth", d.PC_RF_MAX_DEPTH),
        n_jobs=-1,
        random_state=d.PC_SEED,
    )
    model.fit(x_train, y_all[selected])
    return {
        "group": group,
        "model": model,
        "feature_names": feature_names,
        "prior_reg": prior_reg.astype(np.float32),
        "expected_reg": expected_reg.astype(np.float32),
        "ref_ds": ref_ds,
    }
