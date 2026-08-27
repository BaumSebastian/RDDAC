"""The per-experiment pointcloud stage: geometric cleaning + RF fin removal.

The runner calls :func:`preflight` once in the main process (simulation check
+ train-or-load of the fin classifier) and :func:`process_experiment` per
experiment. Worker processes rebuild their context lazily on first use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .. import defaults as d
from . import classifier, features, geometry
from . import registration as reg
from .simulation import SimContext
from .training import train_all

#: Runner contract: this module owns its h5 groups (see runner dispatch).
PROCESSES_FILE = True
IMPLEMENTED = True

_CTX: dict = {}


def preflight(data_dir: str | Path, config: dict, console=None, rebuild_models: bool = False) -> None:
    """Fail fast on missing simulations; train-or-load the fin classifier once.

    Args:
        data_dir: The RDDAC data directory.
        config: The ``[pointcloud]`` parameter table.
        console: Optional rich console.
        rebuild_models: Force retraining even with a valid cache.
    """
    sim_ctx = SimContext(data_dir)
    if not sim_ctx.available():
        raise FileNotFoundError(
            f"simulations not found under {sim_ctx.sim_dir} — the pointcloud stage aligns each scan "
            "to its matching DDACS simulation; run `rddac download` (without --no-sim) first"
        )
    train_all(data_dir, config, console=console, rebuild=rebuild_models)


def _context(data_dir: str | Path, config: dict) -> dict:
    """Per-process lazy context: sim access, model bundles, calibration."""
    key = str(data_dir)
    if _CTX.get("key") != key:
        fp = classifier.fingerprint(
            config.get("rf_n_estimators", d.PC_RF_N_ESTIMATORS), config.get("rf_max_depth", d.PC_RF_MAX_DEPTH)
        )
        bundles = classifier.load_bundles(data_dir, fp)
        if bundles is None:
            raise RuntimeError("fin classifier cache missing/stale — preflight should have trained it")
        _CTX.update(key=key, sim=SimContext(data_dir), bundles=bundles, calib=geometry.load_calibration())
    return _CTX


def process_experiment(raw, out, *, data_dir: str | Path, overwrite: bool, **params) -> str:
    """Process both operations of one experiment into ``out``.

    Args:
        raw: Open raw experiment (read-only h5).
        out: Open output h5 (append mode).
        data_dir: The RDDAC data directory (simulations + model cache).
        overwrite: Recompute even when the output group exists.
        **params: The ``[pointcloud]`` parameter table.

    Returns:
        Runner status string.
    """
    if "pointcloud" not in raw:
        return "no_group"
    if "pointcloud" in out and not overwrite:
        return "exists"
    ctx = _context(data_dir, params)

    part_geometry = str(raw.attrs["geometry"])
    blankholder_kn = int(raw.attrs["blankholder_force"])
    sheet_mean, oil_mean, means_source = _means_for_matching(raw, out)
    match = ctx["sim"].match(part_geometry, blankholder_kn, sheet_mean, oil_mean)

    if "pointcloud" in out:
        del out["pointcloud"]
    pc_group = out.create_group("pointcloud")
    pc_group.attrs["simulation_id"] = match["simulation_id"]
    pc_group.attrs["matched_shtk"] = match["matched_shtk"]
    pc_group.attrs["matched_fc"] = match["matched_fc"]
    pc_group.attrs["matching_error_shtk"] = match["error_shtk"]
    pc_group.attrs["matching_error_fc"] = match["error_fc"]
    pc_group.attrs["measured_sheet_thickness_mean"] = sheet_mean
    pc_group.attrs["measured_oil_thickness_mean"] = oil_mean
    pc_group.attrs["matching_means_source"] = means_source

    for op in ("op10", "op20"):
        sim_pts = ctx["sim"].points(match["simulation_id"], op)
        group_key = f"{part_geometry}_{op}"
        _process_op(raw, pc_group, op, sim_pts, ctx, group_key, part_geometry, params)
    return "processed"


def _means_for_matching(raw, out) -> tuple[float, float, str]:
    """Sheet/oil means for simulation matching, from processed output when present."""
    if "sheet_thickness" in out and "oil_thickness" in out:
        sheet_mean = float(np.nanmean(out["sheet_thickness/data"][:, 1]))
        oil_mean = float(np.nanmean(out["oil_thickness/data"][:, 1]))
        return sheet_mean, oil_mean, "processed"
    from .training import _sheet_oil_means

    sheet_mean, oil_mean = _sheet_oil_means(raw)
    return sheet_mean, oil_mean, "recomputed"


def _process_op(raw, pc_group, op: str, sim_pts: np.ndarray, ctx: dict, group_key: str, part_geometry: str, p: dict):
    grp = raw[f"pointcloud/{op}"]
    shape = (int(grp.attrs["y_shape"]), int(grp.attrs["x_shape"]))
    z_2d = grp["z"][:].reshape(shape)
    lumi_2d = grp["luminescence"][:].reshape(shape)
    calib = ctx["calib"]

    max_angle = (
        p.get("max_wall_angle_concave_deg", d.PC_MAX_WALL_ANGLE_CONCAVE_DEG)
        if part_geometry == "concave"
        else p.get("max_wall_angle_convex_deg", d.PC_MAX_WALL_ANGLE_CONVEX_DEG)
    )
    min_component = p.get("min_component_size", d.PC_MIN_COMPONENT_SIZE)

    valid = geometry.lumi_valid_mask(lumi_2d, p.get("lumi_min_patch_size", d.PC_LUMI_MIN_PATCH_SIZE)) & (z_2d > 0)
    y_mm_per_px = geometry.y_calibration(valid, calib["x_mm_per_pixel"])
    points = geometry.extract_points(z_2d, valid, calib["x_mm_per_pixel"], y_mm_per_px, calib["z_mm_per_unit"])

    tree_xy = cKDTree(points[:, :2])
    angle_seeds = geometry.seed_angle_outliers(points, tree_xy, p.get("k_angle", d.PC_K_ANGLE), max_angle)
    component_seeds = geometry.seed_small_components(points, min_component)
    pre_icp = angle_seeds | component_seeds

    rotation, translation, icp_stats = geometry.run_icp(
        points[~pre_icp],
        sim_pts,
        max_iterations=p.get("icp_max_iterations", d.PC_ICP_MAX_ITERATIONS),
        n_sample=p.get("icp_sample_size", d.PC_ICP_SAMPLE_SIZE),
        seed=d.PC_SEED,
    )
    aligned = points @ rotation.T + translation
    z_offset = geometry.z_at_center(aligned[~pre_icp]) - geometry.z_at_center(sim_pts)
    aligned[:, 2] -= z_offset

    tree_aligned = cKDTree(aligned[:, :2])
    mono_seeds = geometry.seed_radial_monotonicity(
        aligned, tree_aligned, p.get("k_mono", d.PC_K_MONO), p.get("z_tolerance_mm", d.PC_Z_TOLERANCE_MM)
    )
    seeds = pre_icp | mono_seeds
    geometric_outliers, closing_iters = geometry.morphological_closing(
        seeds,
        tree_aligned,
        aligned[:, :2],
        p.get("k_closing", d.PC_K_CLOSING),
        p.get("max_closing_iter", d.PC_MAX_CLOSING_ITER),
    )

    # RF fin cleaner on grid features (aligned z + kd sim distance + lumi).
    z_grid = np.full(shape, np.nan, dtype=np.float32)
    z_grid[valid] = aligned[:, 2].astype(np.float32)
    sim_grid = features.kd_sim_distance_grid(aligned, valid, sim_pts)
    lumi_values = lumi_2d[valid].astype(np.float64)
    lo, hi = float(lumi_values.min()), float(lumi_values.max())
    lumi_grid = np.full(shape, np.nan, dtype=np.float32)
    lumi_grid[valid] = ((lumi_values - lo) / (hi - lo) * 255.0 if hi > lo else lumi_values).astype(np.float32)

    bundle = ctx["bundles"][group_key]
    base = features.compute_features(z_grid.astype(np.float64), sim_grid, lumi_grid, valid)
    x_base, _ = features.features_to_array(base, valid)
    xf = reg.register(valid, bundle["ref_ds"])
    extra = features.registered_columns(
        z_grid,
        valid,
        bundle["prior_reg"].astype(np.float64),
        bundle["expected_reg"].astype(np.float64),
        xf,
        d.PC_DX_MM,
        d.PC_DY_MM,
    )
    rf_removed = classifier.predict_outliers(
        bundle, np.hstack([x_base, extra]), p.get("rf_threshold", d.PC_RF_THRESHOLD)
    )

    removed = geometric_outliers | rf_removed
    cleaned = aligned[~removed]
    sweep = geometry.seed_small_components(cleaned, min_component)
    final_points = cleaned[~sweep].astype(np.float32)

    op_group = pc_group.create_group(op)
    op_group.create_dataset("z", data=final_points, compression="gzip", compression_opts=4)
    op_group.create_dataset(
        "luminescence", data=geometry.pack_luminescence(lumi_2d, valid), compression="gzip", compression_opts=4
    )
    op_group.attrs["x_mm_per_pixel"] = calib["x_mm_per_pixel"]
    op_group.attrs["y_mm_per_pixel"] = y_mm_per_px
    op_group.attrs["z_mm_per_unit"] = calib["z_mm_per_unit"]
    op_group.attrs["icp_rotation"] = rotation
    op_group.attrs["icp_translation"] = translation
    op_group.attrs["icp_z_offset"] = z_offset
    for key, value in icp_stats.items():
        op_group.attrs[key] = value
    op_group.attrs["n_valid_pixels"] = int(valid.sum())
    op_group.attrs["n_angle_seeds"] = int(angle_seeds.sum())
    op_group.attrs["n_component_seeds"] = int(component_seeds.sum())
    op_group.attrs["n_mono_seeds"] = int(mono_seeds.sum())
    op_group.attrs["n_geometric_outliers"] = int(geometric_outliers.sum())
    op_group.attrs["closing_iterations"] = closing_iters
    op_group.attrs["n_rf_removed"] = int(rf_removed.sum())
    op_group.attrs["n_final_sweep"] = int(sweep.sum())
    op_group.attrs["n_final_points"] = int(len(final_points))
    op_group.attrs["outlier_pct"] = 100.0 * float(removed.sum()) / max(int(valid.sum()), 1)
    op_group.attrs["rf_threshold"] = p.get("rf_threshold", d.PC_RF_THRESHOLD)
    op_group.attrs["registration_theta_dy_dx"] = np.array(xf, dtype=np.float64)
