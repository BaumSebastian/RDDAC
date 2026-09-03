"""Geometric pointcloud stages: masking, calibration, outlier seeds, ICP.

Ported from the validated internal pipeline. All functions are pure and
operate on numpy arrays; nothing here touches files except
:func:`load_calibration`, which reads the packaged ``calibration.json``.

Performance note: the seed stages build kNN graphs over ~3 M points per scan
(1.5-2 min per experiment). Since the points lie on the regular pixel grid,
they could be reformulated as image operations on the z grid (gradients,
``ndimage`` labelling/closing) for an estimated 2-3x speed-up. Not done yet:
it changes the validated detection and needs re-validation against the
bundled labels plus a classifier retrain. See the "Runtime" section of the
Point Clouds documentation page.
"""

from __future__ import annotations

import json
import warnings
from importlib import resources

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree, cKDTree


def load_calibration() -> dict:
    """The packaged sensor calibration (``x_mm_per_pixel``, ``y_mm_per_pixel``, ``z_mm_per_unit``)."""
    path = resources.files("rddac._preprocess") / "calibration.json"
    with path.open() as f:
        return json.load(f)["calibration"]


def lumi_valid_mask(lumi_2d: np.ndarray, min_patch_size: int, background_value: float = 0) -> np.ndarray:
    """Valid-surface mask from the luminescence grid via connected components.

    Args:
        lumi_2d: ``(H, W)`` raw luminescence.
        min_patch_size: Minimum pixel count for a connected patch to be kept.
        background_value: Values at/below this are background.

    Returns:
        ``(H, W)`` boolean mask; True marks pixels on a sufficiently large patch.
    """
    foreground = lumi_2d > background_value
    labeled, n = ndimage.label(foreground)
    if n == 0:
        return np.zeros_like(lumi_2d, dtype=bool)
    sizes = ndimage.sum(foreground, labeled, index=range(1, n + 1))
    valid_labels = np.where(sizes >= min_patch_size)[0] + 1
    return np.isin(labeled, valid_labels)


def pack_luminescence(lumi_2d: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Pack luminescence into the processed uint8 format.

    Background is 0; valid pixels are min-max normalized into 1..255 so zero
    stays reserved for "no surface" (Pillow-ready grayscale).

    Args:
        lumi_2d: ``(H, W)`` raw luminescence.
        valid_mask: ``(H, W)`` boolean validity mask.

    Returns:
        ``(H, W) uint8`` grid.
    """
    out = np.zeros(lumi_2d.shape, dtype=np.uint8)
    values = lumi_2d[valid_mask].astype(np.float64)
    if len(values) == 0:
        return out
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else 1.0
    out[valid_mask] = np.clip(np.round((values - lo) / span * 254.0) + 1, 1, 255).astype(np.uint8)
    return out


def extract_points(
    z_2d: np.ndarray,
    valid_mask: np.ndarray,
    x_mm_per_pixel: float,
    y_mm_per_pixel: float,
    z_mm_per_unit: float,
) -> np.ndarray:
    """Extract the centered ``(N, 3)`` point cloud in mm from a raw z grid.

    x/y are centered on the valid-pixel centroid; z is converted to mm and
    centered on its mean (helps ICP convergence).

    Args:
        z_2d: ``(H, W)`` raw z grid (sensor units, 0 = invalid).
        valid_mask: ``(H, W)`` boolean validity mask.
        x_mm_per_pixel: X calibration.
        y_mm_per_pixel: Y calibration (packaged constant, see ``calibration.json``).
        z_mm_per_unit: Z calibration.

    Returns:
        ``(N, 3) float64`` points in the scan frame.
    """
    y_idx, x_idx = np.where(valid_mask)
    x_mm = (x_idx - x_idx.astype(float).mean()) * x_mm_per_pixel
    y_mm = (y_idx - y_idx.astype(float).mean()) * y_mm_per_pixel
    z_mm = z_2d[valid_mask].astype(np.float64) * z_mm_per_unit
    z_mm -= z_mm.mean()
    return np.column_stack([x_mm, y_mm, z_mm])


def seed_angle_outliers(pts: np.ndarray, tree_xy: cKDTree, k: int, max_wall_angle_deg: float) -> np.ndarray:
    """Seed outliers via SVD-based local surface angle.

    Fits a local plane to each point's k nearest xy-neighbours; points whose
    surface normal deviates more than ``max_wall_angle_deg`` from vertical are
    seeded.

    Args:
        pts: ``(N, 3)`` points.
        tree_xy: KD-tree over ``pts[:, :2]``.
        k: Neighbours per local plane fit.
        max_wall_angle_deg: Angle-from-vertical cutoff in degrees.

    Returns:
        ``(N,)`` boolean seed mask.
    """
    _, idxs = tree_xy.query(pts[:, :2], k=k + 1)
    nbrs = pts[idxs]
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    angles = np.degrees(np.arccos(np.clip(np.abs(vt[:, 2, 2]), 0, 1)))
    return angles > max_wall_angle_deg


def seed_radial_monotonicity(pts: np.ndarray, tree_xy: cKDTree, k: int, z_tolerance: float) -> np.ndarray:
    """Seed outliers where z increases when moving radially outward.

    The part center is estimated from the top z-quintile; a point is seeded
    when it sits more than ``z_tolerance`` above the median of its more-inward
    neighbours.

    Args:
        pts: ``(N, 3)`` points.
        tree_xy: KD-tree over ``pts[:, :2]``.
        k: Neighbours considered per point.
        z_tolerance: Allowed z increase in mm.

    Returns:
        ``(N,)`` boolean seed mask.
    """
    z_thresh = np.percentile(pts[:, 2], 80)
    center_xy = pts[pts[:, 2] >= z_thresh, :2].mean(axis=0)
    r = np.hypot(pts[:, 0] - center_xy[0], pts[:, 1] - center_xy[1])

    _, idxs = tree_xy.query(pts[:, :2], k=k + 1)
    nb_idx = idxs[:, 1:]
    inward = r[nb_idx] < (r[:, None] - 0.1)
    n_inward = inward.sum(axis=1)
    z_nb = np.where(inward, pts[nb_idx, 2], np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median_inward = np.nanmedian(z_nb, axis=1)
    return (n_inward > 2) & (pts[:, 2] - median_inward > z_tolerance)


def seed_small_components(pts: np.ndarray, min_component_size: int) -> np.ndarray:
    """Seed outliers from small 3D connected components.

    Builds an epsilon-graph (eps = 3x the median nearest-neighbour distance)
    and marks every connected component smaller than ``min_component_size``.

    Args:
        pts: ``(N, 3)`` points.
        min_component_size: Minimum component size to survive.

    Returns:
        ``(N,)`` boolean seed mask.
    """
    n = len(pts)
    tree_3d = cKDTree(pts)
    dd, _ = tree_3d.query(pts, k=2)
    eps = np.median(dd[:, 1]) * 3
    pairs = tree_3d.query_pairs(r=eps, output_type="ndarray")
    if len(pairs) > 0:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adj = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    else:
        adj = csr_matrix((n, n), dtype=bool)
    _, labels = connected_components(adj, directed=False)
    counts = np.bincount(labels)
    return (counts < min_component_size)[labels]


def morphological_closing(
    seed_mask: np.ndarray, tree_xy: cKDTree, pts_xy: np.ndarray, k: int = 8, max_iter: int = 15
) -> tuple[np.ndarray, int]:
    """Clean a seed mask via repeated dilation + erosion on the kNN graph.

    Args:
        seed_mask: ``(N,)`` boolean seed mask.
        tree_xy: KD-tree over ``pts_xy``.
        pts_xy: ``(N, 2)`` xy coordinates.
        k: Neighbours per point in the closing graph.
        max_iter: Iteration cap.

    Returns:
        ``(final_mask, n_iterations)``.
    """
    _, idxs = tree_xy.query(pts_xy, k=k + 1)
    nb_idx = idxs[:, 1:]
    outlier = seed_mask.copy()
    iterations = 0
    for iterations in range(1, max_iter + 1):
        prev = outlier.sum()
        outlier = outlier | outlier[nb_idx].any(axis=1)  # dilation
        outlier = outlier & ~(~outlier)[nb_idx].any(axis=1)  # erosion
        if outlier.sum() == prev:
            break
    return outlier, iterations


def z_at_center(points: np.ndarray, radius: float = 5.0) -> float:
    """Median z of the points within ``radius`` mm of x=y=0 (nearest point as fallback)."""
    dist = np.hypot(points[:, 0], points[:, 1])
    mask = dist < radius
    if mask.sum() == 0:
        return float(points[np.argmin(dist), 2])
    return float(np.median(points[mask, 2]))


def _voxel_pool(points: np.ndarray, voxel_mm: float = 0.35) -> np.ndarray:
    """One representative point per voxel, so dense regions do not outweigh sparse ones."""
    vox = np.round(points / voxel_mm).astype(np.int64)
    _, keep = np.unique(vox, axis=0, return_index=True)
    return points[np.sort(keep)]


def _distance_stats(distances: np.ndarray) -> dict:
    """Robust summary statistics of nearest-neighbour distances (icp_* attrs)."""
    q1, q2, q3 = (float(np.percentile(distances, p)) for p in (25, 50, 75))
    iqr = q3 - q1
    return {
        "icp_mean_distance": float(np.mean(distances)),
        "icp_std_distance": float(np.std(distances)),
        "icp_median_distance": q2,
        "icp_q1_distance": q1,
        "icp_q3_distance": q3,
        "icp_whisker_low": float(max(np.min(distances), q1 - 1.5 * iqr)),
        "icp_whisker_high": float(min(np.max(distances), q3 + 1.5 * iqr)),
    }


def _deck_center(points: np.ndarray, z_lo: float) -> np.ndarray | None:
    """Outline midpoint (x, y) of the deck: all points above ``z_lo``.

    The midpoint of the robust per-axis extents depends only on the deck
    boundary, not on the scan's anisotropic interior point density.
    """
    deck = points[points[:, 2] > z_lo]
    if len(deck) < 100:
        return None
    lo, hi = np.percentile(deck[:, :2], [0.5, 99.5], axis=0)
    return (lo + hi) / 2.0


def run_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    n_sample: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rigid ICP of ``source`` onto ``target`` (point-to-point, SVD updates).

    The source is subsampled for speed with a SEEDED rng so alignment is
    reproducible run to run.

    Args:
        source: ``(N, 3)`` points to align.
        target: ``(M, 3)`` reference points.
        max_iterations: ICP iterations.
        n_sample: Source subsample size for the iterations.
        seed: Subsample rng seed.

    Returns:
        ``(R, t, stats)`` — apply as ``points @ R.T + t``; ``stats`` holds
        robust summary statistics of the final nearest-neighbour distances.
    """
    rng = np.random.RandomState(seed)
    # Density-neutral subsample: one representative per 0.35 mm voxel first, so
    # densely scanned directions (x pitch 0.077 vs y pitch 0.158 mm) and lightly
    # cleaned regions do not outweigh sparse ones in the fit.
    pool = _voxel_pool(source)
    src = pool[rng.choice(len(pool), n_sample, replace=False)].copy() if len(pool) > n_sample else pool.copy()

    r_total = np.eye(3)
    t_total = np.zeros(3)
    for _ in range(max_iterations):
        tree = KDTree(target)
        _, idx = tree.query(src)
        matched = target[idx]

        src_centroid = src.mean(axis=0)
        tgt_centroid = matched.mean(axis=0)
        h = (src - src_centroid).T @ (matched - tgt_centroid)
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        t = tgt_centroid - r @ src_centroid
        src = src @ r.T + t
        r_total = r @ r_total
        t_total = r @ t_total + t

    final_distances, _ = KDTree(target).query(src)
    return r_total, t_total, _distance_stats(final_distances)


def align_to_simulation(
    points: np.ndarray,
    sim_pts: np.ndarray,
    *,
    inlier_mask: np.ndarray | None = None,
    anchor_height_mm: float = 3.0,
    max_iterations: int = 50,
    n_sample: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Rigidly align a scan to its simulation, anchored on the cup.

    Two ICP passes plus a deck centering. The first pass uses all inlier
    points and brings the scan into the simulation frame; it is robust while
    fins are still present. The second is restricted to the cup (bottom and
    walls): every point higher than the simulation's flange level plus
    ``anchor_height_mm``, selected identically on both sides. This keeps the
    flange, the region most affected by springback and draw-in, from biasing
    the pose, so the remaining wall and radius deviations are attributed to
    the part, not to the alignment. Because the ICP cost is nearly flat in
    x/y for these level-topped parts, a final x/y shift matches the occupancy
    centroids of the two decks, distributing any real width difference
    symmetrically. After each step the z offset is fixed at the cup centre.

    Args:
        points: ``(N, 3)`` calibrated scan points.
        sim_pts: ``(M, 3)`` simulation points (mirrored full part).
        inlier_mask: Optional ``(N,)`` boolean mask of points to fit on
            (e.g. everything except geometric outlier seeds). All points are
            transformed regardless.
        anchor_height_mm: Height above the simulation's flange from which
            points count as cup.
        max_iterations: ICP iterations per pass.
        n_sample: ICP subsample size per pass.
        seed: Subsampling seed (reproducible runs).

    Returns:
        ``(aligned, rotation, translation, stats)``: the transformed points,
        the composed rigid transform (``aligned == points @ rotation.T +
        translation``, z offsets included), and the ICP statistics of the
        cup pass plus ``n_anchor_points``.
    """
    inlier = np.ones(len(points), dtype=bool) if inlier_mask is None else inlier_mask
    z_axis = np.array([0.0, 0.0, 1.0])

    rotation_1, translation_1, _ = run_icp(points[inlier], sim_pts, max_iterations, n_sample, seed)
    aligned = points @ rotation_1.T + translation_1
    z_offset_1 = z_at_center(aligned[inlier]) - z_at_center(sim_pts)
    aligned[:, 2] -= z_offset_1

    flange_z = float(np.percentile(sim_pts[:, 2], 0.5)) + anchor_height_mm
    cup = inlier & (aligned[:, 2] > flange_z)
    sim_cup = sim_pts[sim_pts[:, 2] > flange_z]
    if cup.sum() < 1000 or len(sim_cup) < 100:  # degenerate scan: keep the first pass
        rotation = rotation_1
        translation = translation_1 - z_offset_1 * z_axis
        stats = {"n_anchor_points": int(cup.sum())}
        return aligned, rotation, translation, stats

    rotation_2, translation_2, _ = run_icp(aligned[cup], sim_cup, max_iterations, n_sample, seed)
    aligned = aligned @ rotation_2.T + translation_2
    z_offset_2 = z_at_center(aligned[inlier]) - z_at_center(sim_pts)
    aligned[:, 2] -= z_offset_2

    # The point-to-point cost is nearly flat in x/y (deck and flange are level
    # planes), so the ICP pose inside that valley is ambiguous and tends to
    # pile the real scan-vs-simulation width difference onto one side. Anchor
    # x/y symmetrically instead: match the occupancy centroids of both decks.
    deck_z = float(sim_pts[:, 2].max()) - 2.0
    center_scan = _deck_center(aligned[inlier], deck_z)
    center_sim = _deck_center(sim_pts, deck_z)
    deck_shift = np.zeros(2) if center_scan is None or center_sim is None else center_sim - center_scan
    aligned[:, :2] += deck_shift
    z_offset_3 = z_at_center(aligned[inlier]) - z_at_center(sim_pts)
    aligned[:, 2] -= z_offset_3

    rotation = rotation_2 @ rotation_1
    translation = (
        rotation_2 @ (translation_1 - z_offset_1 * z_axis)
        + translation_2
        - z_offset_2 * z_axis
        + np.append(deck_shift, -z_offset_3)
    )
    distances, _ = KDTree(sim_cup).query(_voxel_pool(aligned[cup]))
    stats = dict(
        _distance_stats(distances),
        n_anchor_points=int(cup.sum()),
        deck_shift_x=float(deck_shift[0]),
        deck_shift_y=float(deck_shift[1]),
    )
    return aligned, rotation, translation, stats
