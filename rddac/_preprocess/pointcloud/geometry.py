"""Geometric pointcloud stages: masking, calibration, outlier seeds, ICP.

Ported from the validated internal pipeline. All functions are pure and
operate on numpy arrays; nothing here touches files except
:func:`load_calibration`, which reads the packaged ``calibration.json``.
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
    """The packaged sensor calibration (``x_mm_per_pixel``, ``z_mm_per_unit``)."""
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


def y_calibration(valid_mask: np.ndarray, x_mm_per_pixel: float) -> float:
    """Per-scan y calibration from the square-part assumption.

    Args:
        valid_mask: ``(H, W)`` boolean validity mask.
        x_mm_per_pixel: Known x calibration from the sensor spec.

    Returns:
        Millimeters per pixel along y.
    """
    y_idx, x_idx = np.where(valid_mask)
    x_span_px = x_idx.max() - x_idx.min()
    y_span_px = y_idx.max() - y_idx.min()
    return (x_span_px * x_mm_per_pixel) / y_span_px


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
        y_mm_per_pixel: Y calibration (see :func:`y_calibration`).
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
    src = source[rng.choice(len(source), n_sample, replace=False)].copy() if len(source) > n_sample else source.copy()

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
    q1, q2, q3 = (float(np.percentile(final_distances, p)) for p in (25, 50, 75))
    iqr = q3 - q1
    stats = {
        "icp_mean_distance": float(np.mean(final_distances)),
        "icp_std_distance": float(np.std(final_distances)),
        "icp_median_distance": q2,
        "icp_q1_distance": q1,
        "icp_q3_distance": q3,
        "icp_whisker_low": float(max(np.min(final_distances), q1 - 1.5 * iqr)),
        "icp_whisker_high": float(min(np.max(final_distances), q3 + 1.5 * iqr)),
    }
    return r_total, t_total, stats
