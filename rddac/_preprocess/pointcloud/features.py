"""Feature grids for the RF fin cleaner.

All base features are computed on ``(H, W)`` grids and preserve NaN for
invalid pixels; :func:`features_to_array` flattens them to the RF's row
layout. The four registered-frame features (position prior, deviation from
the consensus surface, registered x/y) are appended by
:func:`registered_columns`.

The feature schema (names and order) is part of the model contract: a
trained model is only valid for exactly this layout.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace, maximum_filter, minimum_filter, uniform_filter
from scipy.spatial import cKDTree

from .registration import from_reference

#: Names appended by :func:`registered_columns`, in order.
REGISTERED_FEATURES = ("prior", "dd_dev", "reg_x", "reg_y")


def _local_std(z: np.ndarray, size: int) -> np.ndarray:
    mean = uniform_filter(np.nan_to_num(z, nan=0.0), size=size)
    mean_sq = uniform_filter(np.nan_to_num(z**2, nan=0.0), size=size)
    var = np.clip(mean_sq - mean**2, 0, None)
    out = np.sqrt(var).astype(np.float32)
    out[np.isnan(z)] = np.nan
    return out


def _local_range(z: np.ndarray, size: int) -> np.ndarray:
    mask = np.isnan(z)
    out = maximum_filter(np.where(mask, -np.inf, z), size=size) - minimum_filter(np.where(mask, np.inf, z), size=size)
    out[mask] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def _max_abs_diff_from_center(z: np.ndarray, size: int) -> np.ndarray:
    z_filled = np.nan_to_num(z, nan=0.0)
    mask = np.isnan(z)
    diff_high = np.abs(z_filled - maximum_filter(z_filled, size=size))
    diff_low = np.abs(z_filled - minimum_filter(z_filled, size=size))
    out = np.maximum(diff_high, diff_low).astype(np.float32)
    out[mask] = np.nan
    return out


def _gradient_features(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_filled = np.nan_to_num(z, nan=0.0)
    mask = np.isnan(z)
    gy, gx = np.gradient(z_filled)
    grad_mag = np.hypot(gx, gy).astype(np.float32)
    surface_angle = np.degrees(np.arctan(grad_mag)).astype(np.float32)
    grad_mag[mask] = np.nan
    surface_angle[mask] = np.nan
    return grad_mag, surface_angle


def _laplacian(z: np.ndarray) -> np.ndarray:
    out = laplace(np.nan_to_num(z, nan=0.0)).astype(np.float32)
    out[np.isnan(z)] = np.nan
    return out


def _radial_distance(valid_mask: np.ndarray) -> np.ndarray:
    y_idx, x_idx = np.where(valid_mask)
    if len(y_idx) == 0:
        return np.full(valid_mask.shape, np.nan, dtype=np.float32)
    cy, cx = y_idx.mean(), x_idx.mean()
    yy, xx = np.mgrid[: valid_mask.shape[0], : valid_mask.shape[1]]
    out = np.hypot(yy - cy, xx - cx).astype(np.float32)
    out[~valid_mask] = np.nan
    return out


def _valid_neighbor_fraction(valid_mask: np.ndarray, size: int) -> np.ndarray:
    out = uniform_filter(valid_mask.astype(np.float64), size=size).astype(np.float32)
    out[~valid_mask] = np.nan
    return out


def kd_sim_distance_grid(points_aligned: np.ndarray, valid_mask: np.ndarray, sim_pts: np.ndarray) -> np.ndarray:
    """Per-pixel nearest-neighbour distance to the simulation point cloud.

    Args:
        points_aligned: ``(N, 3)`` ICP-aligned points, one per True pixel of
            ``valid_mask`` in ``np.where`` order.
        valid_mask: ``(H, W)`` boolean validity mask.
        sim_pts: ``(M, 3)`` simulation points (mirrored full part).

    Returns:
        ``(H, W) float32`` grid, NaN outside the mask.
    """
    distances, _ = cKDTree(sim_pts).query(points_aligned)
    grid = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    grid[valid_mask] = distances.astype(np.float32)
    return grid


def compute_features(
    z_mm: np.ndarray, sim_distance: np.ndarray, lumi: np.ndarray, valid_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """All base feature grids for one sample.

    Args:
        z_mm: ``(H, W)`` aligned z in mm, NaN invalid.
        sim_distance: ``(H, W)`` distance to the matched simulation, NaN invalid.
        lumi: ``(H, W)`` processed luminescence, NaN invalid.
        valid_mask: ``(H, W)`` boolean mask.

    Returns:
        Mapping feature name to ``(H, W) float32`` grid.
    """
    features: dict[str, np.ndarray] = {}
    for size in (5, 11, 21):
        features[f"z_std_{size}"] = _local_std(z_mm, size)
    for size in (5, 11):
        features[f"z_range_{size}"] = _local_range(z_mm, size)
    for size in (5, 11):
        features[f"z_max_diff_{size}"] = _max_abs_diff_from_center(z_mm, size)
    grad_mag, surface_angle = _gradient_features(z_mm)
    features["grad_mag"] = grad_mag
    features["surface_angle"] = surface_angle
    features["laplacian"] = _laplacian(z_mm)
    features["sim_distance"] = sim_distance.astype(np.float32)
    features["lumi_value"] = lumi.astype(np.float32)
    features["lumi_std_5"] = _local_std(lumi, 5)
    features["radial_distance"] = _radial_distance(valid_mask)
    features["valid_neighbor_frac_5"] = _valid_neighbor_fraction(valid_mask, 5)
    return features


def features_to_array(features: dict[str, np.ndarray], valid_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Stack feature grids into the RF row layout.

    Args:
        features: Mapping from :func:`compute_features`.
        valid_mask: ``(H, W)`` boolean mask.

    Returns:
        ``(X, names)`` with ``X`` of shape ``(n_valid, n_features) float32``;
        columns follow ``sorted(names)`` — part of the model contract.
    """
    names = sorted(features.keys())
    x = np.column_stack([features[name][valid_mask] for name in names]).astype(np.float32)
    return x, names


def registered_columns(
    z_mm: np.ndarray,
    valid_mask: np.ndarray,
    prior_reg: np.ndarray,
    expected_reg: np.ndarray,
    xf: tuple[int, float, float],
    dx_mm: float,
    dy_mm: float,
) -> np.ndarray:
    """The four registered-frame feature columns for one sample.

    ``dd_dev`` is the slope-normalized deviation from the consensus surface.
    Axis 0 of the grids is y, so the gradient spacings are ``(dy_mm, dx_mm)``.

    Args:
        z_mm: ``(H, W)`` aligned z in mm.
        valid_mask: ``(H, W)`` boolean mask.
        prior_reg: Reference-frame position prior P(outlier | position).
        expected_reg: Reference-frame consensus surface.
        xf: Registration ``(theta, dy, dx)`` of this sample (see
            :func:`.registration.register`).
        dx_mm: Pixel spacing along x (columns).
        dy_mm: Pixel spacing along y (rows).

    Returns:
        ``(n_valid, 4) float32`` — columns ``prior, dd_dev, reg_x, reg_y``.
    """
    theta, dy, dx = xf
    prior = from_reference(prior_reg, theta, dy, dx)
    expected = from_reference(expected_reg, theta, dy, dx)
    gy, gx = np.gradient(expected, dy_mm, dx_mm)
    norm = np.sqrt(1.0 + gx**2 + gy**2)
    dd = np.abs(np.nan_to_num(z_mm, nan=0.0) - expected) / norm
    height, width = valid_mask.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    reg_y = from_reference(yy, theta, dy, dx)
    reg_x = from_reference(xx, theta, dy, dx)
    return np.column_stack([prior[valid_mask], dd[valid_mask], reg_x[valid_mask], reg_y[valid_mask]]).astype(np.float32)
