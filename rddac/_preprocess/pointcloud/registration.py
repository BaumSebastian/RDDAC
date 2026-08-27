"""Silhouette registration into a common frame for cross-sample features.

The RF fin cleaner uses features accumulated across labeled samples
(position-prior, consensus surface). Samples are registered by their
downsampled valid-mask silhouettes: a small rotation sweep plus centroid
shift, scored by IoU. Part placement across the dataset is within +-3 deg,
so the +-8 deg sweep always suffices.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import shift as ndshift


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    y_idx, x_idx = np.where(mask)
    return float(y_idx.mean()), float(x_idx.mean())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def register(valid_mask: np.ndarray, ref_ds: np.ndarray, factor: int = 4) -> tuple[int, float, float]:
    """Best ``(theta_deg, dy, dx)`` aligning a silhouette to the reference.

    Args:
        valid_mask: ``(H, W)`` boolean silhouette of the sample.
        ref_ds: Reference silhouette, already downsampled by ``factor``.
        factor: Downsampling stride used for the search.

    Returns:
        Rotation in degrees and full-resolution y/x shifts.
    """
    v_ds = valid_mask[::factor, ::factor]
    ref_cy, ref_cx = _centroid(ref_ds)
    best: tuple[float, int, float, float] | None = None
    for theta in range(-8, 9, 2):
        rotated = rotate(v_ds.astype(np.float32), theta, reshape=False, order=0) > 0.5
        if not rotated.any():
            continue
        cy, cx = _centroid(rotated)
        dy, dx = ref_cy - cy, ref_cx - cx
        shifted = ndshift(rotated.astype(np.float32), (dy, dx), order=0) > 0.5
        score = _iou(shifted, ref_ds)
        if best is None or score > best[0]:
            best = (score, theta, dy * factor, dx * factor)
    assert best is not None, "empty silhouette"
    return best[1], best[2], best[3]


def to_reference(grid: np.ndarray, theta: int, dy: float, dx: float, order: int = 0) -> np.ndarray:
    """Map a full-resolution grid into the reference frame (rotate, then shift)."""
    return ndshift(rotate(grid.astype(np.float64), theta, reshape=False, order=order), (dy, dx), order=order)


def from_reference(grid: np.ndarray, theta: int, dy: float, dx: float) -> np.ndarray:
    """Map a reference-frame grid back into the sample frame (inverse of :func:`to_reference`)."""
    return rotate(ndshift(grid, (-dy, -dx), order=1), -theta, reshape=False, order=1)
