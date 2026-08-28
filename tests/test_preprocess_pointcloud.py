"""Tests for the pointcloud preprocessing stages (rddac._preprocess.pointcloud).

Everything here runs on small synthetic grids — no real dataset needed.
"""

import numpy as np
import pytest

scipy = pytest.importorskip("scipy", reason="needs the rddac[preprocessing] extra")
pytest.importorskip("sklearn", reason="needs the rddac[preprocessing] extra")
from scipy.ndimage import rotate as nd_rotate  # noqa: E402
from scipy.ndimage import shift as nd_shift  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from rddac._preprocess.pointcloud import features, geometry  # noqa: E402
from rddac._preprocess.pointcloud import registration as reg  # noqa: E402


def _dome(height=60, width=80, peak=20.0):
    """Smooth dome z-grid (mm) over a full valid mask."""
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    cy, cx = height / 2, width / 2
    z = peak - ((yy - cy) ** 2 / height + (xx - cx) ** 2 / width)
    return z


def _dome_points(z_scale=1.0):
    z = _dome() * z_scale
    yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]].astype(float)
    return np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])


class TestGeometrySeeds:
    """The three outlier seed stages on constructed shapes."""

    def test_angle_seeds_fire_on_vertical_fin_not_on_dome(self):
        pts = _dome_points()
        fin = pts[:40].copy()
        fin[:, 2] += 8.0  # a lifted patch creates near-vertical walls to its neighbours
        all_pts = np.vstack([pts, fin])
        tree = cKDTree(all_pts[:, :2])
        seeds = geometry.seed_angle_outliers(all_pts, tree, k=8, max_wall_angle_deg=70.0)
        assert seeds[len(pts) :].any(), "fin points must be seeded"
        interior = seeds[: len(pts)]
        assert interior.mean() < 0.1, "smooth dome must stay mostly unseeded"

    def test_monotonicity_seeds_fire_on_outward_rise(self):
        pts = _dome_points()
        radius = np.hypot(pts[:, 0] - 40, pts[:, 1] - 30)
        ring = radius > 25
        raised = pts.copy()
        raised[ring, 2] += 5.0  # z rises moving outward: physically impossible
        tree = cKDTree(raised[:, :2])
        seeds = geometry.seed_radial_monotonicity(raised, tree, k=12, z_tolerance=1.0)
        # The stage seeds the BOUNDARY of the rise (interior ring points have
        # equally-raised inward neighbours); closing grows it later.
        edge = ring & (radius < 27.5)
        assert seeds[edge].mean() > 0.2
        assert seeds[~ring].mean() < 0.05

    def test_small_component_seeds(self):
        pts = _dome_points()
        floater = np.column_stack([np.linspace(0, 3, 20), np.linspace(0, 3, 20), np.full(20, 120.0)])
        all_pts = np.vstack([pts, floater])
        seeds = geometry.seed_small_components(all_pts, min_component_size=50)
        assert seeds[len(pts) :].all(), "floating cluster below min size must be seeded"
        assert not seeds[: len(pts)].any()

    def test_closing_terminates_and_returns_bool(self):
        pts = _dome_points()
        seed = np.zeros(len(pts), dtype=bool)
        seed[100] = True
        tree = cKDTree(pts[:, :2])
        mask, iterations = geometry.morphological_closing(seed, tree, pts[:, :2])
        assert mask.dtype == bool and 1 <= iterations <= 15


class TestIcp:
    """Seeded, reproducible rigid alignment."""

    def test_sublattice_translation_recovered(self):
        # NN-correspondence ICP on a unit-spaced grid locks to the lattice for
        # large shifts; production spacing is 0.077 mm, so the realistic case
        # is a sub-lattice offset — that one must be recovered tightly.
        source = _dome_points()
        shift = np.array([0.3, -0.2, 0.5])
        r, t, stats = geometry.run_icp(source, source + shift, max_iterations=30, n_sample=2000)
        assert stats["icp_median_distance"] < 0.1
        assert np.abs(t - shift).max() < 0.1
        assert np.abs(r - np.eye(3)).max() < 5e-3

    def test_rotation_misalignment_reduced(self):
        source = _dome_points()
        source[:, 2] += 6.0 * ((np.abs(source[:, 0] - 15) < 6) & (np.abs(source[:, 1] - 12) < 6))
        angle = np.radians(4.0)
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
        target = source @ rot.T + np.array([2.0, -1.5, 0.7])
        initial = float(np.median(np.linalg.norm(source - target, axis=1)))
        r, t, stats = geometry.run_icp(source, target, max_iterations=50, n_sample=2000)
        assert stats["icp_median_distance"] < 0.5 * initial, "must reduce the misalignment substantially"
        assert np.abs(np.linalg.det(r) - 1.0) < 1e-6, "proper rotation"

    def test_deterministic(self):
        source = _dome_points()
        target = source + np.array([1.0, 0.0, 0.0])
        first = geometry.run_icp(source, target, max_iterations=5, n_sample=500)
        second = geometry.run_icp(source, target, max_iterations=5, n_sample=500)
        assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


class TestRegistration:
    """Silhouette registration and grid mapping."""

    def test_recovers_rotation_and_shift(self):
        mask = np.zeros((200, 240), dtype=bool)
        mask[40:150, 60:180] = True
        mask[40:70, 60:100] = False  # asymmetry so rotation is observable
        moved = nd_shift(nd_rotate(mask.astype(float), 4.0, reshape=False, order=0), (12, -8), order=0) > 0.5
        theta, dy, dx = reg.register(moved, mask[::4, ::4])
        assert theta == pytest.approx(-4, abs=2)
        assert dy == pytest.approx(-12, abs=6) and dx == pytest.approx(8, abs=6)

    def test_roundtrip_identity(self):
        grid = _dome()
        back = reg.from_reference(reg.to_reference(grid, 0, 0.0, 0.0, order=1), 0, 0.0, 0.0)
        inner = (slice(5, -5), slice(5, -5))
        assert np.abs(back[inner] - grid[inner]).max() < 1e-6


class TestFeatures:
    """Feature schema and the gradient-order regression guard."""

    def test_feature_matrix_schema(self):
        z = _dome()
        valid = np.ones_like(z, dtype=bool)
        sim = np.abs(z) * 0.1
        lumi = np.full_like(z, 128.0)
        base = features.compute_features(z, sim, lumi, valid)
        x, names = features.features_to_array(base, valid)
        assert x.shape == (valid.sum(), len(names))
        assert names == sorted(names), "column order is sorted names (model contract)"
        assert x.dtype == np.float32

    def test_gradient_spacing_order_fixed(self):
        # Consensus ramp along y with slope s (mm/mm). With correct spacings
        # (dy, dx) the slope-normalized deviation of z = expected + 1 equals
        # 1/sqrt(1+s^2); the historical swapped order gives a different value.
        height, width = 60, 80
        dy_mm, dx_mm = 0.1589, 0.0769
        slope = 2.0
        yy = np.mgrid[0:height, 0:width][0].astype(float)
        expected = slope * yy * dy_mm
        z = expected + 1.0
        valid = np.ones_like(z, dtype=bool)
        cols = features.registered_columns(z, valid, np.zeros_like(z), expected, (0, 0.0, 0.0), dx_mm, dy_mm)
        dd = cols[:, 1].reshape(height, width)[5:-5, 5:-5]
        correct = 1.0 / np.sqrt(1.0 + slope**2)
        swapped = 1.0 / np.sqrt(1.0 + (slope * dy_mm / dx_mm) ** 2)
        assert np.abs(dd - correct).max() < 1e-3
        assert abs(correct - swapped) > 0.1, "test must discriminate the two orders"

    def test_kd_sim_distance_grid(self):
        pts = _dome_points()
        valid = np.ones((60, 80), dtype=bool)
        grid = features.kd_sim_distance_grid(pts, valid, pts + np.array([0, 0, 2.0]))
        assert grid.shape == (60, 80)
        assert np.nanmax(grid) <= 2.0 + 1e-6


class TestPacking:
    """Processed luminescence packaging."""

    def test_uint8_range_and_background(self):
        lumi = np.linspace(100, 900, 60 * 80).reshape(60, 80)
        valid = np.zeros((60, 80), dtype=bool)
        valid[10:50, 10:70] = True
        packed = geometry.pack_luminescence(lumi, valid)
        assert packed.dtype == np.uint8
        assert packed[~valid].max() == 0
        assert packed[valid].min() >= 1 and packed[valid].max() == 255


class TestAlignToSimulation:
    """Two-pass, cup-anchored alignment returns a consistent composed transform."""

    def _cup(self, n=40000, seed=0):
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-100, 100, size=(n, 2))
        r = np.abs(xy).max(axis=1)
        z = np.where(r < 50, 30.0, np.where(r < 60, 30.0 - 3.0 * (r - 50), 0.0))  # bottom, walls, flange
        return np.column_stack([xy, z])

    def test_composed_transform_matches_output_and_anchor_excludes_flange(self):
        sim = self._cup(seed=1)
        angle = np.radians(1.5)
        rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        scan = self._cup(seed=2) @ rot.T + np.array([1.5, -0.8, 4.0])
        aligned, rotation, translation, stats = geometry.align_to_simulation(scan, sim, n_sample=10000)
        assert np.allclose(aligned, scan @ rotation.T + translation, atol=1e-6)
        assert stats["n_anchor_points"] > 0 and stats["n_anchor_points"] < len(scan), "flange excluded"
        d, _ = cKDTree(sim).query(aligned)
        assert np.median(d) < 0.5
