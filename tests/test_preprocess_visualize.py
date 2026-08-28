"""Tests for the processing figures (rddac._preprocess.visualize)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rddac._preprocess import visualize


def _oil():
    pos = np.arange(0, 210, dtype=float)
    val = np.round(1.2 + 0.2 * np.sin(pos / 30), 2)
    val[100] = np.nan
    val[103] = 3.5
    return np.column_stack([pos, val])


def _force():
    t = np.arange(0, 3.8, 1 / 300)
    lc = 23.0 + 100.0 * np.exp(-((t - 1.2) ** 2))
    return np.column_stack([t, lc, lc, lc, lc, np.full(len(t), 22.3), np.linspace(448, 289, len(t)), 4 * lc])


def _sheet():
    pos = np.linspace(5.0, 60.0, 260)
    thick = np.full(260, 1000.0)
    thick[10:15] = -9999.0
    return np.column_stack([pos, thick])


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestFigures:
    """Each function returns a figure with the documented panel layout."""

    def test_oil_three_panels_with_outlier_marked(self):
        fig = visualize.plot_oil_processing(_oil())
        assert len(fig.axes) == 3
        assert "1 outliers" in fig.axes[1].get_title()

    def test_force_two_panels_with_window_lines(self):
        fig = visualize.plot_force_processing([_force(), _force()])
        assert len(fig.axes) == 2
        assert len(fig.axes[0].lines) == 2 + 2, "two traces plus two window lines"
        assert len(fig.axes[1].lines) == 2

    def test_sheet_two_panels_with_cutoff(self):
        fig = visualize.plot_sheet_processing([_sheet()])
        assert len(fig.axes) == 2
        assert len(fig.axes[0].lines) == 1 + 1, "trace plus cutoff line"

    def test_luminescence_four_panels(self):
        pytest.importorskip("scipy", reason="needs the rddac[preprocessing] extra")
        lumi = np.zeros((40, 60))
        lumi[5:35, 5:45] = 500.0  # large patch (kept)
        lumi[2:4, 50:52] = 300.0  # tiny reflection (removed)
        fig = visualize.plot_luminescence_processing(lumi, lumi_min_patch_size=100)
        assert len(fig.axes) >= 4
        assert "(2)" in fig.axes[1].get_title()
        assert "1 kept, 1 removed" in fig.axes[2].get_title()

    def test_pointcloud_two_panels_with_stats(self):
        pytest.importorskip("scipy", reason="needs the rddac[preprocessing] extra")
        lumi = np.zeros((40, 60))
        lumi[5:35, 5:45] = 500.0
        z = np.where(lumi > 0, 1000.0, 0.0)
        processed = np.random.default_rng(0).normal(size=(50, 3))
        fig = visualize.plot_pointcloud_processing(
            z, lumi, processed, stats={"outlier_pct": 12.3}, lumi_min_patch_size=100
        )
        with pytest.raises(ValueError, match="no valid pixels"):
            visualize.plot_pointcloud_processing(z, lumi, processed)  # default 20040-px threshold
        assert "12.3 % removed" in fig.axes[1].get_title()
        assert "1,200 points" in fig.axes[0].get_title()
        sim = np.random.default_rng(1).normal(size=(80, 3))
        fig = visualize.plot_pointcloud_processing(
            z, lumi, processed, sim_points=sim, stats={"simulation_id": 4242}, lumi_min_patch_size=100
        )
        titles = [ax.get_title() for ax in fig.axes]
        assert any("Matched simulation 4242" in t for t in titles)
        assert any("Distance to simulation" in t and "median" in t for t in titles)

    def test_params_are_forwarded(self):
        fig = visualize.plot_force_processing([_force()], time_window_start=0.5, time_window_end=1.5)
        assert fig.axes[1].get_xlim()[1] == pytest.approx(1.0)


class TestCli:
    """The maintainer entry point writes one image per invocation."""

    def test_writes_image_for_loose_h5(self, tmp_path):
        import h5py

        for i in (1, 2):
            with h5py.File(tmp_path / f"{i:04d}.h5", "w") as f:
                f.create_dataset("sheet_thickness/data", data=_sheet())
        out = tmp_path / "img"
        visualize.main(["sheet", "--data-dir", str(tmp_path), "--ids", "1-2", "--out", str(out)])
        assert (out / "sheet_processing.png").is_file()

    def test_pointcloud_needs_processed_file(self, tmp_path):
        pytest.importorskip("scipy", reason="needs the rddac[preprocessing] extra")
        import h5py

        lumi = np.zeros((40, 60), dtype=np.float32)
        lumi[5:35, 5:45] = 500.0
        with h5py.File(tmp_path / "0001.h5", "w") as f:
            g = f.create_group("pointcloud/op10")
            g.attrs["y_shape"], g.attrs["x_shape"] = 40, 60
            g.create_dataset("z", data=np.where(lumi > 0, 1000.0, 0.0).ravel())
            g.create_dataset("luminescence", data=lumi.ravel())
        out = tmp_path / "img"
        visualize.main(["luminescence", "--data-dir", str(tmp_path), "--id", "1", "--out", str(out)])
        assert (out / "luminescence_processing_scan_op10.png").is_file()
        with pytest.raises(SystemExit, match="processed file not found"):
            visualize.main(["pointcloud", "--data-dir", str(tmp_path), "--id", "1", "--out", str(out)])
