"""Tests for `rddac.open_h5` and `rddac.inspect_h5`."""

from __future__ import annotations

import zipfile

import h5py
import numpy as np
import pytest

import rddac

from .conftest import REAL_EXPERIMENT_IDS, X_SHAPE, Y_SHAPE


class TestOpenH5:
    def test_returns_h5py_file(self, synthetic_data_dir):
        with rddac.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            assert isinstance(f, h5py.File)
            assert "force/data" in f
            assert "pointcloud/op10/z" in f

    def test_zero_padded_member_resolution(self, synthetic_data_dir):
        """Experiment 1 lives as `0001.h5` inside the zip (RDDAC_SPEC.id_format)."""
        zip_path = synthetic_data_dir / "concave.zip"
        with zipfile.ZipFile(zip_path) as zf:
            assert "0001.h5" in zf.namelist()
            assert "1.h5" not in zf.namelist()
        # open_h5 takes the *unpadded* integer id.
        with rddac.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            assert f.attrs["id"] == 1

    def test_attrs_round_trip(self, synthetic_data_dir):
        with rddac.open_h5(2, data_dir=str(synthetic_data_dir)) as f:
            assert f.attrs["id"] == 2
            assert f.attrs["geometry"] == "concave"
            assert f.attrs["oil_type"] == "coarse"
            assert bool(f.attrs["has_pointcloud"]) is True

    def test_rddac_raw_structure(self, synthetic_data_dir):
        """The synthetic files model the RDDAC raw h5 layout."""
        with rddac.open_h5(3, data_dir=str(synthetic_data_dir)) as f:
            assert f["force/data"].shape[1] == 8
            assert f["force/data"].dtype == np.float32
            assert f["sheet_thickness/data"].shape[1] == 2
            assert f["oil_thickness/data"].shape[1] == 2
            for op in ("op10", "op20"):
                g = f[f"pointcloud/{op}"]
                assert g.attrs["x_shape"] == X_SHAPE
                assert g.attrs["y_shape"] == Y_SHAPE
                assert g["z"].shape == (X_SHAPE * Y_SHAPE,)
                assert g["luminescence"].shape == (X_SHAPE * Y_SHAPE,)

    def test_read_only_mode(self, synthetic_data_dir):
        with rddac.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            assert f.mode == "r"

    def test_missing_experiment_raises_filenotfound(self, synthetic_data_dir):
        with pytest.raises(FileNotFoundError):
            rddac.open_h5(99999, data_dir=str(synthetic_data_dir))

    def test_iterating_multiple_experiments(self, synthetic_data_dir):
        for exp_id in (1, 2, 3):
            with rddac.open_h5(exp_id, data_dir=str(synthetic_data_dir)) as f:
                assert f.attrs["id"] == exp_id

    def test_dataset_kwarg_skips_reparse(self, synthetic_data_dir):
        """A pre-loaded dataset is used as-is; source/data_dir are ignored."""
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        with rddac.open_h5(1, dataset=ds) as f:
            assert f.attrs["id"] == 1


class TestInspectH5:
    def test_accepts_open_file(self, synthetic_data_dir, capsys):
        with rddac.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            rddac.inspect_h5(f)
        out = capsys.readouterr().out
        assert "pointcloud/" in out
        assert "force/" in out
        assert "@id = 1" in out

    def test_accepts_path(self, synthetic_data_dir, tmp_path, capsys):
        # Extract one h5 to disk to test the path variant.
        zip_path = synthetic_data_dir / "concave.zip"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        rddac.inspect_h5(tmp_path / "0001.h5")
        out = capsys.readouterr().out
        assert "pointcloud/" in out

    def test_tree_characters_present(self, synthetic_data_dir, capsys):
        with rddac.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            rddac.inspect_h5(f)
        out = capsys.readouterr().out
        assert "├──" in out or "└──" in out

    def test_dataset_shapes_printed(self, synthetic_data_dir, capsys):
        with rddac.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            rddac.inspect_h5(f)
        out = capsys.readouterr().out
        assert f"({X_SHAPE * Y_SHAPE},)" in out  # flat scan buffers


class TestOpenH5RealData:
    """Guarded real-data checks — only a couple of experiment ids, never all 9000."""

    @pytest.mark.parametrize("exp_id", REAL_EXPERIMENT_IDS)
    def test_open_and_attrs(self, real_data_dir, exp_id):
        with rddac.open_h5(exp_id, data_dir=str(real_data_dir)) as f:
            assert int(f.attrs["id"]) == exp_id
            assert f.attrs["geometry"] in ("concave", "convex")
            assert "force/data" in f
            assert f["force/data"].shape[1] == 8

    def test_real_scan_grid_attrs(self, real_data_dir):
        with rddac.open_h5(REAL_EXPERIMENT_IDS[0], data_dir=str(real_data_dir)) as f:
            if not bool(f.attrs["has_pointcloud"]):
                pytest.skip("experiment has no pointcloud")
            g = f["pointcloud/op10"]
            assert g["z"].shape == (int(g.attrs["x_shape"]) * int(g.attrs["y_shape"]),)

    def test_inspect_real_file(self, real_data_dir, capsys):
        with rddac.open_h5(REAL_EXPERIMENT_IDS[1], data_dir=str(real_data_dir)) as f:
            rddac.inspect_h5(f)
        out = capsys.readouterr().out
        assert "force/" in out
        assert "sheet_thickness/" in out
