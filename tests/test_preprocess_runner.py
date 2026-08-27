"""Tests for the preprocessing runner (rddac._preprocess.runner)."""

import h5py
import numpy as np
import pandas as pd
import pytest

from rddac._preprocess import config, force, oil
from rddac._preprocess.runner import PROCESSED_MARKER, parse_ids, run


def _oil_trace():
    pos = np.arange(0, 210, dtype=float)
    return pos, np.round(1.2 + 0.2 * np.sin(pos / 30), 2)


def _force_table():
    t = np.arange(0, 3.8, 1 / 300)
    n = len(t)
    lc = 23.0 + 100.0 * np.exp(-((t - 1.2) ** 2))
    return np.column_stack([t, lc, lc, lc, lc, np.full(n, 22.34), np.linspace(448.66, 289.56, n), 4 * lc]).astype(
        np.float32
    )


def _make_raw_dir(path, ids=(0, 1)):
    for i in ids:
        with h5py.File(path / f"{i:04d}.h5", "w") as f:
            f.attrs["id"] = i
            f.attrs["geometry"] = "concave"
            f.attrs["n_force_measurements"] = 1141  # obsolete attr, must be dropped
            g = f.create_group("force")
            g.create_dataset("data", data=_force_table())
            g.attrs["columns"] = list(force.COLUMNS)
            pos, val = _oil_trace()
            f.create_group("oil_thickness").create_dataset("data", data=np.column_stack([pos, val]).astype(np.float32))
            spos = np.arange(10, 10 + 0.5 * 208, 0.5)
            f.create_group("sheet_thickness").create_dataset(
                "data", data=np.column_stack([spos, np.full(208, 995.5)]).astype(np.float32)
            )
    pd.DataFrame({"index": list(ids), "split": ["train"] * len(ids)}).to_csv(
        path / "process_parameters.csv", index=False
    )


class TestParseIds:
    def test_ranges_and_singles(self):
        assert parse_ids("0-3,7") == {0, 1, 2, 3, 7}
        assert parse_ids("42") == {42}


class TestRunner:
    """End-to-end over a synthetic loose-h5 raw directory."""

    def test_end_to_end_and_rerun_skips(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _make_raw_dir(raw)
        out = tmp_path / "processed"

        stats = run(["force", "sheet", "oil"], data_dir=str(raw), out_dir=str(out), quiet=True)
        assert stats["force"] == {"processed": 2}
        assert (out / PROCESSED_MARKER).is_file()

        with h5py.File(out / "0000.h5", "r") as f:
            assert f["force/data"].shape == (600, 8)
            assert f["oil_thickness/data"].shape == (200, 2)
            assert f["sheet_thickness/data"].shape == (200, 2)
            assert list(f["oil_thickness"].attrs["columns"]) == list(oil.COLUMNS)
            assert "n_hampel_outliers" in f["oil_thickness"].attrs
            assert "n_force_measurements" not in f.attrs, "obsolete root attr dropped"

        stats = run(["force", "sheet", "oil"], data_dir=str(raw), out_dir=str(out), quiet=True)
        assert stats["force"] == {"exists": 2}

    def test_selection_limited_to_locally_available_files(self, tmp_path):
        """A full process_parameters.csv with a partial bundle must not try every id."""
        import zipfile

        import pandas as pd

        from rddac._preprocess.runner import _experiment_ids

        pd.DataFrame({"index": range(50), "split": ["train"] * 50}).to_csv(
            tmp_path / "process_parameters.csv", index=False
        )
        (tmp_path / "h5").mkdir()
        with zipfile.ZipFile(tmp_path / "h5" / "sample.zip", "w") as zf:
            zf.writestr("0007.h5", b"")
            zf.writestr("0042.h5", b"")
        (tmp_path / "0003.h5").write_bytes(b"")
        assert _experiment_ids(str(tmp_path), None, None) == [3, 7, 42]
        assert _experiment_ids(str(tmp_path), "0-10", "train") == [3, 7]

    def test_missing_simulations_skip_or_fail(self, tmp_path):
        """Pointcloud without simulations: skipped when not named, error when named."""
        pytest.importorskip("sklearn", reason="needs the rddac[preprocessing] extra")
        from rddac._preprocess.runner import run

        (tmp_path / "0001.h5").write_bytes(b"")
        stats = run(["pointcloud"], str(tmp_path), str(tmp_path / "out"), quiet=True, skip_unavailable=True)
        assert stats == {"elapsed_s": 0.0}
        with pytest.raises(FileNotFoundError, match="simulations not found"):
            run(["pointcloud"], str(tmp_path), str(tmp_path / "out"), quiet=True, skip_unavailable=False)

    def test_no_manifest_generated(self, tmp_path):
        """The published Croissant manifest is the single source of truth; none is written."""
        raw = tmp_path / "raw"
        raw.mkdir()
        _make_raw_dir(raw, ids=(0,))
        out = tmp_path / "processed"
        run(["oil"], data_dir=str(raw), out_dir=str(out), quiet=True)
        assert not (out / "metadata.json").exists()
        assert sorted(p.name for p in out.iterdir()) == [PROCESSED_MARKER, "0000.h5"]

    def test_ids_and_split_selection(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _make_raw_dir(raw, ids=(0, 1, 2))
        out = tmp_path / "processed"
        stats = run(["oil"], data_dir=str(raw), out_dir=str(out), ids="1-2", quiet=True)
        assert stats["oil"] == {"processed": 2}
        assert not (out / "0000.h5").exists()

    def test_config_overrides_reach_processing_and_are_stamped(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _make_raw_dir(raw, ids=(0,))
        out = tmp_path / "processed"
        cfg = {"oil": {"max_sensor_position": 100, "output_length": 100}}
        run(["oil"], data_dir=str(raw), out_dir=str(out), quiet=True, config=cfg)
        with h5py.File(out / "0000.h5", "r") as f:
            assert f["oil_thickness/data"].shape == (100, 2)
            assert f["oil_thickness"].attrs["max_sensor_position"] == 100
            assert f["oil_thickness"].attrs["hampel_k"] == config.DEFAULTS["oil"]["hampel_k"]
