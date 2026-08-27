"""End-to-end test of the pointcloud stage on a synthetic dome-with-fin dataset.

SimContext, the bundled labels, and the label fingerprint are monkeypatched so
the full chain — preflight retrain, per-experiment processing, output
packaging — runs in seconds without the real dataset or simulations.
"""

import numpy as np
import pandas as pd
import pytest

from rddac._preprocess.pointcloud import classifier, stage, training
from rddac._preprocess.runner import run

H, W = 60, 80
Z_UNIT = 0.0077545  # matches calibration.json so sensor units are realistic
FIN = (slice(10, 22), slice(60, 72))  # the labeled artifact region
EXPERIMENT_IDS = (1, 2, 3)


def _z_grid_mm(exp_id: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dome = 20.0 - ((yy - H / 2) ** 2 / H + (xx - W / 2) ** 2 / W) + 0.01 * exp_id
    dome[FIN] += 7.0  # the fin: lifted patch every experiment at the same spot
    return dome


def _sim_points() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dome = 20.0 - ((yy - H / 2) ** 2 / H + (xx - W / 2) ** 2 / W)
    pts = np.column_stack([xx.ravel(), yy.ravel(), dome.ravel()])
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 1] -= pts[:, 1].mean()
    pts[:, 2] -= pts[:, 2].mean()
    return pts


class _FakeSim:
    def __init__(self, data_dir):
        self.sim_dir = None

    def available(self):
        return True

    def match(self, geometry, blankholder_force_kn, sheet_um, oil_gm2):
        return {
            "simulation_id": 4242,
            "matched_shtk": 1.0,
            "matched_fc": 0.1,
            "target_shtk": sheet_um / 1000.0,
            "target_fc": 0.1,
            "error_shtk": 0.0,
            "error_fc": 0.0,
        }

    def points(self, simulation_id, op):
        return _sim_points()


@pytest.fixture()
def pc_env(tmp_path, monkeypatch):
    """Synthetic raw dir + patched sim access, labels, and groups."""
    raw = tmp_path / "raw"
    raw.mkdir()
    import h5py

    for exp_id in EXPERIMENT_IDS:
        with h5py.File(raw / f"{exp_id:04d}.h5", "w") as f:
            f.attrs["geometry"] = "concave"
            f.attrs["blankholder_force"] = 100
            z_units = (_z_grid_mm(exp_id) / Z_UNIT).astype(np.float32)
            lumi = np.full((H, W), 500.0, dtype=np.float32)
            for op in ("op10", "op20"):
                grp = f.create_group(f"pointcloud/{op}")
                grp.create_dataset("z", data=z_units.ravel())
                grp.create_dataset("luminescence", data=lumi.ravel())
                grp.attrs["y_shape"] = H
                grp.attrs["x_shape"] = W
            pos = np.arange(0, 210, dtype=float)
            f.create_group("oil_thickness").create_dataset(
                "data", data=np.column_stack([pos, np.full_like(pos, 1.2)]).astype(np.float32)
            )
            spos = np.arange(10, 10 + 0.5 * 208, 0.5)
            f.create_group("sheet_thickness").create_dataset(
                "data", data=np.column_stack([spos, np.full(208, 995.5)]).astype(np.float32)
            )
    pd.DataFrame(
        {
            "index": list(EXPERIMENT_IDS),
            "geometry": ["concave"] * 3,
            "blankholder_force": [100] * 3,
            "split": ["train"] * 3,
        }
    ).to_csv(raw / "process_parameters.csv", index=False)

    label = np.zeros((H, W), dtype=bool)
    label[FIN] = True
    tasks = [(f"{i:04d}_{op}", i, op) for i in EXPERIMENT_IDS for op in ("op10", "op20")]

    monkeypatch.setattr(training, "labeled_tasks", lambda: tasks)
    monkeypatch.setattr(training, "load_label", lambda name: label)
    monkeypatch.setattr(training, "GROUPS", ("concave_op10", "concave_op20"))
    monkeypatch.setattr(classifier, "labels_sha256", lambda: "synthetic-test")
    monkeypatch.setattr(training, "SimContext", _FakeSim)
    monkeypatch.setattr(stage, "SimContext", _FakeSim)
    stage._CTX.clear()

    cfg = {
        "pointcloud": {
            "lumi_min_patch_size": 10,
            "min_component_size": 5,
            "icp_sample_size": 2000,
            "icp_max_iterations": 10,
            "k_angle": 8,
            "k_mono": 10,
            "rf_n_estimators": 10,
            "rf_max_depth": 6,
        }
    }
    return raw, tmp_path / "out", cfg


class TestStageEndToEnd:
    """Full pointcloud chain: retrain in preflight, process, package."""

    def test_run_trains_processes_and_packages(self, pc_env):
        raw, out, cfg = pc_env
        stats = run(["pointcloud"], data_dir=str(raw), out_dir=str(out), quiet=True, config=cfg)
        assert stats["pointcloud"] == {"processed": 3}

        import h5py

        with h5py.File(out / "0001.h5", "r") as f:
            group = f["pointcloud"]
            assert group.attrs["simulation_id"] == 4242
            for op in ("op10", "op20"):
                z = group[f"{op}/z"][:]
                lumi = group[f"{op}/luminescence"][:]
                assert z.dtype == np.float32 and z.ndim == 2 and z.shape[1] == 3
                assert lumi.dtype == np.uint8 and lumi.shape == (H, W)
                attrs = group[op].attrs
                for key in ("icp_rotation", "icp_translation", "n_rf_removed", "n_final_points", "rf_threshold"):
                    assert key in attrs, key
                # the fin must be substantially removed
                assert attrs["n_final_points"] < H * W, "some points must be removed"
                assert attrs["n_geometric_outliers"] + attrs["n_rf_removed"] > 0

        # model cache written and reused on a second run
        assert (classifier.cache_dir(out) / "meta.json").is_file()
        assert not (raw / "models").exists(), "the raw directory is never written to"
        stats2 = run(["pointcloud"], data_dir=str(raw), out_dir=str(out), quiet=True, config=cfg)
        assert stats2["pointcloud"] == {"exists": 3}

    def test_fin_region_is_removed(self, pc_env):
        raw, out, cfg = pc_env
        run(["pointcloud"], data_dir=str(raw), out_dir=str(out), ids="1", quiet=True, config=cfg)
        import h5py

        with h5py.File(out / "0001.h5", "r") as f:
            z = f["pointcloud/op10/z"][:]
        # points surviving near the fin's z level (dome top ~20 + 7 lift) must be rare
        assert (z[:, 2] > z[:, 2].mean() + 5.0).mean() < 0.02
