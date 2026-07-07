"""Tests for `rddac.streaming` — iter_view, export_to_numpy(_per_sim), load_export."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

import rddac

from .conftest import N_FORCE, REAL_EXPERIMENT_IDS, X_SHAPE, Y_SHAPE


class TestIterViewZippedLayout:
    """The default synthetic fixture stores zero-padded `.h5` members grouped
    into geometry zips (concave.zip / convex.zip), like the published dataset."""

    def test_yields_one_record_per_experiment(self, synthetic_data_dir):
        records = list(
            rddac.streaming.iter_view("force-curve", data_dir=str(synthetic_data_dir))
        )
        assert len(records) == 3
        for rec in records:
            # `_sim_id` is the private scratch key iter_view attaches to each record.
            assert set(rec.keys()) == {"force_data", "_sim_id"}
            assert rec["force_data"].shape == (N_FORCE, 8)

    def test_sim_ids_are_attached_in_csv_order(self, synthetic_data_dir):
        records = list(
            rddac.streaming.iter_view("force-curve", data_dir=str(synthetic_data_dir))
        )
        assert [rec["_sim_id"] for rec in records] == [1, 2, 3]

    def test_sim_ids_filter(self, synthetic_data_dir):
        records = list(
            rddac.streaming.iter_view(
                "force-curve",
                data_dir=str(synthetic_data_dir),
                sim_ids=[1, 3],
            )
        )
        assert [rec["_sim_id"] for rec in records] == [1, 3]

    def test_where_filter(self, synthetic_data_dir):
        records = list(
            rddac.streaming.iter_view(
                "force-curve",
                data_dir=str(synthetic_data_dir),
                where=lambda row: row["geometry"] == "concave",
            )
        )
        assert len(records) == 2

        records = list(
            rddac.streaming.iter_view(
                "force-curve",
                data_dir=str(synthetic_data_dir),
                where=lambda row: row["split"] == "test",
            )
        )
        assert len(records) == 1

    def test_where_combined_with_sim_ids(self, synthetic_data_dir):
        records = list(
            rddac.streaming.iter_view(
                "force-curve",
                data_dir=str(synthetic_data_dir),
                sim_ids=[2, 3],
                where=lambda row: row["geometry"] == "concave",
            )
        )
        assert [rec["_sim_id"] for rec in records] == [2]


class TestIterViewLooseLayout:
    """`rddac download --extract --remove-zip` produces loose `.h5` files."""

    @pytest.fixture
    def loose_data_dir(self, synthetic_data_dir, tmp_path):
        """Mirror the synthetic dataset with the h5 files extracted from their zips."""
        out = tmp_path / "rddac_loose"
        (out / "h5").mkdir(parents=True)
        # Copy manifest + csv unchanged.
        (out / "metadata.json").write_text((synthetic_data_dir / "metadata.json").read_text())
        (out / "process_parameters.csv").write_text(
            (synthetic_data_dir / "process_parameters.csv").read_text()
        )
        # Unpack each zip into h5/<0-padded id>.h5.
        for zp in synthetic_data_dir.glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(out / "h5")
        return out

    def test_loose_files_are_zero_padded(self, loose_data_dir):
        names = sorted(p.name for p in (loose_data_dir / "h5").glob("*.h5"))
        assert names == ["0001.h5", "0002.h5", "0003.h5"]

    def test_yields_one_record_per_experiment(self, loose_data_dir):
        records = list(
            rddac.streaming.iter_view("force-curve", data_dir=str(loose_data_dir))
        )
        assert len(records) == 3
        for rec in records:
            assert rec["force_data"].shape == (N_FORCE, 8)

    def test_index_prefers_loose_over_zip(self, loose_data_dir, synthetic_data_dir):
        """When both layouts exist side by side, loose files win and zips are ignored."""
        # Copy the original zips into the loose dir so both formats coexist.
        for zp in synthetic_data_dir.glob("*.zip"):
            (loose_data_dir / "h5" / zp.name).write_bytes(zp.read_bytes())

        index = rddac.streaming._build_unified_index(loose_data_dir)
        assert set(index) == {1, 2, 3}
        # Every indexed experiment should resolve to a `.h5` file, not a `.zip`.
        for sim_id, path in index.items():
            assert path.endswith(
                ".h5"
            ), f"experiment {sim_id} resolved to {path!r}; loose layout should win"

    def test_index_parses_zero_padded_stems(self, synthetic_data_dir):
        """`0001.h5` must index as experiment 1 (int of the padded stem)."""
        index = rddac.streaming._build_unified_index(synthetic_data_dir)
        assert set(index) == {1, 2, 3}
        for path in index.values():
            assert path.endswith(".zip")


class TestIterViewDatasetKwarg:
    """`add_view` -> `iter_view(dataset=ds)` is the non-PyTorch equivalent of
    `RDDACDataset(dataset=ds)`."""

    def test_custom_view_via_dataset_kwarg(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(
            ds,
            "scan-only",
            fields={"scan": ("pointcloud_op10_z", None)},
        )

        # Without dataset=, the manifest is re-parsed and the custom view is invisible.
        with pytest.raises(ValueError):
            list(rddac.streaming.iter_view("scan-only", data_dir=str(synthetic_data_dir)))

        # With dataset=, the in-memory mutation carries through.
        records = list(
            rddac.streaming.iter_view(
                "scan-only",
                data_dir=str(synthetic_data_dir),
                dataset=ds,
            )
        )
        assert len(records) == 3
        for rec in records:
            assert rec["scan"].shape == (X_SHAPE * Y_SHAPE,)  # flat scan buffer

    def test_metadata_columns_joined_from_csv(self, synthetic_data_dir):
        """Views mixing field-map and process-parameters sources stream both."""
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(ds, "force-signals", fields={
            "force": "force_data",
            "geometry": "process-parameters/geometry",
            "blankholder_force": "process-parameters/blankholder_force",
            "split": "process-parameters/split",
        })
        records = list(
            rddac.streaming.iter_view(
                "force-signals", data_dir=str(synthetic_data_dir), dataset=ds
            )
        )
        assert len(records) == 3
        by_id = {rec["_sim_id"]: rec for rec in records}
        assert by_id[1]["geometry"] == "concave"
        assert by_id[3]["geometry"] == "convex"
        assert int(by_id[3]["blankholder_force"]) == 150
        assert by_id[2]["split"] == "val"
        assert by_id[1]["force"].shape == (N_FORCE, 8)

    def test_all_modalities_view_streams_every_modality(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(ds, "full-experiment", fields={
            "force": "force_data",
            "sheet_thickness": "sheet_thickness_data",
            "oil_thickness": "oil_thickness_data",
            "op10_z": "pointcloud_op10_z",
            "op10_luminescence": "pointcloud_op10_luminescence",
            "op20_z": "pointcloud_op20_z",
            "op20_luminescence": "pointcloud_op20_luminescence",
        })
        rec = next(
            iter(
                rddac.streaming.iter_view(
                    "full-experiment",
                    data_dir=str(synthetic_data_dir),
                    dataset=ds,
                    sim_ids=[1],
                )
            )
        )
        assert rec["force"].shape == (N_FORCE, 8)
        assert rec["sheet_thickness"].shape[1] == 2
        assert rec["oil_thickness"].shape[1] == 2
        for key in ("op10_z", "op10_luminescence", "op20_z", "op20_luminescence"):
            assert rec[key].shape == (X_SHAPE * Y_SHAPE,)


class TestIterViewInvalidView:
    def test_unknown_view_raises(self, synthetic_data_dir):
        with pytest.raises(ValueError):
            list(rddac.streaming.iter_view("nonexistent", data_dir=str(synthetic_data_dir)))


class TestExportToNumpy:
    """`export_to_numpy` materialises a view as flat .npy memmap files."""

    def test_basic_round_trip(self, synthetic_data_dir, tmp_path):
        out = tmp_path / "shards"
        paths = rddac.streaming.export_to_numpy(
            "force-curve",
            out,
            data_dir=str(synthetic_data_dir),
        )
        # `_sim_id` is stripped before writing; `sim_ids` is the canonical shard.
        assert set(paths.keys()) == {"force_data", "sim_ids"}
        for p in paths.values():
            assert p.is_file()

        streamed = list(
            rddac.streaming.iter_view("force-curve", data_dir=str(synthetic_data_dir))
        )
        sim_ids = np.load(paths["sim_ids"])
        force = np.load(paths["force_data"], mmap_mode="r")
        assert force.shape == (len(streamed), *streamed[0]["force_data"].shape)
        for i, rec in enumerate(streamed):
            np.testing.assert_array_equal(force[i], rec["force_data"])
        assert sim_ids.dtype == np.int64
        assert sim_ids.tolist() == [rec["_sim_id"] for rec in streamed]

    def test_per_field_transforms(self, synthetic_data_dir, tmp_path):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(
            ds,
            "force-row",
            fields={"force_row": ("force_data", 2)},
        )
        out = tmp_path / "shards-tx"
        paths = rddac.streaming.export_to_numpy(
            "force-row",
            out,
            data_dir=str(synthetic_data_dir),
            dataset=ds,
            transforms={"force_row": lambda arr: arr.astype(np.float64)},
        )
        loaded = np.load(paths["force_row"], mmap_mode="r")
        assert loaded.dtype == np.float64
        assert loaded.shape == (3, 8)  # one force row per experiment

    def test_record_transform_combines_fields(self, synthetic_data_dir, tmp_path):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(
            ds,
            "combo-view",
            fields={
                "a": ("force_data", 2),
                "b": ("force_data", 3),
            },
        )
        out = tmp_path / "shards-combo"
        paths = rddac.streaming.export_to_numpy(
            "combo-view",
            out,
            data_dir=str(synthetic_data_dir),
            dataset=ds,
            record_transform=lambda rec: {"delta": rec["b"] - rec["a"]},
        )
        assert set(paths.keys()) == {"delta", "sim_ids"}
        delta = np.load(paths["delta"], mmap_mode="r")
        assert delta.shape[0] == np.load(paths["sim_ids"]).shape[0]

    def test_sim_ids_filter_subset(self, synthetic_data_dir, tmp_path):
        out = tmp_path / "shards-subset"
        paths = rddac.streaming.export_to_numpy(
            "force-curve",
            out,
            data_dir=str(synthetic_data_dir),
            sim_ids=[1, 3],
        )
        assert np.load(paths["sim_ids"]).tolist() == [1, 3]
        assert np.load(paths["force_data"], mmap_mode="r").shape[0] == 2

    def test_empty_export_raises(self, synthetic_data_dir, tmp_path):
        with pytest.raises(ValueError):
            rddac.streaming.export_to_numpy(
                "force-curve",
                tmp_path / "shards-empty",
                data_dir=str(synthetic_data_dir),
                sim_ids=[999_999_999],
            )


class TestExportToNumpyPerSim:
    """`export_to_numpy_per_sim` writes one `<sim_id>.npz` per experiment,
    allowing per-experiment shapes to vary (the raw force tables do)."""

    def test_one_npz_per_experiment(self, synthetic_data_dir, tmp_path):
        out = rddac.streaming.export_to_numpy_per_sim(
            "force-curve",
            tmp_path / "per-sim",
            data_dir=str(synthetic_data_dir),
        )
        names = sorted(p.name for p in out.glob("*.npz"))
        assert names == ["1.npz", "2.npz", "3.npz"]

    def test_round_trip_matches_stream(self, synthetic_data_dir, tmp_path):
        out = rddac.streaming.export_to_numpy_per_sim(
            "force-curve",
            tmp_path / "per-sim",
            data_dir=str(synthetic_data_dir),
        )
        streamed = {
            rec["_sim_id"]: rec
            for rec in rddac.streaming.iter_view(
                "force-curve", data_dir=str(synthetic_data_dir)
            )
        }
        for sim_id, rec in streamed.items():
            with np.load(out / f"{sim_id}.npz") as npz:
                np.testing.assert_array_equal(npz["force_data"], rec["force_data"])

    def test_compressed_flag(self, synthetic_data_dir, tmp_path):
        out = rddac.streaming.export_to_numpy_per_sim(
            "force-curve",
            tmp_path / "per-sim-c",
            data_dir=str(synthetic_data_dir),
            sim_ids=[1],
            compressed=True,
        )
        with np.load(out / "1.npz") as npz:
            assert npz["force_data"].shape == (N_FORCE, 8)


class TestLoadExport:
    """`load_export` is the lazy reader counterpart to `export_to_numpy`."""

    @pytest.fixture
    def shard_dir(self, synthetic_data_dir, tmp_path):
        out = tmp_path / "shards"
        rddac.streaming.export_to_numpy("force-curve", out, data_dir=str(synthetic_data_dir))
        return out

    def test_basic_attributes(self, shard_dir):
        export = rddac.streaming.load_export(shard_dir)
        assert len(export) == 3
        assert set(export.fields) == {"force_data"}

    def test_getitem_returns_dict(self, shard_dir):
        export = rddac.streaming.load_export(shard_dir)
        rec = export[0]
        assert set(rec.keys()) == set(export.fields)
        for v in rec.values():
            assert isinstance(v, np.ndarray)

    def test_iteration_matches_indexing(self, shard_dir):
        export = rddac.streaming.load_export(shard_dir)
        iterated = list(export)
        assert len(iterated) == len(export)
        for i, rec in enumerate(iterated):
            for alias in export.fields:
                np.testing.assert_array_equal(rec[alias], export[i][alias])

    def test_by_sim_id(self, shard_dir):
        export = rddac.streaming.load_export(shard_dir)
        first_sim_id = int(export.sim_ids[0])
        rec_by_id = export.by_sim_id(first_sim_id)
        rec_by_idx = export[0]
        for alias in export.fields:
            np.testing.assert_array_equal(rec_by_id[alias], rec_by_idx[alias])
        with pytest.raises(KeyError):
            export.by_sim_id(999_999_999)

    def test_index_out_of_range(self, shard_dir):
        export = rddac.streaming.load_export(shard_dir)
        with pytest.raises(IndexError):
            _ = export[len(export)]

    def test_missing_sim_ids_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rddac.streaming.load_export(tmp_path)

    def test_fields_subset(self, shard_dir):
        export = rddac.streaming.load_export(shard_dir, fields=["force_data"])
        assert export.fields == ("force_data",)
        assert set(export[0].keys()) == {"force_data"}

    def test_unknown_field_raises(self, shard_dir):
        with pytest.raises(ValueError, match="unknown field"):
            rddac.streaming.load_export(shard_dir, fields=["force_data", "typo"])

    def test_works_with_torch_dataloader(self, shard_dir):
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        export = rddac.streaming.load_export(shard_dir)
        loader = DataLoader(export, batch_size=min(2, len(export)), shuffle=False)
        batch = next(iter(loader))
        assert set(batch.keys()) == set(export.fields)
        for v in batch.values():
            assert v.shape[0] <= 2


class TestIterViewRealData:
    """Guarded real-data streaming — restricted to a couple of experiment ids."""

    def test_force_curve_over_guarded_ids(self, real_data_dir):
        records = list(
            rddac.streaming.iter_view(
                "force-curve",
                data_dir=str(real_data_dir),
                sim_ids=REAL_EXPERIMENT_IDS,
            )
        )
        assert [rec["_sim_id"] for rec in records] == REAL_EXPERIMENT_IDS
        for rec in records:
            assert rec["force_data"].ndim == 2
            assert rec["force_data"].shape[1] == 8
            assert rec["force_data"].dtype == np.float32

    def test_thickness_view_over_guarded_ids(self, real_data_dir):
        records = list(
            rddac.streaming.iter_view(
                "thickness",
                data_dir=str(real_data_dir),
                sim_ids=REAL_EXPERIMENT_IDS[:2],
                where=lambda row: bool(row["has_oil"]),
            )
        )
        assert 1 <= len(records) <= 2
        for rec in records:
            assert rec["sheet_thickness_data"].shape[1] == 2
            assert rec["oil_thickness_data"].shape[1] == 2
