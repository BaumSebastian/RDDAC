"""Tests for `rddac.pytorch.RDDACDataset`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from rddac.pytorch import RDDACDataset  # noqa: E402

from .conftest import N_FORCE, REAL_EXPERIMENT_IDS  # noqa: E402


class TestBasicIteration:
    def test_field_specs_resolved(self, synthetic_data_dir):
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        assert ds._field_specs == {"force_data": ("force/data", None)}

    def test_sim_ids_loaded_from_csv(self, synthetic_data_dir):
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        assert ds._sim_ids == [1, 2, 3]

    def test_h5_index_built_from_zero_padded_members(self, synthetic_data_dir):
        """`0001.h5` inside a geometry zip indexes as experiment 1."""
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        assert set(ds._h5_index.keys()) == {1, 2, 3}
        for path in ds._h5_index.values():
            assert path.endswith(".zip")

    def test_iterates_one_record_per_experiment(self, synthetic_data_dir):
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        records = list(ds)
        assert len(records) == 3
        for rec in records:
            assert set(rec.keys()) == {"force_data"}
            assert rec["force_data"].shape == (N_FORCE, 8)

    def test_invalid_view_raises(self, synthetic_data_dir):
        with pytest.raises(ValueError):
            RDDACDataset(view="nonexistent", data_dir=str(synthetic_data_dir))


class TestDatasetKwarg:
    """Cover the `dataset=` entry point: add_view -> RDDACDataset(dataset=ds)."""

    def test_custom_view_via_dataset_kwarg(self, synthetic_data_dir):
        import rddac

        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(
            ds,
            "scan-only",
            fields={"scan": ("pointcloud_op10_z", None)},
        )

        # Without `dataset=`, the constructor reloads the on-disk manifest and
        # the custom view is invisible.
        with pytest.raises(ValueError):
            RDDACDataset(
                view="scan-only",
                data_dir=str(synthetic_data_dir),
            )

        # With `dataset=`, the mutation carries through and iteration works.
        ds_iter = RDDACDataset(
            view="scan-only",
            data_dir=str(synthetic_data_dir),
            dataset=ds,
        )
        records = list(ds_iter)
        assert len(records) == 3
        for rec in records:
            assert "scan" in rec
            assert rec["scan"].ndim == 1  # flat (x_shape*y_shape,) scan buffer

    def test_sliced_custom_view(self, synthetic_data_dir):
        import rddac

        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(ds, "force-rows", fields={"rows": ("force_data", [0, 1])})
        ds_iter = RDDACDataset(view="force-rows", data_dir=str(synthetic_data_dir), dataset=ds)
        for rec in ds_iter:
            assert rec["rows"].shape == (2, 8)


class TestFilters:
    def test_sim_ids_allowlist(self, synthetic_data_dir):
        ds = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            sim_ids=[1, 3],
        )
        assert ds._sim_ids == [1, 3]
        assert len(list(ds)) == 2

    def test_where_predicate(self, synthetic_data_dir):
        # Experiments 1, 2 are concave; 3 is convex.
        concave = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            where=lambda row: row["geometry"] == "concave",
        )
        assert concave._sim_ids == [1, 2]

        convex = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            where=lambda row: row["geometry"] == "convex",
        )
        assert convex._sim_ids == [3]

        none = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            where=lambda row: row["geometry"] == "rectangular",
        )
        assert none._sim_ids == []

    def test_where_on_boolean_column(self, synthetic_data_dir):
        ds = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            where=lambda row: bool(row["has_pointcloud"]),
        )
        assert ds._sim_ids == [1, 2, 3]


class TestDataLoader:
    def test_batch_size_two(self, synthetic_data_dir):
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        batches = list(loader)
        # 3 records, batch_size=2 -> [2, 1]
        assert [tuple(b["force_data"].shape) for b in batches] == [
            (2, N_FORCE, 8),
            (1, N_FORCE, 8),
        ]

    def test_num_workers_sharding_no_duplicates(self, synthetic_data_dir):
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        loader = DataLoader(ds, batch_size=1, num_workers=3)
        seen = set()
        for batch in loader:
            arr = batch["force_data"].numpy()
            # Each batch carries one record; use the first element as a fingerprint.
            seen.add(float(arr[0, 0, 0]))
        assert len(seen) == 3, "expected exactly one record per experiment, no duplicates"


class TestShuffle:
    def test_seeded_shuffle_is_reproducible(self, synthetic_data_dir):
        a = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            shuffle=True,
            seed=42,
        )
        b = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            shuffle=True,
            seed=42,
        )
        # Same seed + same epoch -> same shard order
        order_a = [rec["force_data"][0, 0].item() for rec in a]
        order_b = [rec["force_data"][0, 0].item() for rec in b]
        assert order_a == order_b

    def test_set_epoch_changes_order(self, synthetic_data_dir):
        ds = RDDACDataset(
            view="force-curve",
            data_dir=str(synthetic_data_dir),
            shuffle=True,
            seed=42,
        )
        ds.set_epoch(0)
        epoch0 = [rec["force_data"][0, 0].item() for rec in ds]
        ds.set_epoch(1)
        epoch1 = [rec["force_data"][0, 0].item() for rec in ds]
        # Either order may match by chance, but with seed=42 across the 3
        # synthetic ids the two permutations differ.
        assert epoch0 != epoch1


class TestShardMath:
    def test_sharding_partitions_sim_ids_exactly(self, synthetic_data_dir):
        """Pure-math check on the worker x DDP slicing: for every shard count,
        the union of `[shard::total]` slices is exactly the sim_id list."""
        ds = RDDACDataset(view="force-curve", data_dir=str(synthetic_data_dir))
        sim_ids = ds._sim_ids
        for total in (1, 2, 3, 4, 8):
            seen: list[int] = []
            for shard in range(total):
                seen.extend(sim_ids[shard::total])
            assert sorted(seen) == sorted(sim_ids), f"shard math wrong at total={total}"


class TestRealData:
    """Guarded real-data checks — a couple of experiment ids, never all 9000."""

    def test_streams_guarded_ids(self, real_data_dir):
        ds = RDDACDataset(
            view="force-curve",
            data_dir=str(real_data_dir),
            sim_ids=REAL_EXPERIMENT_IDS,
        )
        assert ds._sim_ids == REAL_EXPERIMENT_IDS
        records = list(ds)
        assert len(records) == len(REAL_EXPERIMENT_IDS)
        for rec in records:
            assert rec["force_data"].ndim == 2
            assert rec["force_data"].shape[1] == 8

    def test_h5_index_covers_guarded_ids(self, real_data_dir):
        ds = RDDACDataset(
            view="force-curve",
            data_dir=str(real_data_dir),
            sim_ids=REAL_EXPERIMENT_IDS,
        )
        for exp_id in REAL_EXPERIMENT_IDS:
            assert exp_id in ds._h5_index
