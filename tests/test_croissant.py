"""Tests for `rddac.load`, `rddac.add_view` and internal parsers."""

from __future__ import annotations

import pytest

import rddac
from rddac.croissant import _normalize_field_spec, _slicing_to_jsonpath

# ---------------------------------------------------------------------------
# Pure parsers
# ---------------------------------------------------------------------------


class TestNormalizeFieldSpec:
    def test_bare_string_means_whole_field(self):
        assert _normalize_field_spec("foo") == ("foo", None)

    def test_tuple_explicit_none(self):
        assert _normalize_field_spec(("foo", None)) == ("foo", None)

    def test_tuple_integer(self):
        assert _normalize_field_spec(("foo", 2)) == ("foo", 2)

    def test_tuple_list(self):
        assert _normalize_field_spec(("foo", [1, 2, 3])) == ("foo", [1, 2, 3])

    def test_rejects_wrong_tuple_arity(self):
        with pytest.raises(TypeError):
            _normalize_field_spec(("foo",))

    def test_rejects_non_string_field_id(self):
        with pytest.raises(TypeError):
            _normalize_field_spec((42, 0))


class TestSlicingToJsonPath:
    def test_none_is_no_transform(self):
        assert _slicing_to_jsonpath(None) is None

    def test_integer_single_index(self):
        assert _slicing_to_jsonpath(2) == "$[2]"

    def test_list_multi_index(self):
        assert _slicing_to_jsonpath([2, 3]) == "$[2,3]"

    def test_rejects_bool(self):
        # bool is an int subclass; we don't want True/False to silently work.
        with pytest.raises(TypeError):
            _slicing_to_jsonpath(True)

    def test_rejects_mixed_list(self):
        with pytest.raises(TypeError):
            _slicing_to_jsonpath([1, "x"])


# ---------------------------------------------------------------------------
# `load` over the synthetic dataset
# ---------------------------------------------------------------------------


class TestLoadSynthetic:
    def test_returns_mlc_dataset(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        assert type(ds).__module__.startswith("mlcroissant")
        assert ds.metadata.name == "synthetic-rddac"

    def test_mapping_picks_up_local_files(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        # 2 geometry zips + 1 csv
        assert len(ds.mapping or {}) == 3

    def test_record_sets_parsed(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        ids = {rs.id for rs in ds.metadata.record_sets}
        assert ids == {"process-parameters", "field-map", "force-curve"}

    def test_file_set_contained_in_parsed(self, synthetic_data_dir):
        """Regression check: `cr:containedIn` alias in the manifest's @context."""
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        fs = next(n for n in ds.metadata.distribution if type(n).__name__ == "FileSet")
        assert len(fs.contained_in) == 2

    def test_process_parameters_descriptions(self, synthetic_data_dir):
        """Column descriptions come from the manifest, not the package."""
        from rddac.croissant import process_parameters_descriptions

        ds = rddac.load(data_dir=str(synthetic_data_dir))
        desc = process_parameters_descriptions(ds)
        assert set(desc) == {
            "index",
            "experiment_id",
            "category",
            "geometry",
            "blankholder_force",
            "mean_punch_temp",
            "oil_type",
            "has_pointcloud",
            "has_oil",
            "split",
        }

    def test_field_map_helper(self, synthetic_data_dir):
        from rddac.croissant import field_map

        ds = rddac.load(data_dir=str(synthetic_data_dir))
        fm = field_map(ds)
        assert "force_data" in fm
        assert "pointcloud_op10_z" in fm
        assert fm["force_data"].source.transforms[0].regex == "force/data"


# ---------------------------------------------------------------------------
# `add_view`
# ---------------------------------------------------------------------------


class TestAddView:
    def test_adds_record_set_in_place(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        before = {rs.id for rs in ds.metadata.record_sets}
        rddac.add_view(
            ds,
            "my-view",
            fields={
                "first_force_row": ("force_data", 0),
                "op10_z": "pointcloud_op10_z",
            },
        )
        after = {rs.id for rs in ds.metadata.record_sets}
        assert after == before | {"my-view"}

    def test_field_specs_round_trip(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(
            ds,
            "my-view",
            fields={
                "whole": "force_data",
                "one": ("force_data", 2),
                "subset": ("force_data", [2, 3]),
            },
        )
        rs = next(r for r in ds.metadata.record_sets if r.id == "my-view")
        by_name = {f.name: f for f in rs.fields}

        # "whole" should have no transform
        assert not by_name["whole"].source.transforms

        # "one" should have $[2]
        assert by_name["one"].source.transforms[0].json_path == "$[2]"

        # "subset" should have $[2,3]
        assert by_name["subset"].source.transforms[0].json_path == "$[2,3]"

    def test_returns_dataset_for_chaining(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        out = rddac.add_view(ds, "tmp", fields={"x": "pointcloud_op10_z"})
        assert out is ds

    def test_mixed_sources_with_metadata_columns(self, synthetic_data_dir):
        """Qualified 'process-parameters/<col>' ids pull CSV columns into a view."""
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(
            ds,
            "mixed",
            fields={
                "force": "force_data",
                "geometry": "process-parameters/geometry",
                "bhf": "process-parameters/blankholder_force",
            },
        )
        rs = next(r for r in ds.metadata.record_sets if r.id == "mixed")
        by_name = {f.name: f for f in rs.fields}
        assert by_name["geometry"].source.uuid == "process-parameters/geometry"
        assert by_name["force"].source.uuid == "field-map/force_data"

    def test_slicing_on_csv_source_rejected(self, synthetic_data_dir):
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        with pytest.raises(ValueError):
            rddac.add_view(
                ds,
                "bad",
                fields={"geometry": ("process-parameters/geometry", 0)},
            )

    def test_accumulates_across_calls(self, synthetic_data_dir):
        """Two add_view calls on the same dataset must both survive."""
        ds = rddac.load(data_dir=str(synthetic_data_dir))
        rddac.add_view(ds, "view-a", fields={"a": "force_data"})
        rddac.add_view(ds, "view-b", fields={"b": "sheet_thickness_data"})
        ids = {rs.id for rs in ds.metadata.record_sets}
        assert {"view-a", "view-b"} <= ids
