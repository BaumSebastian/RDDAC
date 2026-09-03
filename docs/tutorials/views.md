# Build your own view

`rddac.add_view` appends a custom `RecordSet` to the in memory dataset. The published manifest on DaRUS stays unchanged; only your in process copy grows.

The companion notebook at [`notebooks/02_views.ipynb`](https://github.com/BaumSebastian/RDDAC/tree/main/notebooks/02_views.ipynb) reproduces every cell below.

## 1. Load the dataset

```python
import rddac

from pathlib import Path
DATA_DIR = Path('./data')      # repository root
# DATA_DIR = Path('../data')   # uncomment instead when running from inside notebooks/
experiment_id = 0

ds = rddac.load(data_dir=DATA_DIR)
print([rs.id for rs in ds.metadata.record_sets])
```

Output:

```
['process-parameters', 'field-map',
 'pointcloud-op10', 'pointcloud-op20',
 'force-curve', 'thickness']
```

The dataset already ships several published RecordSets (`force-curve`, `pointcloud-op10`, ...). When none of them fits a particular task, build your own.

## 2. Append a custom RecordSet

```python
rddac.add_view(
    ds,
    "my-view",
    fields={
        "force":  "force_data",                    # whole field, shortcut form
        "sheet":  ("sheet_thickness_data", None),  # whole field, explicit form
        "op10_z": "pointcloud_op10_z",             # a second modality in the same view
    },
)
```

The first argument is the loaded dataset, the second is the new RecordSet's identifier, and `fields` maps an alias to a field-map field.

Field IDs (such as `force_data`) come from the field-map RecordSet declared in the Croissant manifest. The complete list lives in the [HDF5 structure reference](../hdf5-structure.md#complete-field-reference).

## `fields` value shapes

| Value | Meaning | Yielded shape / dtype |
|-------|---------|------------------------|
| `"field_id"` | whole field-map field (shortcut) | full HDF5 shape, e.g. `(n, 8) float32` for `force_data` |
| `("field_id", None)` | whole field-map field (explicit) | same as above |
| `("field_id", int)` | one index along axis 0 (field-map only) | one axis dropped |
| `("field_id", [int, int, ...])` | subset along axis 0 (field-map only) | subset along axis 0 |
| `"<record-set>/<field>"` | qualified id, pulls from any RecordSet (e.g. `"process-parameters/blankholder_force"`) | depends on the source, scalars for CSV columns, full arrays for field-map fields |

The `(field_id, slicing)` tuple form is kept for one-to-one DDACS compatibility, where it selects simulation timesteps. RDDAC's raw fields have no timestep axis and their leading dimension `n` varies per experiment, so slicing is rarely useful here; the bare-string form covers almost every RDDAC view.

## Mix in process-parameters columns

`add_view` also accepts **qualified** field IDs of the form `"<record-set>/<field>"`, which lets a single view combine HDF5 fields from `field-map` with CSV columns from `process-parameters`. The two sources are joined on the experiment id automatically. `mlcroissant` requires an explicit cross-source `references` link to validate the manifest, and `add_view` injects one on the first field-map field for you, so no manual join wiring is needed on the consumer side.

```python
rddac.add_view(
    ds,
    "my-view-with-params",
    fields={
        "force":             "force_data",
        "geometry":          "process-parameters/geometry",
        "blankholder_force": "process-parameters/blankholder_force",
        "oil_type":          "process-parameters/oil_type",
        "split":             "process-parameters/split",
    },
)
```

Iterating via [`rddac.streaming.iter_view`](streaming.md) yields one record per experiment with both kinds of fields side by side: HDF5 arrays and CSV columns appear under their respective aliases.

```python
for rec in rddac.streaming.iter_view(
    "my-view-with-params",
    data_dir=DATA_DIR,
    dataset=ds,
    sim_ids=[experiment_id],
):
    print(f"force.shape={rec['force'].shape}")
    print(f"geometry={rec['geometry']!r}  blankholder_force={rec['blankholder_force']}")
    print(f"oil_type={rec['oil_type']!r}  split={rec['split']!r}")
```

Output:

```
force.shape=(1140, 8)
geometry='concave'  blankholder_force=100
oil_type='coarse'  split='val'
```

Slicing is rejected for non-`field-map` sources: process-parameters columns are scalar per record and have no axis to slice.

## 3. Inspect the new view's fields

```python
view = next(rs for rs in ds.metadata.record_sets if rs.id == "my-view")
for f in view.fields:
    transforms = [t.json_path for t in (f.source.transforms or [])]
    print(f"  {f.id:20s} <- {f.source.uuid:35s} transforms={transforms}")
```

Output:

```
my-view/force   <- field-map/force_data              transforms=[]
my-view/sheet   <- field-map/sheet_thickness_data    transforms=[]
my-view/op10_z  <- field-map/pointcloud_op10_z       transforms=[]
```

`f.source.uuid` is the field-map field that supplies the data, `transforms` records the optional axis-0 selection.

## 4. Iterate records

Each record is a dict keyed by the aliases declared in `add_view`. The canonical iteration path is `rddac.streaming.iter_view` (or `RDDACDataset` for training), **not** `ds.records()`:

!!! warning "`ds.records()` does not scale to a partial download"
    `mlcroissant` walks every zip referenced by the FileSet at iterator setup, before yielding the first record. With the **small bundle** only `sample.zip` is on disk, so `ds.records("my-view")` raises a `GenerationError` on the first missing geometry zip. With the **full release** it works but pays the full FileSet setup walk first.

    Two replacements both work and both scale without the FileSet setup walk:

    - **[`rddac.streaming.iter_view`](streaming.md)** is the no-PyTorch generator. Same semantics, same per-record dict, no torch dependency.

      ```python
      for rec in rddac.streaming.iter_view("my-view", data_dir=DATA_DIR, dataset=ds):
          ...
      ```

    - **[`RDDACDataset`](pytorch.md#8-stream-a-custom-view-add_view-dataset)** wraps the same iteration in a `torch.utils.data.IterableDataset` for `DataLoader` batching, multi-worker sharding and DDP.

      ```python
      from rddac.pytorch import RDDACDataset
      custom_ds = RDDACDataset(view="my-view", data_dir=DATA_DIR, dataset=ds)
      ```

    Both build a lazy `experiment id -> local zip` index at construction, open a zip only when a record is requested, scale to the full release, and silently skip missing zips on a partial download. The `dataset=` kwarg carries the in-memory `add_view` mutation through.

## 5. Read other RecordSets

The **`process-parameters`** RecordSet is the experiment index: one record per experiment, with every column of `process_parameters.csv` exposed as a named field. It is sourced from the CSV (not from the h5 zips), so the records iterator works regardless of which zips are downloaded locally.

```python
for n, rec in enumerate(ds.records("process-parameters"), start=1):
    if n == 1:
        for k, v in rec.items():
            print(f"  {k:40s} = {v}")
    if n >= 1:
        break
```

Output (first row):

```
process-parameters/index             = 0
process-parameters/experiment_id     = 1
process-parameters/category          = 0
process-parameters/geometry          = b'concave'
process-parameters/blankholder_force = 100
process-parameters/mean_punch_temp   = 20.2
process-parameters/oil_type          = b'coarse'
process-parameters/has_pointcloud    = True
process-parameters/has_oil           = True
process-parameters/split             = b'val'
```

## Where to go next

- [Streaming and numpy export](streaming.md) iterates the same view with `rddac.streaming.iter_view` (no PyTorch) and shows the `export_to_numpy` recipe for fast per-record reads after a one-shot conversion.
- [PyTorch training](pytorch.md) wraps a view (custom or published) in `RDDACDataset` for batched training. The "Stream a custom view" section shows the `add_view` -> `RDDACDataset(dataset=ds)` chain that streams the view you just built without re-parsing the manifest.
- [Visualization](visualization.md) plots arrays pulled from a record.
- [Croissant manifest](../croissant.md) explains the schema and the field-map RecordSet that underpins `add_view`.
