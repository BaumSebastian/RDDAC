# PyTorch training

`RDDACDataset` is a [`torch.utils.data.IterableDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) over a Croissant view. It is the iteration path to reach for when training a model: the underlying `mlcroissant.Dataset.records(...)` aborts on the first missing zip, while `RDDACDataset` builds an `experiment id -> local zip` index at construction time and silently skips experiments whose zip is not present. Full constructor signature in the [API reference](../api/pytorch.md).

The companion notebook at [`notebooks/03_pytorch.ipynb`](https://github.com/BaumSebastian/RDDAC/tree/main/notebooks/03_pytorch.ipynb) reproduces every cell below.

**Prerequisites.** This tutorial assumes working familiarity with:

- [PyTorch's `torch.utils.data` API](https://docs.pytorch.org/docs/stable/data.html), in particular [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) and the `Dataset` / `IterableDataset` protocols.
- [`pandas.Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) (the `where` filter receives one row of `process_parameters.csv` as a Series).
- [Croissant 1.1 / `mlcroissant`](https://doi.org/10.1145/3650203.3663326) for the manifest model and the `RecordSet` vocabulary used here.

## 1. Construct an `RDDACDataset`

`RDDACDataset(view, data_dir, ...)` loads the manifest the same way `rddac.load` does, resolves the view's fields to HDF5 paths, walks `data_dir` to build the local zip index, and reads `process_parameters.csv` to apply optional filters. After construction `ds._sim_ids` is the list of experiment ids the dataset will attempt to iterate, and `ds._h5_index` is the subset for which a zip is actually on disk.

```python
import rddac
from rddac.pytorch import RDDACDataset
from torch.utils.data import DataLoader

from pathlib import Path
DATA_DIR = Path('./data')      # repository root
# DATA_DIR = Path('../data')   # uncomment instead when running from inside notebooks/
experiment_id = 0

ds = RDDACDataset(view='force-curve', data_dir=DATA_DIR)
print('view:           ', ds.view)
print('field specs:    ', ds._field_specs)
print('total sim_ids:  ', len(ds._sim_ids), '(every experiment in process_parameters.csv)')
print('locally indexed:', len(ds._h5_index), '(only these will actually stream)')
```

Output (with the small bundle on disk):

```
view:            force-curve
field specs:     {'force_data': ('force/data', None)}
total sim_ids:   9000 (every experiment in process_parameters.csv)
locally indexed: 18   (only these will actually stream)
```

## 2. Iterate one batch through a `DataLoader`

`RDDACDataset` plugs straight into a [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). The default `collate_fn` stacks each field along a new leading batch axis:

```python
loader = DataLoader(ds, batch_size=1, num_workers=0)
for batch in loader:
    for k, v in batch.items():
        print(f'  {k:12s} shape={tuple(v.shape)} dtype={v.dtype}')
    break
```

Output:

```
force_data   shape=(1, 1140, 8) dtype=torch.float32
```

!!! warning "Raw records have variable shapes — mind the batch size"
    The sample count `n` of the force and traverse tables **varies per experiment** (raw data as recorded). The default `collate_fn` can only stack tensors of identical shape, so `batch_size > 1` fails on `force-curve` and `thickness` unless you pass a custom `collate_fn` that pads, truncates or resamples to a common length (see [Custom collate](#custom-collate)). The two pointcloud views are fixed-shape — every scan buffer is `(6400000,)` — and batch without extra work, at ~25 MB per field per record.

## 3. Filter via the Croissant manifest

Both filters run against `process_parameters.csv` **before** any zip is opened, so IO scales with the surviving experiments rather than with the full 9,000.

The row keys are not magic: they come straight from the Croissant manifest. `metadata.json` declares `process_parameters.csv` as a `FileObject` and exposes its columns as the `process-parameters` `RecordSet`. The same fields are what `ds.records('process-parameters')` yields when iterating through `mlcroissant` directly. `RDDACDataset` simply consumes those rows at construction time and applies the predicate before any zip is touched.

### What `where` receives

`RDDACDataset` reads `process_parameters.csv` with `pandas` and runs `where` once per row via `df.apply(where, axis=1)`. The `row` argument is a [`pandas.Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) whose index is the CSV column names, so you can read columns with either `row['split']` or `row.split`. Values are native Python types:

| Column | Type | Example |
|--------|------|---------|
| `index`, `experiment_id`, `category` | `int` | `0`, `1`, `0` |
| `geometry` | `str` | `'concave'`, `'convex'` |
| `blankholder_force` | `int` | `100`, `300`, `500` |
| `mean_punch_temp` | `float` | `20.2` |
| `oil_type` | `str` | `'coarse'`, `'medium'`, `'fine'` |
| `has_pointcloud`, `has_oil` | `bool` | `True` |
| `split` | `str` | `'train'`, `'val'`, `'test'` |

- `where=<callable>`: any function `pd.Series -> bool`. Available column names match the manifest's `process-parameters` fields.
- `sim_ids=[...]`: explicit allowlist of integers, applied before `where`. (The name is kept for drop-in DDACS compatibility; the values are RDDAC experiment ids.)

Both can be combined; the predicate is applied after the allowlist.

```python
convex = RDDACDataset(
    view='force-curve',
    data_dir=DATA_DIR,
    where=lambda row: row['geometry'] == 'convex',
)
print(f'convex-only sim_ids: {len(convex._sim_ids):>5d} (of 9000)')

ids_only = RDDACDataset(
    view='force-curve',
    data_dir=DATA_DIR,
    sim_ids=[experiment_id],
)
print(f'sim_ids=[experiment_id]: {len(ids_only._sim_ids):>3d}')
```

Output:

```
convex-only sim_ids:  4500 (of 9000)
sim_ids=[experiment_id]:   1
```

### Filtering out missing measurements

The most important RDDAC-specific predicate: views that read the `oil_thickness/` or `pointcloud/` groups must exclude the experiments where the group does not exist, otherwise the HDF5 read raises a `KeyError` mid-iteration. The published `thickness` view reads the oil group:

```python
thickness = RDDACDataset(
    view='thickness',
    data_dir=DATA_DIR,
    where=lambda row: row['has_oil'],       # {{ missing_oil() }} experiments lack the oil group
)
```

The same applies to `has_pointcloud` for the two pointcloud views (`where=lambda row: row['has_pointcloud']`).

### `where` only sees CSV columns

The predicate runs before any HDF5 is opened, so it cannot read HDF5 attributes. In practice every per-experiment root attribute of the HDF5 files is already mirrored in `process_parameters.csv` (`geometry`, `blankholder_force`, `oil_type`, `mean_punch_temp`, ...), so the CSV path covers all per-experiment filtering today. For an HDF5-only attribute, scan the archives once to build an id allowlist and pass it through `sim_ids=...`.

## 4. Train / val / test splits

`process_parameters.csv` ships with a `split` column whose canonical values are `'train'`, `'val'`, and `'test'` (80 / 10 / 10, seed 42). Because the column is part of the Croissant manifest, the same `where=` predicate that filtered by geometry above works on it. Three `RDDACDataset` instances, one per split, is the fastest way to wire up a training loop without writing any custom partitioning code.

Shuffle the train split for SGD; leave `val`/`test` deterministic for reproducible evaluation.

```python
splits = {}
for name in ('train', 'val', 'test'):
    splits[name] = RDDACDataset(
        view='force-curve',
        data_dir=DATA_DIR,
        where=lambda row, n=name: row['split'] == n,
        shuffle=(name == 'train'),
        seed=42,
    )

for name, split_ds in splits.items():
    streamable = sum(1 for sid in split_ds._sim_ids if sid in split_ds._h5_index)
    print(f"{name:>5s}: {len(split_ds._sim_ids):>5d} sim_ids (of 9000), "
          f"{streamable:>2d} streamable now (zip on disk)")
```

Output (with the small bundle on disk):

```
train:  7200 sim_ids (of 9000),  14 streamable now (zip on disk)
  val:   900 sim_ids (of 9000),   2 streamable now (zip on disk)
 test:   900 sim_ids (of 9000),   2 streamable now (zip on disk)
```

`len(_sim_ids)` is the CSV-side count after the predicate; the second column counts only the rows whose zip is on disk and will actually yield a record. With the full release on disk, the second column matches `len(_sim_ids)`.

## 5. Shuffle + `set_epoch`

`shuffle=True` permutes each shard with a seed derived from `seed + epoch + shard_id`. Worker shards stay disjoint, so two workers do not produce the same experiment. Call `set_epoch(n)` once per epoch to get a different permutation each time. Without it, every epoch sees the same order.

```python
ds_shuf = RDDACDataset(
    view='force-curve',
    data_dir=DATA_DIR,
    shuffle=True,
    seed=42,
)
for epoch in range(2):
    ds_shuf.set_epoch(epoch)
    # iterate ds_shuf ... each epoch visits the shard in a fresh order
```

## 6. Sharding

Workers and DDP ranks are detected from `torch.utils.data.get_worker_info()` and `torch.distributed`. The same `RDDACDataset` instance works under `num_workers=0`, `num_workers=N`, and DDP without constructor changes: each shard slices `ds._sim_ids` by `(rank * num_workers + worker_id)` modulo `(world_size * num_workers)`, so the partition is exhaustive and disjoint.

## 7. Partial download: graceful skip vs `records()` abort

`RDDACDataset` is the iteration path to use when only a subset of zips is on disk. `mlcroissant.Dataset.records(...)` walks every zip in the FileSet and aborts at the first missing one; `RDDACDataset` checks the local index per experiment and silently skips.

```python
# 1) mlcroissant records : aborts on the first missing zip.
raw = rddac.load(data_dir=DATA_DIR)
try:
    next(iter(raw.records('force-curve')))
except Exception as e:
    print(f'records(): {type(e).__name__}')

# 2) RDDACDataset : graceful skip.
yielded = sum(1 for _ in RDDACDataset(view='force-curve', data_dir=DATA_DIR))
print(f'RDDACDataset: yielded {yielded} record(s) without error')
```

Output (with the small bundle on disk):

```
records(): GenerationError
RDDACDataset: yielded 18 record(s) without error
```

## 8. Stream a custom view (`add_view` + `dataset=`)

The published views (`force-curve`, `pointcloud-op10`, ...) cover the common targets, but the [Build your own view](views.md) tutorial shows how to compose a custom `RecordSet` with `rddac.add_view`. `add_view` mutates the in-memory `mlcroissant.Dataset` you hand it; the published manifest on DaRUS stays unchanged.

To stream that custom view through `RDDACDataset`, pass the same loaded object via the `dataset=` kwarg. Without it, `RDDACDataset.__init__` would re-parse `metadata.json` from disk and the in-memory mutation would be invisible (it would raise `ValueError: view 'my-view' not found`). With it, the constructor uses the caller's object as-is.

```python
ds_manifest = rddac.load(data_dir=DATA_DIR)
rddac.add_view(
    ds_manifest,
    'force-only',
    fields={
        'force': 'force_data',
    },
)

custom_ds = RDDACDataset(
    view='force-only',
    data_dir=DATA_DIR,
    dataset=ds_manifest,
)

print('view:           ', custom_ds.view)
print('field specs:    ', custom_ds._field_specs)
print('total sim_ids:  ', len(custom_ds._sim_ids))
```

Output (with the small bundle on disk):

```
view:            force-only
field specs:     {'force': ('force/data', None)}
total sim_ids:   9000
```

Same flow with `where=`, `sim_ids=`, and `shuffle=` works against `custom_ds` — the custom view is just another `RecordSet` once it lives on `ds_manifest`.

If you do not need a `DataLoader` or PyTorch at all, `rddac.streaming.iter_view(view='force-only', data_dir=DATA_DIR, dataset=ds_manifest)` is the no-torch equivalent that yields the same records one at a time. The [Streaming and numpy export](streaming.md) tutorial covers it and shows the matching `streaming.export_to_numpy` recipe for materialising a view as flat `.npy` shards.

## Custom collate

Records with variable sample counts need a custom `collate_fn`. A simple, information-preserving option for the 1D signals is to resample every record to a fixed length on the time axis before stacking:

```python
import numpy as np
import torch

N_SAMPLES = 1000

def resample_collate(batch):
    """Interpolate each force table to N_SAMPLES rows, then stack."""
    out = []
    for rec in batch:
        force = np.asarray(rec['force_data'])           # (n, 8), n varies
        t_old = np.linspace(0.0, 1.0, force.shape[0])
        t_new = np.linspace(0.0, 1.0, N_SAMPLES)
        resampled = np.stack([np.interp(t_new, t_old, force[:, c]) for c in range(force.shape[1])], axis=1)
        out.append(resampled)
    return {'force_data': torch.as_tensor(np.stack(out))}   # (B, N_SAMPLES, 8)

loader = DataLoader(ds, batch_size=16, collate_fn=resample_collate)
```

Padding plus a mask works too when absolute sample positions matter.

## Where to go next

- [Build your own view](views.md) explains how the field map drives `RDDACDataset` field selection.
- [Visualization](visualization.md) plots arrays pulled from a single record.
- [Streaming and numpy export](streaming.md) covers the no-PyTorch `iter_view` and the one-shot `export_to_numpy` materialisation.
- [API reference](../api/pytorch.md) lists every constructor argument.
