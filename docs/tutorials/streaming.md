# Streaming and numpy export

`rddac.streaming` is the offline-iteration counterpart to `RDDACDataset`. It exists to solve two problems the published `mlcroissant.Dataset.records(view)` path does not:

The companion notebook at [`notebooks/06_streaming.ipynb`](https://github.com/BaumSebastian/RDDAC/tree/main/notebooks/06_streaming.ipynb) reproduces every cell below.

1. **Setup cost.** `records()` walks every zip in the FileSet before yielding the first record and aborts on the first missing one. `streaming.iter_view` opens zips on demand, skips missing ones, and yields the first record immediately.
2. **Inner-loop training cost.** Iterating an HDF5 view is bound by gzip decompression and h5py per-call overhead. For a 9,000-experiment epoch this is the bottleneck. `streaming.export_to_numpy` walks the view once and writes flat `.npy` memmap shards; from then on the training loop reads records in microseconds via `np.load(..., mmap_mode='r')`.

Both functions accept either the published Croissant views (`force-curve`, `pointcloud-op10`, ...) or any custom view added with `rddac.add_view`, via the `dataset=` kwarg.

```python
import time
from pathlib import Path

import numpy as np

import rddac

DATA_DIR   = Path('./data')     # repository root
# DATA_DIR = Path('../data')   # uncomment instead when running from inside notebooks/
EXPORT_DIR = DATA_DIR / 'tutorial_export'
SAMPLE_IDS = [0, 500, 1002, 1500, 2000, 2500, 3000, 3500, 4000,
              4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500]   # `rddac download --small`
```

## 1. Register a view

A custom view as introduced in [Build your own view](views.md): the raw force table plus selected metadata columns, exposed as an in-memory mutation of the loaded `ds`. The published manifest on DaRUS is not modified.

```python
ds = rddac.load(data_dir=DATA_DIR)
rddac.add_view(ds, 'force-signals', fields={
    'force': 'force_data',                                        # (n, 8) HDF5 table
    'geometry': 'process-parameters/geometry',                    # CSV columns ride along
    'blankholder_force': 'process-parameters/blankholder_force',
    'split': 'process-parameters/split',
})
```

## 2. Preview one record with `streaming.iter_view`

`rddac.streaming.iter_view(view='force-signals', dataset=ds)` is the plain-Python counterpart to `RDDACDataset.__iter__`. No PyTorch dependency, no `mlcroissant.Dataset.records()` FileSet walk. It opens each zip on demand and yields one `dict` per experiment: HDF5 fields as numpy arrays, CSV columns as scalars.

```python
for rec in rddac.streaming.iter_view('force-signals', data_dir=DATA_DIR, dataset=ds, sim_ids=[0]):
    for alias, value in rec.items():
        arr = np.asarray(value)
        if arr.ndim:
            print(f'  {alias:18s} shape={arr.shape}  dtype={arr.dtype}')
        else:
            print(f'  {alias:18s} = {value!r}')
    break
```

Output (with the small bundle on disk):

```
  force              shape=(1140, 8)  dtype=float32
  geometry           = 'concave'
  blankholder_force  = 100
  split              = 'val'
  _sim_id            = 0
```

`_sim_id` is a private scratch key carrying the experiment id — handy inside transforms (e.g. per-experiment RNG seeding); `export_to_numpy` strips it before writing shards.

Replace `sim_ids=[0]` with `where=lambda row: row['split'] == 'train'` or simply omit both arguments to walk every experiment in `process_parameters.csv`. Remember the availability flags for views that touch the oil or pointcloud groups: `where=lambda row: row['has_oil']` (see [Missing measurements](../dataset.md#missing-measurements)).

The same function transparently handles both layouts: zipped (the default) and loose `.h5` files (the layout after `rddac download --extract --remove-zip`). Loose files take precedence when both contain the same experiment id.

## 3. Export to numpy with a shape-fixing transform

`export_to_numpy` walks the view once, applies any per-field or whole-record transforms, and writes each output alias into its own pre-allocated `.npy` memmap of shape `(n_experiments, *field_shape)`. It also writes `sim_ids.npy` so the row order is recoverable.

That pre-allocation implies a contract: **every record must produce the same shape per alias** — but the raw RDDAC force table is `(n, 8)` with `n` varying per experiment. The `record_transform` below fixes that and is a good template for the way training pipelines actually consume the dataset:

1. **Resample to a fixed length.** Linear interpolation onto 1,000 uniformly spaced points per channel gives every record the same `(1000, 8)` shape.
2. **Derive training targets.** The scalar `max_total_force` is precomputed at export time instead of being recomputed every epoch.
3. **Carry conditions and labels along.** The blankholder force rides next to the tensors, and the categorical geometry becomes a small-int label — anything that fits in a numpy array can ride alongside the main fields, declared by the `record_transform`. No separate metadata file required.

`transforms={alias: fn}` would do the same on a single field; `record_transform` is the right tool when the output keys do not match the input keys, as here.

```python
N_SAMPLES = 1000

def resample_and_emit(rec):
    """Interpolate the force table to a fixed length and derive targets."""
    force = np.asarray(rec['force'])                    # (n, 8), n varies per experiment
    t_old = np.linspace(0.0, 1.0, force.shape[0])
    t_new = np.linspace(0.0, 1.0, N_SAMPLES)
    resampled = np.stack(
        [np.interp(t_new, t_old, force[:, c]) for c in range(force.shape[1])], axis=1
    )
    return {
        'force':             resampled.astype(np.float32),        # (1000, 8), fixed
        'max_total_force':   np.float32(force[:, 7].max()),       # scalar target [kN]
        'blankholder_force': np.float32(rec['blankholder_force']),
        'label_geometry':    np.int8(0 if rec['geometry'] == 'concave' else 1),
    }


paths = rddac.streaming.export_to_numpy(
    'force-signals',
    EXPORT_DIR,
    data_dir=DATA_DIR,
    dataset=ds,
    sim_ids=SAMPLE_IDS,
    record_transform=resample_and_emit,
)
for alias, path in paths.items():
    size_kb = path.stat().st_size / 1024
    print(f'  {alias:18s} -> {path}   ({size_kb:6.1f} KB)')
```

Output:

```
  force              -> data/tutorial_export/force.npy   ( 562.6 KB)
  max_total_force    -> data/tutorial_export/max_total_force.npy   (   0.2 KB)
  blankholder_force  -> data/tutorial_export/blankholder_force.npy   (   0.2 KB)
  label_geometry     -> data/tutorial_export/label_geometry.npy   (   0.1 KB)
  sim_ids            -> data/tutorial_export/sim_ids.npy   (   0.3 KB)
```

## 4. Read the shards back with `load_export`

`rddac.streaming.load_export(directory)` opens the exported folder behind the standard Python data model (`__len__`, `__getitem__`, `__iter__`, plus `by_sim_id`). Each row is a plain `dict[str, np.ndarray]`. Reads are sub-millisecond after the first access and the full release fits even when it doesn't fit in RAM — only the rows you actually access are loaded from disk.

??? note "Why it is fast: memory-mapped reads in one paragraph (skip if not interested)"
    The shards are opened with `mmap_mode='r'`, which maps each `.npy` file directly into the process's virtual address space. Accessing `arr[i]` becomes an ordinary memory read; the operating system fetches whatever page that lives on from disk on demand and keeps a copy in its page cache. No pickle, no buffer allocation, no copy on the read path. The same page cache is shared across processes, so a `DataLoader(num_workers=N)` does not multiply the cached data by `N`. Cold reads pay the disk seek + page fault; warm reads are RAM-fast.

`load_export` deliberately does **not** import torch. It just satisfies the same `len + getitem` protocol PyTorch's map-style `Dataset` uses, so the returned object plugs into `DataLoader`, `tf.data.Dataset.from_generator`, JAX, or a plain Python loop without any adapter. Pass `fields=["force", "max_total_force"]` to load a subset; unknown names raise `ValueError` so typos surface immediately.

```python
export = rddac.streaming.load_export(EXPORT_DIR)

print(f'len(export)    = {len(export)}')
print(f'export.fields  = {export.fields}')
print(f'export.sim_ids = {export.sim_ids.tolist()}')

record = export[0]
for alias, value in record.items():
    print(f'  {alias:18s} shape={value.shape}  dtype={value.dtype}')
```

Output:

```
len(export)    = 18
export.fields  = ('blankholder_force', 'force', 'label_geometry', 'max_total_force')
export.sim_ids = [0, 500, 1002, 1500, 2000, 2500, 3000, 3500, 4000,
                  4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500]
  blankholder_force  shape=()  dtype=float32
  force              shape=(1000, 8)  dtype=float32
  label_geometry     shape=()  dtype=int8
  max_total_force    shape=()  dtype=float32
```

Random access by experiment id is one call:

```python
record = export.by_sim_id(4500)          # first convex sample experiment
print(f"max total force = {float(record['max_total_force']):.1f} kN")
```

Output:

```
max total force = 365.7 kN
```

For a PyTorch training loop, pass the same `export` straight into a `DataLoader` — the map-style `Dataset` protocol is just `len + getitem`, which this object provides natively:

```python
from torch.utils.data import DataLoader
loader = DataLoader(export, batch_size=16, shuffle=True)
```

## 5. Time the two paths back to back

Both paths compute the sum of `max_total_force` so we can verify numerical equality. The **per-record ratio** is what carries over to a full training loop: multiply by your epoch's experiment count to see the real win.

```python
t0 = time.perf_counter()
stream_sum, n = 0.0, 0
for rec in rddac.streaming.iter_view('force-signals', data_dir=DATA_DIR, dataset=ds, sim_ids=SAMPLE_IDS):
    stream_sum += float(np.asarray(rec['force'])[:, 7].max())
    n += 1
t_stream = time.perf_counter() - t0

t0 = time.perf_counter()
mmap_sum, m = 0.0, 0
for record in export:
    mmap_sum += float(record['max_total_force'])
    m += 1
t_mmap = time.perf_counter() - t0

print(f'iter_view   : {n} experiments in {1000 * t_stream:.2f} ms  ({1000 * t_stream / n:.2f} ms/exp)')
print(f'load_export : {m} experiments in {1000 * t_mmap:.2f} ms  ({1000 * t_mmap / m:.3f} ms/exp)')
```

Output (small bundle, warm cache):

```
iter_view   : 18 experiments in 282.59 ms  (15.70 ms/exp)
load_export : 18 experiments in 0.08 ms  (0.004 ms/exp)
```

Three orders of magnitude faster per record after a one-time export, and the numerical results match exactly (the target was precomputed by the same code path). On the full release the gap widens further: the per-record `iter_view` cost includes gzip decompression of every field the view touches — for the pointcloud views that is ~25 MB per buffer per experiment.

## 6. When records have variable shapes (`export_to_numpy_per_sim`)

`export_to_numpy` pre-allocates one memmap per alias, sized from record 0 as `(n_experiments, *field_shape)`. Every subsequent record must produce the exact same shape per alias; a shape mismatch raises a `ValueError` pointing here. The constraint is the right contract once a transform pinned the shapes — like the resampling above, or the fixed `(6400000,)` scan buffers. It is **not** the right contract for exports that keep the raw, per-experiment sample counts.

For those cases, use `rddac.streaming.export_to_numpy_per_sim`. Same iteration pipeline (`iter_view` + per-field `transforms` + whole-record `record_transform`), same `_sim_id` enrichment, but the writer is one `np.savez(<experiment_id>.npz)` per record instead of one memmap per alias. Each `.npz` carries all the aliases for one experiment; consumers reload via `np.load(path)` and access by key.

```python
rddac.streaming.export_to_numpy_per_sim(
    'force-signals',
    './data/raw_export',
    data_dir=DATA_DIR,
    dataset=ds,
    compressed=False,   # True for np.savez_compressed (smaller, slower load)
)
```

Trade-offs vs `export_to_numpy`:

- **No memmap layout.** Each load deserialises a small zip via numpy. Still fast warm, but it loses the OS-page-cache sharing across DataLoader workers that `export_to_numpy` gets for free.
- **No fixed-shape constraint.** Variable-`n` tables and ragged tensors are fine.
- **One file per experiment.** Random access by id is just `np.load(out_dir / f"{experiment_id}.npz")`.

When the data permits, prefer fixing the shape (resample the signals, keep the scan buffers whole, or crop/pad to a common size) and using `export_to_numpy` — the mmap path is faster. If nothing fits, `export_to_numpy_per_sim` is the honest answer.

## Where to go next

- The same `transforms` / `record_transform` pattern is exactly how you turn a categorical column into a small-int label: `transforms={'oil_type': lambda v: {'coarse': 0, 'medium': 1, 'fine': 2}[v]}`.
- `streaming.iter_view` is independent of the storage layout: it transparently reads loose `.h5` files (after `rddac download --extract --remove-zip`) and zipped archives. Loose files take precedence when both exist — see the [Loose HDF5 recipe](loose-h5.md).
- For batched, sharded training there is [`RDDACDataset`](pytorch.md) (PyTorch), which uses the same view-driven mechanics and benefits from the same `add_view` mutations via its `dataset=` kwarg.
