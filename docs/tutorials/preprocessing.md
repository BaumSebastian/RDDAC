# 7. Preprocessing

This tutorial runs the reference preprocessing on the small bundle, inspects the self-describing output, and shows how to adjust parameters reproducibly. The complete stage documentation lives in the [Preprocessing section](../preprocessing/index.md).

The companion notebook at [`notebooks/07_preprocessing.ipynb`](https://github.com/BaumSebastian/RDDAC/tree/main/notebooks/07_preprocessing.ipynb) reproduces every cell below.

## 1. Install and download

```bash
pip install 'rddac[preprocessing]'
rddac download --small -y          # ~174 MB sample bundle into ./data
```

The traverse and force stages run on the small bundle as-is. The `pointcloud` stage additionally needs the full release plus the DDACS simulations (`rddac download`, without `--no-sim`); section 6 shows how it degrades gracefully without them.

## 2. Run the reference preprocessing

```bash
rddac preprocess oil force sheet --data-dir ./data
```

Output (with the small bundle on disk):

```
oil: 18 processed
force: 18 processed
sheet: 18 processed
finished in 2.8 s
```

A second run reports `already in the output file, skipped (use --overwrite to recompute)`; experiments without a measurement (e.g. `has_oil = False`) show up as `skipped, the raw file has no such measurement`.

Output lands in `./data/processed/` as loose `<id>.h5` files. Raw files are never modified.

!!! note "There is only one `metadata.json`: the original one"
    `rddac preprocess` does **not** write a manifest into `processed/`. The Croissant manifest that `rddac download` placed in `./data/metadata.json` describes both layers, because the processed files keep the raw HDF5 paths (`force/data`, `oil_thickness/data`, ...). When you consume the processed layer you therefore change **only** `data_dir=` and point `source=` at the original manifest, otherwise `rddac` looks for a `metadata.json` inside `processed/`, finds none, and falls back to fetching it from DaRUS. Section 7 shows the call.

## 3. Inspect the self-describing output

```python
import glob
import h5py

path = sorted(glob.glob('data/processed/*.h5'))[0]
with h5py.File(path) as f:
    print(f['force/data'].shape, f['oil_thickness/data'].shape, f['sheet_thickness/data'].shape)
    print(dict(f['oil_thickness'].attrs))
```

Output:

```
(600, 8) (200, 2) (200, 2)
{'columns': array(['sensor_position', 'oil_value'], dtype=object), 'units': array(['mm', 'g/m^2'], dtype=object), 'n_measurements': 200,
 'n_raw_in_range': 411, 'n_nan_removed': 0, 'n_hampel_outliers': 2, 'n_positions_interpolated': 0,
 'max_sensor_position': 200, 'output_length': 200, 'hampel_window': 5, 'hampel_k': 3.0, 'value_quantization': 0.01}
```

Every group carries the parameter values used and the per-file cleaning statistics: a processed file documents itself even when separated from the code that made it.

## 4. Processing figures per modality

The figures on the [preprocessing pages](../preprocessing/index.md) are drawn by `rddac._preprocess.visualize`. Each `plot_*_processing` function takes the **raw** arrays and derives the processed view through the modality's own `process` function, so a figure is by construction what `rddac preprocess` does. Rendered on the small bundle:

```python
from rddac._preprocess import visualize
from rddac._preprocess.h5_access import open_raw

with open_raw(0, 'data') as raw:
    oil = raw['oil_thickness/data'][:]
visualize.plot_oil_processing(oil)          # one experiment: raw, Hampel outliers, fixed grid
```

<img src="../../images/preprocessing/oil_processing.png" width="700">

`plot_force_processing(list_of_raw_tables)` and `plot_sheet_processing(...)` overlay several experiments, here all 18 of the bundle:

<img src="../../images/preprocessing/force_processing.png" width="700">
<img src="../../images/preprocessing/sheet_processing.png" width="700">

## 5. Adjust parameters reproducibly

```bash
rddac preprocess --dump-config > my.toml   # complete defaults, ready to edit
rddac preprocess oil --config my.toml --overwrite
```

The first lines of the dumped configuration:

```toml
# rddac preprocess configuration: these are the defaults (the
# reference recipe). Edit values and pass the file via
#   rddac preprocess --config my.toml
# Publish it next to your code to make the variant reproducible.

[oil]
max_sensor_position = 200
output_length = 200
hampel_window = 5
hampel_k = 3.0
value_quantization = 0.01
```

Publishing `my.toml` next to your code makes the variant exactly reproducible: raw DOI + `rddac` version + TOML.

If a parameter is not enough, for example you want a different filter altogether, see [Custom processing](../preprocessing/custom.md): it shows how to replace one modality's step with your own code while the reference stages handle the rest.

## 6. The pointcloud stage

With the full release and simulations present, the same command cleans and aligns the scans (`z` → `(N, 3) float32`, `luminescence` → `(2000, 3200) uint8`); on first use it retrains the random-forest fin classifier from the bundled labels (one-time, ~30–90 min):

```bash
rddac preprocess pointcloud --workers 8
```

Without the simulations the stage is skipped with a notice (or fails with an actionable error when named explicitly). Its first step, the validity mask from the luminescence grid, needs only raw data and can be previewed on the small bundle with `visualize.plot_luminescence_processing(lumi_2d)`:

<img src="../../images/preprocessing/luminescence_processing_op10.png" width="900">

See [Point Clouds](../preprocessing/pointcloud.md) for the full pipeline and the held-out quality numbers.

## 7. Stream the processed layer

Two things change between the layers, nothing else:

| | raw | processed |
| --- | --- | --- |
| `data_dir=` | `data/` | `data/processed/` |
| `source=` (manifest) | `data/metadata.json` (found automatically) | `data/metadata.json`, **the same original file**, passed explicitly |

The views, field names and everything else stay identical; only the shapes reflect the processing:

```python
from pathlib import Path
from rddac.streaming import iter_view

DATA_DIR = Path('data')
for rec in iter_view('force-curve', data_dir=DATA_DIR / 'processed',
                     source=DATA_DIR / 'metadata.json', sim_ids=[0, 500]):
    print(rec['_sim_id'], rec['force_data'].shape)      # (600, 8) instead of the raw (n, 8)
```

Output:

```
0 (600, 8)
500 (600, 8)
```

Everything from the [streaming tutorial](streaming.md) (custom views, `export_to_numpy`, `RDDACDataset`) applies unchanged.

## Where to go next

- [Preprocessing overview](../preprocessing/index.md): schema, output rules, reproducibility model
- [Custom processing](../preprocessing/custom.md): replace a stage with your own algorithm
- [Consuming the processed layer](../preprocessing/index.md#consuming-the-processed-layer): the `data_dir=` switch in one place
- [Streaming and numpy export](streaming.md): views, exports and `RDDACDataset` at scale
