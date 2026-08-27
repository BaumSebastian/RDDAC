# Preprocessing

The published dataset is raw by design: checksum-pinned files carrying the measurements exactly as recorded. Preprocessing is an **optional, versioned layer on top**: `rddac preprocess` reads the raw experiments and writes NEW files into a separate directory, so the published [Croissant manifest](../croissant.md) stays valid and the raw data remains the citable source of truth.

> **Reproducibility model**: raw dataset DOI + `rddac` version (+ your config TOML, if any) → byte-identical processed dataset.

## Quickstart

```bash
pip install 'rddac[preprocessing]'        # scipy, scikit-learn, joblib for the pointcloud stage
rddac preprocess                          # all modalities -> <data-dir>/processed
rddac preprocess oil force --ids 0-999    # subset of modalities and experiments
```

## What each modality does

| Modality | Raw → processed | Processing |
| --- | --- | --- |
| [`force`](force.md) | ~(1140, 8) → `(600, 8) float32` | forming-window trim, rest-offset removal, quantization, all 8 columns kept |
| [`sheet`](sheet.md) | ~(208, 2) → `(200, 2) float32` | tail selection, position normalization, error codes → `NaN` |
| [`oil`](oil.md) | ~(420, 2) → `(200, 2) float32` | dropout removal, NaN-robust Hampel filter, grid interpolation |
| [`pointcloud`](pointcloud.md) | scan grids → `z (N, 3) float32` + `luminescence (2000, 3200) uint8` | calibration, geometric outlier stages, ICP alignment to the matched DDACS simulation, random-forest fin cleaner |

Every processed group is **self-describing**: the parameter values used and the per-file cleaning statistics are stamped into its HDF5 attributes, so a file separated from the code that made it still documents itself.

## Adjusting parameters

```bash
rddac preprocess --dump-config > my.toml  # complete defaults, ready to edit
$EDITOR my.toml
rddac preprocess --config my.toml         # your variant, reproducibly
```

The defaults are the citable reference recipe. A variant is reproduced exactly by publishing the TOML next to your code.

## Output directory rules

- Default output is `<data-dir>/processed`; re-runs **append**: existing modality groups are kept unless `--overwrite` is given.
- Raw files are never modified. The CLI refuses output locations that could shadow or mutate the raw dataset (the raw directory itself, or any directory that looks like one); a `.rddac-processed` marker identifies valid output directories.

## Consuming the processed layer

No second manifest is generated. The Croissant manifest downloaded by `rddac download` (`./data/metadata.json`, also part of `--small`) is the single source of truth for both layers: the processed files keep the raw HDF5 paths (`force/data`, `oil_thickness/data`, ...), so every view in it works on the processed directory as well. Switching layers changes **only** `data_dir=`. The manifest stays the original one: pass `source=` explicitly, because no `metadata.json` exists inside the processed directory (without `source=`, `rddac` would fall back to fetching the manifest from DaRUS):

```python
from pathlib import Path
from rddac.streaming import iter_view

DATA_DIR = Path("./data")
MANIFEST = DATA_DIR / "metadata.json"                      # downloaded with the dataset

raw = iter_view("force-curve", data_dir=DATA_DIR, sim_ids=[0])                        # (n, 8) raw
proc = iter_view("force-curve", data_dir=DATA_DIR / "processed", source=MANIFEST, sim_ids=[0])  # (600, 8) processed
```

The processed directory holds loose `.h5` files, which `rddac.streaming` and `RDDACDataset` read directly (faster than zip members, no `BytesIO` round trip). As with any loose layout, `mlcroissant`'s own `records()` walk is not available, see the [loose HDF5 recipe](../tutorials/loose-h5.md).
