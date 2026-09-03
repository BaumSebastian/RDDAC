# Loose HDF5 recipe

`rddac download --extract --remove-zip` unpacks each zip in place and deletes the archive on success. After that, the experiments sit as loose `.h5` files instead of zip members.

The companion notebook at [`notebooks/05_loose_h5.ipynb`](https://github.com/BaumSebastian/RDDAC/tree/main/notebooks/05_loose_h5.ipynb) reproduces every cell below.

Two paths work on this layout:

- **View-driven**: [`rddac.streaming.iter_view`](streaming.md) uses a unified index that recognises loose `.h5` files alongside zips, so the same view code you wrote against zips keeps working after extraction. This is the path for anyone who already has a view defined.
- **Manual `h5py.File`**: when you don't have (or don't need) a view, the pandas-plus-`h5py` recipe below gives you full access to every group and dataset in the file. That is what this recipe walks through.

## 1. Download with extract + remove

From a shell, fetch the small bundle into a throwaway directory so the project-local `./data` stays intact:

```bash
rddac download --small --extract --remove-zip --out /tmp/rddac_loose -y
```

`--extract` unzips each archive in place; `--remove-zip` deletes the zip afterwards. See the [CLI reference](../cli.md#rddac-download) for the rest of the flags.

## 2. Inspect the loose layout

`metadata.json` and `process_parameters.csv` sit at the bundle root; the extracted experiments sit alongside them as zero-padded `.h5` files.

```python
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

DATA_DIR = Path('/tmp/rddac_loose')

for entry in sorted(DATA_DIR.iterdir())[:6]:
    print(entry.name)
```

Output:

```
0000.h5
0500.h5
1002.h5
1500.h5
2000.h5
2500.h5
```

(plus the remaining sample experiments, `metadata.json`, and `process_parameters.csv`.)

## 3. Read `process_parameters.csv` with pandas

The CSV is the experiment index: one row per experiment, with every process parameter exposed as a named column. Filter it in pandas before touching any HDF5 file: IO scales with the surviving rows, not with the full {{ experiment_count() }}.

```python
params = pd.read_csv(DATA_DIR / 'process_parameters.csv')
print(f'rows: {len(params)}, columns: {list(params.columns)}')
print()
print(params.head(3).to_string())
```

Output:

```
rows: 9000, columns: ['index', 'experiment_id', 'category', 'geometry', 'blankholder_force', 'mean_punch_temp', 'oil_type', 'has_pointcloud', 'has_oil', 'split']

   index  experiment_id  category geometry  blankholder_force  mean_punch_temp oil_type  has_pointcloud  has_oil  split
0      0              1         0  concave                100             20.2   coarse            True     True    val
1      1              2         0  concave                100             20.3   coarse            True     True  train
2      2              3         0  concave                100             20.4   coarse            True     True    val
```

## 4. Iterate the loose files with `h5py`

Walk the (filtered) rows, build the zero-padded path, skip experiments whose `.h5` is missing locally, and open each one with `h5py.File`. Below: take the concave subset, then read the sheet-thickness range from every loose file that landed on disk. With the small bundle only the 9 concave sample experiments exist, so the loop yields 9 lines.

One raw-data detail: the traverse tables contain occasional **sensor error values** (large negative numbers where the sensor lost contact). The published files carry the data exactly as recorded, so filter to plausible values before taking statistics; outlier cleaning is what the optional [preprocessing](../preprocessing/sheet.md) step (`rddac preprocess sheet oil`) does.

```python
concave = params.query("geometry == 'concave'")
print(f'concave experiments in CSV: {len(concave):>5d} of {len(params)}')

found = 0
for _, row in concave.iterrows():
    h5_path = DATA_DIR / f"{row['index']:04d}.h5"      # index 42 -> 0042.h5
    if not h5_path.is_file():
        continue
    with h5py.File(h5_path, 'r') as f:
        thickness = f['sheet_thickness/data'][:, 1]     # um
    valid = thickness[thickness > 0]                    # drop sensor error values
    print(f"  experiment {row['index']:>4d}  thickness in range of "
          f"{valid.min():.1f} um - {valid.max():.1f} um  "
          f"(valid samples: {len(valid)}/{len(thickness)})")
    found += 1
print(f'\nopened {found} loose h5 file(s)')
```

Output:

```
concave experiments in CSV:  4500 of 9000

  experiment    0  thickness in range of 985.9 um - 1000.4 um  (valid samples: 206/208)
  experiment  500  thickness in range of 950.1 um - 988.7 um  (valid samples: 206/209)
  experiment 1002  thickness in range of 983.2 um - 998.4 um  (valid samples: 206/208)
  experiment 1500  thickness in range of 974.9 um - 995.9 um  (valid samples: 206/208)
  experiment 2000  thickness in range of 986.4 um - 1000.7 um  (valid samples: 206/208)
  experiment 2500  thickness in range of 989.7 um - 1005.5 um  (valid samples: 206/208)
  experiment 3000  thickness in range of 985.9 um - 1004.6 um  (valid samples: 206/208)
  experiment 3500  thickness in range of 959.6 um - 1003.7 um  (valid samples: 206/208)
  experiment 4000  thickness in range of 967.5 um - 1002.0 um  (valid samples: 206/208)

opened 9 loose h5 file(s)
```

Remember the two availability flags: guard reads of `oil_thickness/` with `row['has_oil']` and reads of `pointcloud/` with `row['has_pointcloud']`, otherwise the missing group raises a `KeyError` (see [Missing measurements](../dataset.md#missing-measurements)).

## Where to go next

- [Build your own view](views.md) and [PyTorch training](pytorch.md) cover the zipped-bundle path that the Croissant API supports.
- [HDF5 structure reference](../hdf5-structure.md) lists every dataset and attribute available inside each loose `.h5`.
- [Streaming and numpy export](streaming.md) shows how `rddac.streaming.iter_view` reads the same loose layout view-driven, without writing the per-record `h5py.File` loop yourself.
