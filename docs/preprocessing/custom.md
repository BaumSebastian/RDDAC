# Custom Processing

`rddac preprocess` is the **reference implementation** of the processed layout, not the only allowed one. The raw dataset is immutable and readable through the public API, so anyone with a better algorithm can produce their own processed layer: the contract is the **schema of the output files**, not a Python interface.

## Three levels of adjustment

| Level | When | How |
| --- | --- | --- |
| 1. Change parameters | the algorithm is fine, a threshold or window is not | `--config my.toml`, no code |
| 2. Replace one step | you want a different filter for one modality, our stages for the rest | write that modality's group yourself, let `rddac preprocess` do the others |
| 3. Replace everything | a different pipeline altogether | produce files matching the schema in your own directory |

### Level 1: parameters

Every threshold of every stage is a TOML key (`rddac preprocess --dump-config` prints them all with their defaults). A stricter oil-film cleaning, for example:

```toml
# my.toml
[oil]
hampel_window = 10   # wider neighbourhood (mm)
hampel_k = 2.5       # flag outliers earlier
```

```bash
rddac preprocess oil --config my.toml --overwrite
```

The values used are stamped into `oil_thickness` attributes of every output file (`hampel_window`, `hampel_k`, ...), and the docs figure can be redrawn for the variant with the same overrides: `python -m rddac._preprocess.visualize oil --id 0 --config my.toml --out .`.

### Level 2: replace one step, keep the rest

Say you prefer a plain running-median filter to the Hampel filter for the oil film. Write the `oil_thickness` group yourself and leave the other modalities to the reference stages. The output directory is shared: `rddac preprocess` appends to existing files and never touches groups it did not create unless `--overwrite` is given.

```python
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import medfilt

import rddac
from rddac._preprocess.h5_access import available_ids

DATA_DIR, OUT_DIR = Path("./data"), Path("./data/processed")
OUT_DIR.mkdir(exist_ok=True)

def median_oil(raw: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Running median instead of Hampel: truncate, drop NaN, filter, average duplicates, fill the grid."""
    df = pd.DataFrame(raw, columns=["pos", "val"]).dropna()
    df["pos"] = df["pos"].round().astype(int)
    df = df[df["pos"] < 200].sort_values("pos")
    df["val"] = medfilt(df["val"].to_numpy(), kernel)
    grid = pd.DataFrame({"pos": range(200)}).merge(df.groupby("pos", as_index=False)["val"].mean(), how="left")
    grid["val"] = grid["val"].interpolate().bfill().ffill()
    return grid.to_numpy(dtype=np.float32)

for exp_id in sorted(available_ids(DATA_DIR)):
    with rddac.open_h5(exp_id, data_dir=DATA_DIR) as raw:
        cleaned = median_oil(raw["oil_thickness/data"][:])
    with h5py.File(OUT_DIR / f"{exp_id:04d}.h5", "a") as out:
        if "oil_thickness" in out:
            del out["oil_thickness"]
        g = out.create_group("oil_thickness")
        g.create_dataset("data", data=cleaned)
        g.attrs["columns"], g.attrs["units"] = ["sensor_position", "oil_value"], ["mm", "g/m^2"]
        g.attrs["producer"], g.attrs["median_kernel"] = "median_oil v1", 7     # honest provenance
```

```bash
rddac preprocess force sheet        # the reference stages fill in the other groups
```

The result is one processed layer: your oil profiles next to our force and sheet tables, streamable through the same views (`iter_view("oil-thickness", data_dir="./data/processed", source="./data/metadata.json")`). To compare against the reference, run `rddac preprocess oil --out ./data/reference` and plot both.

### Level 3: the contract

Read raw experiments with the public API, run your algorithm, and write files matching the [processed schema](index.md#what-each-modality-does) into your own output directory. The contract is the schema, not a Python interface:

```python
import h5py
import rddac

for exp_id in range(9000):
    with rddac.open_h5(exp_id, data_dir="./data") as raw:      # immutable input
        oil_raw = raw["oil_thickness/data"][:]

    cleaned = my_better_oil(oil_raw)                            # your algorithm -> (200, 2) float32

    with h5py.File(f"./my_processed/{exp_id:04d}.h5", "a") as out:
        group = out.create_group("oil_thickness")
        group.create_dataset("data", data=cleaned)
        group.attrs["columns"] = ["sensor_position", "oil_value"]
        group.attrs["producer"] = "my_better_oil v1"            # honest provenance
```

Files that match the schema are consumable by the same downstream tooling as ours. Because raw data is canonical and pinned by the published Croissant manifest, *anyone* can reproduce your processed layer from your code: replacing an algorithm always means running it yourself; there is nothing server-side to swap.

Two conventions keep replacements honest:

- **Never write into the raw directory**: the published checksums are the dataset's identity.
- **Stamp what you did** into the group attributes (our stages record every parameter used), so a processed file documents itself even when separated from the code.

## Internal reference

!!! warning "Internal API, may change without notice"
    The implementation lives in the private `rddac._preprocess` package. It is CLI-only by design; the signatures below are shown for orientation (e.g. for a notebook walking through the pipeline), not as a stable interface.

::: rddac._preprocess.oil.process
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.runner.run
    options:
      show_signature: true
      show_signature_annotations: true
