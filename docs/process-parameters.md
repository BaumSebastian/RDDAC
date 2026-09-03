# Process Parameters

Each row in `process_parameters.csv` describes one experiment. The `index` column is the same number as the HDF5 filename inside the zips (zero-padded to four digits, so `index=42` -> `0042.h5`), which makes the CSV the single point of entry for filtering before any HDF5 is opened.

## Columns

{{ process_parameters_table() }}

## Categories

The `category` column (0-17) encodes the geometry x blankholder force x oil type combination. Each category holds 500 repetitions in one contiguous 500-block of experiment ids; `experiment_id` is the repetition index (1-500) inside the category.

| `category` | Geometry | Blankholder force [kN] | Oil type | Experiment ids |
|-----------:|----------|-----------------------:|----------|----------------|
| 0 | concave | 100 | coarse | 0000-0499 |
| 1 | concave | 100 | medium | 1000-1499 |
| 2 | concave | 100 | fine | 0500-0999 |
| 3 | concave | 300 | coarse | 1500-1999 |
| 4 | concave | 300 | medium | 2500-2999 |
| 5 | concave | 300 | fine | 2000-2499 |
| 6 | concave | 500 | coarse | 3000-3499 |
| 7 | concave | 500 | medium | 4000-4499 |
| 8 | concave | 500 | fine | 3500-3999 |
| 9 | convex | 100 | coarse | 4500-4999 |
| 10 | convex | 100 | medium | 5500-5999 |
| 11 | convex | 100 | fine | 5000-5499 |
| 12 | convex | 300 | coarse | 6000-6499 |
| 13 | convex | 300 | medium | 7000-7499 |
| 14 | convex | 300 | fine | 6500-6999 |
| 15 | convex | 500 | coarse | 7500-7999 |
| 16 | convex | 500 | medium | 8500-8999 |
| 17 | convex | 500 | fine | 8000-8499 |

The two `geometry` values split the id range in half: `concave` covers 0000-4499 and `convex` covers 4500-8999.

<img src="../images/agg_oil_heatmap.png" width="700">

*Oil film measurements aggregated over all parts of one category, repetitions of the same nominal parameters, overlaid.*

## Split

The `split` column carries the recommended `train` / `val` / `test` partition: 80 / 10 / 10 (7,200 / 900 / 900 experiments), drawn deterministically with seed 42 so that every category is represented proportionally in each split.

## Missing-measurement flags

`has_pointcloud` and `has_oil` flag whether the `pointcloud/` and `oil_thickness/` HDF5 groups exist for the experiment ({{ missing_pointcloud() }} and {{ missing_oil() }} experiments are missing them, respectively; see [Dataset overview](dataset.md#missing-measurements)). Views that read those groups must filter on the flags, otherwise the read raises a `KeyError` for the affected experiments.

## Sample

```
index,experiment_id,category,geometry,blankholder_force,mean_punch_temp,oil_type,has_pointcloud,has_oil,split
0,1,0,concave,100,20.2,coarse,True,True,val
1,2,0,concave,100,20.3,coarse,True,True,train
2,3,0,concave,100,20.4,coarse,True,True,val
```

## Filtering recipe

```python
import pandas as pd

df = pd.read_csv("./data/process_parameters.csv")

concave      = df[df["geometry"] == "concave"]
high_force   = df[df["blankholder_force"] == 500]
with_oil     = df[df["has_oil"]]
one_category = df[df["category"] == 2]
```

The same predicate is accepted by `RDDACDataset(where=...)` and `rddac.streaming.iter_view(where=...)`, which keeps IO scaled to the surviving rows rather than the full {{ experiment_count() }}:

```python
from rddac.pytorch import RDDACDataset

ds = RDDACDataset(
    view="thickness",
    data_dir="./data",
    where=lambda row: row["has_oil"] and row["split"] == "train",
)
```
