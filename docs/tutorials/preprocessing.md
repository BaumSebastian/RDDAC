# 7. Preprocessing

This tutorial runs the reference preprocessing on the small bundle, inspects the self-describing output, and shows how to adjust parameters reproducibly. The complete stage documentation lives in the [Preprocessing section](../preprocessing/index.md).

The companion notebook at [`notebooks/07_preprocessing.ipynb`](https://github.com/BaumSebastian/RDDAC/tree/main/notebooks/07_preprocessing.ipynb) reproduces every cell below.

## 1. Install and download

```bash
pip install 'rddac[preprocessing]'
rddac download --small -y          # ~174 MB sample bundle into ./data
```

The traverse and force stages run on the small bundle as-is. The `pointcloud` stage additionally needs the full release plus the DDACS simulations (`rddac download`, without `--no-sim`) — section 5 shows how it degrades gracefully without them.

## 2. Run the reference preprocessing

```bash
rddac preprocess oil force sheet --data-dir ./data -q
```

Output lands in `./data/processed/` as loose `<id>.h5` files plus a generated `metadata.json` describing the processed layout. Raw files are never modified.

## 3. Inspect the self-describing output

```python
import glob
import h5py

path = sorted(glob.glob('data/processed/*.h5'))[0]
with h5py.File(path) as f:
    print(f['force/data'].shape, f['oil_thickness/data'].shape, f['sheet_thickness/data'].shape)
    print(dict(f['oil_thickness'].attrs))
```

Every group carries the parameter values used and the per-file cleaning statistics — a processed file documents itself even when separated from the code that made it.

## 4. Adjust parameters reproducibly

```bash
rddac preprocess --dump-config > my.toml   # complete defaults, ready to edit
rddac preprocess oil --config my.toml --overwrite
```

Publishing `my.toml` next to your code makes the variant exactly reproducible: raw DOI + `rddac` version + TOML.

## 5. The pointcloud stage

With the full release and simulations present, the same command cleans and aligns the scans (`z` → `(N, 3) float32`, `luminescence` → `(2000, 3200) uint8`) — on first use it retrains the fin classifier from the bundled labels (one-time, ~30–90 min):

```bash
rddac preprocess pointcloud --workers 8
```

See [Point Clouds](../preprocessing/pointcloud.md) for the pipeline and the held-out quality numbers.

## Where to go next

- [Preprocessing overview](../preprocessing/index.md) — schema, output rules, reproducibility model
- [Custom processing](../preprocessing/custom.md) — replace a stage with your own algorithm
- [Streaming and numpy export](streaming.md) — consume the processed layer at scale
