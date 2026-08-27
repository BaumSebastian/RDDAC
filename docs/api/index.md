# API Reference

The RDDAC package exposes a small set of public functions, all importable from the top level `rddac` module. The surface mirrors the `ddacs` package one to one, so DDACS code ports by swapping the import.

## Modules

- **[Croissant](croissant.md)**: `load` and `add_view`. Loads the Croissant manifest and adds custom RecordSets.
- **[HDF5](h5.md)**: `open_h5`, `inspect_h5`. Reads a single experiment by id and prints the HDF5 hierarchy.
- **[Streaming](streaming.md)**: `iter_view`, `export_to_numpy`, `export_to_numpy_per_sim`, `load_export`. Torch-free iteration and numpy materialisation.
- **[PyTorch](pytorch.md)**: `RDDACDataset`. Streaming `IterableDataset` over a Croissant view. Requires the `[torch]` extra.
- **[Visualization](visualization.md)**: `plot_scan`, `scan_to_pointcloud`, `plot_point_cloud`, `plot_force`, `plot_traverse`. Matplotlib plotting helpers.
- **[Preprocessing](preprocessing.md)**: the `rddac preprocess` CLI and the internal `rddac._preprocess` stages (`oil`, `force`, `sheet`, `pointcloud`) plus the processing-figure helpers. The `pointcloud` stage requires the `[preprocessing]` extra.

## Quick import

```python
import rddac

# Croissant entry points
ds = rddac.load(data_dir="./data")
rddac.add_view(ds, "my-view", fields={"force": "force_data"})

# Single-experiment HDF5 access
with rddac.open_h5(0) as f:
    rddac.inspect_h5(f)

# Streaming (no PyTorch required)
for rec in rddac.streaming.iter_view("force-curve", data_dir="./data", dataset=ds):
    ...

# Visualization
ax, cbar = rddac.plot_scan(z_buffer)

# PyTorch (requires `pip install rddac[torch]`)
from rddac.pytorch import RDDACDataset
loader = RDDACDataset(view="force-curve")
```

Preprocessing has no Python entry point on purpose: `rddac preprocess` is the supported interface (see [Preprocessing](../preprocessing/index.md)); the same views then stream the processed layer via `data_dir=`.
