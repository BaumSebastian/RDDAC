# RDDAC notebooks

Six end-to-end Jupyter notebooks that companion the [online documentation](https://rddac.readthedocs.io). Each notebook is self-contained: it opens with a Walkthrough list and the Assumptions it relies on, then walks through the topic step by step. Reading top to bottom is the intended flow.

RDDAC is the experimental counterpart to the [DDACS](https://ddacs.readthedocs.io) simulation dataset, and the `rddac` package mirrors the `ddacs` API 1:1 — these notebooks mirror the DDACS notebook series the same way.

## Prerequisites

```bash
# Install the package + the PyTorch extra (needed for 03_pytorch.ipynb).
pip install 'rddac[torch]'

# Fetch the small sample bundle once. Writes metadata.json,
# process_parameters.csv, and one sample zip with 18 experiments
# (one per category) into ./data.
rddac download --small -y
```

The full release (`rddac download`, ~87 GB) additionally fetches the matching DDACS simulations into `./data/simulation/`; the notebooks only need the small bundle. Notebooks 05 and 06 write throwaway artifacts into the system temp directory and clean up after themselves, so the data directory stays untouched.

## Run

From the repository root:

```bash
jupyter lab notebooks/
```

Inside each notebook the data directory is hard-coded as `DATA_DIR = '../data'` because the notebooks live in `notebooks/` while the data sits at the repo root (adjust to `'../data'` if you downloaded with the default `rddac download` output directory).

## Index

| Notebook | What it covers |
|----------|----------------|
| [`01_getting_started.ipynb`](01_getting_started.ipynb) | Install, download, `rddac.load`, iterate one record of the experiment index, open one HDF5 file with `rddac.open_h5` + `inspect_h5`, render the OP10 laser scan as a height image. |
| [`02_views.ipynb`](02_views.ipynb) | The `field-map` lookup table, appending a custom `RecordSet` with `rddac.add_view` (whole fields, index slicing, `process-parameters` CSV columns), inspecting the resolved transforms, streaming with `rddac.streaming.iter_view`. |
| [`03_pytorch.ipynb`](03_pytorch.ipynb) | Build an `RDDACDataset`, stream ragged records at `batch_size=1`, batch properly via a fixed-shape custom view + `dataset=`, filter by manifest columns and `sim_ids`, canonical train/val/test splits, shuffle with `set_epoch`, the metadata-column limitation and its workarounds, and a minimal training-loop skeleton. |
| [`04_visualization.ipynb`](04_visualization.ipynb) | OP10 vs OP20 height scans with `plot_scan`, the luminescence channel, `scan_to_pointcloud` + `plot_point_cloud`, press signals with `plot_force`, and the sheet-thickness / oil-film traverses with `plot_traverse` (including masking the raw sensor spikes). |
| [`05_loose_h5.ipynb`](05_loose_h5.ipynb) | The post-`--extract --remove-zip` workflow: inspect the loose layout, filter `process_parameters.csv` with pandas, open loose `.h5` files directly with `h5py`, and stream the loose layout with `rddac.streaming.iter_view`. |
| [`06_streaming.ipynb`](06_streaming.ipynb) | `rddac.streaming.iter_view` in depth (`where=`, `sim_ids=`, `dataset=`), `export_to_numpy` with `transforms` + `record_transform` on a fixed-shape scan-profile view, `load_export` + `by_sim_id`, back-to-back timing of the streaming vs memmap paths, and `export_to_numpy_per_sim` for ragged fields. |

If any of these fail to run end to end after `rddac download --small`, please open an [issue](https://github.com/BaumSebastian/RDDAC/issues) so we can fix the discrepancy with the docs.
