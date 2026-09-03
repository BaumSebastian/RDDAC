<div align="center">
  <img src="https://raw.githubusercontent.com/BaumSebastian/RDDAC/main/docs/images/icon/icon.png" width="150"/>
  <h1>Real Deep Drawing and Cutting (RDDAC) Dataset</h1>
</div>

[![Code License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/rddac/badge/?version=latest)](https://rddac.readthedocs.io)
[![DaRUS Repository](https://img.shields.io/badge/repository-DaRUS-green.svg)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5589)
[![DOI](https://img.shields.io/badge/DOI-10.18419%2FDARUS--5589-blue.svg)](https://doi.org/10.18419/DARUS-5589)
[![Paper](https://img.shields.io/badge/paper-Trans%20Indian%20Inst%20Met-red.svg)](https://doi.org/10.1007/s12666-026-03870-5)

<div align="center">

![Measured point clouds after OP10 and OP20, colored by deviation from the matching DDACS simulation](https://raw.githubusercontent.com/BaumSebastian/RDDAC/main/docs/images/sim2real_sweep.gif)

*Measured point clouds of one experiment after deep drawing (OP10, left) and cutting (OP20, right), colored by the deviation from the matching DDACS simulation.*

</div>

**A large-scale experimental dataset of 9,000 physical deep-drawing and cutting experiments, the real-world counterpart to the [DDACS](https://ddacs.readthedocs.io) FEM simulations.** Each experiment forms a modified quadratic cup from DP600 dual-phase steel (deep drawing in OP10, cutting in OP20) and records press force signals, sheet-thickness and oil-film traverses, and high-resolution 3D laser scans of the part after each operation. Use it to quantify the simulation-to-reality gap, train models on real process data, or validate DDACS-trained surrogates against physical measurements.

|  |  |
|---|---|
| **Experiments** | 9,000 |
| **Total size** | ~87 GB (HDF5, lossless) |
| **Process steps per experiment** | 2 (OP10 deep drawing, OP20 cutting) |
| **Parameter space** | 2 geometries x 3 blankholder forces x 3 oil types (18 categories) |
| **Repetitions** | up to 500 per category |
| **Train / val / test** | 7,200 / 900 / 900 (predefined, seed 42) |
| **Matching simulations** | DDACS `rddac.zip` (~9 GB), fetched by `rddac download` |

**[Documentation](https://rddac.readthedocs.io)** · **[Dataset DOI](https://doi.org/10.18419/DARUS-5589)** · **[Paper](https://doi.org/10.1007/s12666-026-03870-5)**

Try the ~174 MB teaser (18 experiments, manifest, parameter table, runnable tutorials): **[Kaggle](https://www.kaggle.com/datasets/baumsebastian/rddac-teaser)** · **[Hugging Face](https://huggingface.co/datasets/BaumSebastian/rddac-teaser)** · **[Zenodo](https://zenodo.org/records/21274093)**

A Croissant-native Python package for accessing the [RDDAC Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5589) ships with this repo: one CLI for the download and the reference preprocessing, one Python module for access, torch-free streaming and numpy export, plotting helpers, and an optional PyTorch `IterableDataset` for training. Its public surface mirrors the [`ddacs`](https://ddacs.readthedocs.io) package one to one, so code written for the simulations ports by swapping the import.

## Table of Contents

- [What's new in 1.1](#whats-new-in-11)
- [Installation](#installation)
- [Download the dataset](#download-the-dataset)
- [Preprocess the dataset](#preprocess-the-dataset)
- [Basic usage](#basic-usage)
- [PyTorch integration](#pytorch-integration)
- [Tutorials](#tutorials)
- [Version compatibility](#version-compatibility)
- [Citation](#citation)
- [Development](#development)
- [License](#license)

## What's new in 1.1

1.1 adds the reference preprocessing. The published files stay raw by design; `rddac preprocess` derives an ML-ready layer next to them:

- `force`, `sheet`, `oil`: fixed-shape, cleaned tables (forming-window force curves, error-masked thickness and dropout-free oil-film profiles).
- `pointcloud`: calibrated scans, cleaned of fins by a random-forest classifier, aligned to the matching DDACS simulation (needs the `[preprocessing]` extra and the simulations).
- Every parameter is adjustable via TOML and stamped into the output; the same Croissant views stream both layers.

See [Preprocessing](https://rddac.readthedocs.io/en/latest/preprocessing/) for what each stage does and the [changelog](CHANGELOG.md) for the details.

## Installation

```bash
pip install rddac
```

The PyTorch adapter is an optional extra. For hardware-specific PyTorch builds (CUDA, ROCm, MPS), install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then install the extra:

```bash
pip install 'rddac[torch]'
```

The `pointcloud` stage of `rddac preprocess` needs scipy and scikit-learn:

```bash
pip install 'rddac[preprocessing]'
```

## Download the dataset

```bash
# Small sample bundle (~174 MB): manifest, CSV, and one experiment per category.
rddac download --small -y

# Full release (~87 GB), including the matching DDACS simulations (~9 GB).
rddac download

# Real measurements only (skip the simulations).
rddac download --no-sim

# Show available versions on DaRUS.
rddac info
```

Files land in `./data` by default. The same path is the default for `rddac.load(data_dir=...)`, `rddac preprocess --data-dir` and `RDDACDataset(data_dir=...)`, so no further configuration is needed.

All options (`--files`, `--out`, `--extract`, `--remove-zip`, `--quiet`, the global `--token`) are documented in the [CLI reference](https://rddac.readthedocs.io/en/latest/cli/).

By default zip files are kept on disk and are *not* extracted; `mlcroissant` reads HDF5 members in place. Pass `--extract --remove-zip` to switch to a loose-HDF5 layout instead; see the [Loose HDF5 recipe](https://rddac.readthedocs.io/en/latest/tutorials/loose-h5/).

## Preprocess the dataset

```bash
rddac preprocess                   # all modalities (pointcloud needs the simulations)
rddac preprocess oil force sheet   # a subset, e.g. on the small bundle
```

Output lands in `./data/processed`, raw files are never modified, and re-runs only fill in what is missing. All options (`--ids`, `--split`, `--workers`, `--overwrite`, `--config`) are documented in the [CLI reference](https://rddac.readthedocs.io/en/latest/cli/#rddac-preprocess); what each stage does and how to replace one with your own algorithm is in the [preprocessing documentation](https://rddac.readthedocs.io/en/latest/preprocessing/).

## Basic usage

`rddac.load` parses the Croissant manifest; `rddac.open_h5` opens a single experiment in memory and returns an `h5py.File`.

```python
import rddac

# Load the dataset manifest. Lists every published RecordSet.
ds = rddac.load(data_dir="./data")
print([rs.id for rs in ds.metadata.record_sets])

# Open one experiment by id.
with rddac.open_h5(0, data_dir="./data") as f:
    force = f["force/data"][:]                 # (n, 8): time, load cells, temp, position, total force
    sheet = f["sheet_thickness/data"][:]       # (n, 2): sensor position, thickness
    z10 = f["pointcloud/op10/z"][:]            # (6400000,) flat scan buffer
```

The same views stream the processed layer: pass `data_dir="./data/processed"` and `source="./data/metadata.json"` to `rddac.streaming.iter_view`. For custom RecordSets see [Build your own view](https://rddac.readthedocs.io/en/latest/tutorials/views/); for scans, point clouds, force curves and traverses see [Visualization](https://rddac.readthedocs.io/en/latest/tutorials/visualization/).

## PyTorch integration

`RDDACDataset` is a `torch.utils.data.IterableDataset` over a Croissant view. It builds an `id -> local zip` index at construction time and silently skips experiments whose zip is missing, so partial downloads stream fine. Raw tables vary in length per experiment, so batch the processed layer, where every record has a fixed shape:

```python
from rddac.pytorch import RDDACDataset
from torch.utils.data import DataLoader

ds = RDDACDataset(view="force-curve", data_dir="./data/processed", source="./data/metadata.json")
loader = DataLoader(ds, batch_size=16, num_workers=0)

for batch in loader:
    force = batch["force_data"]                # (16, 600, 8) after `rddac preprocess force`
    # ... training step ...
    break
```

For filtering, train / val / test splits, shuffling, and the partial-download story, see [PyTorch training](https://rddac.readthedocs.io/en/latest/tutorials/pytorch/).

## Tutorials

The tutorials walk through the package end to end. Each one is published on Read the Docs as a [tutorial page](https://rddac.readthedocs.io/en/latest/tutorials/) and shipped as an executable notebook under [`notebooks/`](./notebooks). See [`notebooks/README.md`](./notebooks/README.md) for prerequisites and run instructions.

## Version compatibility

The `rddac` package major version tracks the DaRUS dataset major version. The pairing is enforced by the Croissant manifest bundled with each release: a mismatched package version will fail to resolve the field map.

| Package | DaRUS dataset |
|---------|---------------|
| `rddac 1.x` | [v1.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5589&version=1.0) and any future v1.x updates (current) |

Pin the package major to the dataset major you target, for example `pip install 'rddac~=1.0'` to stay on the v1 line.

## Citation

```bibtex
@dataset{baum2026rddac,
  title={Real Deep Drawing and Cutting Dataset},
  author={Baum, Sebastian and Heinzelmann, Pascal},
  year={2026},
  publisher={DaRUS},
  doi={10.18419/DARUS-5589}
}

@article{baum2026deviation,
  title={Statistical Analysis of Simulation to Reality Deviation in Deep Drawing with a Benchmark Dataset},
  author={Baum, Sebastian and Heinzelmann, Pascal and Clau{\ss}, P. and others},
  journal={Transactions of the Indian Institute of Metals},
  volume={79},
  pages={176},
  year={2026},
  doi={10.1007/s12666-026-03870-5}
}
```

## Development

```bash
git clone https://github.com/BaumSebastian/RDDAC.git
cd RDDAC
pip install -e ".[dev,torch]"
pre-commit install   # set up code formatting hooks
pytest               # run the full test suite (PyTorch tests skip without the torch extra)
```

## License

The dataset on DaRUS is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The `rddac` software is licensed under the MIT License, see [LICENSE](LICENSE).

Data files bundled with the package are **not** MIT: the fin labels (`rddac/_preprocess/labels/`, human annotations of the dataset), the scanner calibration (`calibration.json`) and the simulation parameter table (`sim_params.csv`) are data derived from RDDAC/DDACS and are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) like the dataset (see `rddac/_preprocess/labels/LICENSE`).
