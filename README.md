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

<div align="center">

![Measured point clouds after OP10 and OP20, colored by deviation from the matching DDACS simulation](https://raw.githubusercontent.com/BaumSebastian/RDDAC/main/docs/images/sim2real_sweep.gif)

*Measured point clouds of one experiment after deep drawing (OP10, left) and cutting (OP20, right), colored by the deviation from the matching DDACS simulation.*

</div>

**A large-scale experimental dataset of 9,000 physical deep-drawing and cutting experiments — the real-world counterpart to the [DDACS](https://ddacs.readthedocs.io) FEM simulations.** Each experiment forms a modified quadratic cup from DP600 dual-phase steel (deep drawing in OP10, cutting in OP20) and records press force signals, sheet-thickness and oil-film traverses, and high-resolution 3D laser scans of the part after each operation. Use it to quantify the simulation-to-reality gap, train models on real process data, or validate DDACS-trained surrogates against physical measurements.

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

The `rddac` package ships with the dataset and provides a Croissant native interface: one CLI for the download and the reference preprocessing, one Python module for access, and an optional PyTorch `IterableDataset` for training.

## Installation

```bash
pip install rddac
```

The PyTorch adapter is an optional extra. For hardware specific PyTorch builds (CUDA, ROCm, MPS), install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then install the extra:

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
```

## Preprocess the dataset

The published files are raw by design. `rddac preprocess` derives an ML-ready layer next to them (`./data/processed`, raw files untouched): forming-window force curves, cleaned oil-film and sheet-thickness profiles, and calibrated, simulation-aligned point clouds. Every parameter is adjustable via TOML and stamped into the output; the same Croissant views stream both layers.

```bash
rddac preprocess                   # all modalities (pointcloud needs the simulations)
rddac preprocess oil force sheet   # a subset, e.g. on the small bundle
```

See the [preprocessing documentation](https://rddac.readthedocs.io/en/latest/preprocessing/) for what each stage does and how to replace one with your own algorithm.

## Basic usage

```python
import rddac

with rddac.open_h5(0) as f:                    # one experiment by id
    force = f["force/data"][:]                 # (n, 8): time, load cells, temp, position, total force
    sheet = f["sheet_thickness/data"][:]       # (n, 2): sensor position, thickness
    z10   = f["pointcloud/op10/z"][:]          # (6400000,) flat scan buffer
```

The public surface mirrors the [`ddacs`](https://ddacs.readthedocs.io) package one to one — `load`, `add_view`, `open_h5`, `inspect_h5`, `streaming.iter_view` / `export_to_numpy` / `load_export`, and the PyTorch `IterableDataset` share names, signatures, and semantics. Code written against DDACS ports by swapping the import:

```python
# import ddacs as dataset_pkg                  # simulations
import rddac as dataset_pkg                    # real experiments

ds = dataset_pkg.load(data_dir="./data")
for record in dataset_pkg.streaming.iter_view("force-curve", data_dir="./data", dataset=ds):
    ...
```

See the [documentation](https://rddac.readthedocs.io) for the dataset reference (parameter space, HDF5 structure, Croissant manifest) and step-by-step tutorials from a first plot to PyTorch training.

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

## License

The dataset on DaRUS is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The `rddac` software is licensed under the MIT License — see [LICENSE](LICENSE).

Data files bundled with the package are **not** MIT: the fin labels (`rddac/_preprocess/labels/`, human annotations of the dataset), the scanner calibration (`calibration.json`) and the simulation parameter table (`sim_params.csv`) are data derived from RDDAC/DDACS and are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) like the dataset (see `rddac/_preprocess/labels/LICENSE`).
