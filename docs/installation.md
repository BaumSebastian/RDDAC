# Installation

Requires Python 3.10 or newer. PyTorch is optional.

```bash
pip install rddac
```

Verify the install:

```python
import rddac
print(rddac.__version__)
```

## PyTorch

`RDDACDataset` and the rest of `rddac.pytorch` require PyTorch.

```bash
pip install rddac[torch]
```

This pulls the default PyTorch wheel from PyPI (CPU on Windows and macOS, CUDA 12.x on Linux). For a different build (CUDA, ROCm, MPS) install PyTorch first from the [PyTorch selector](https://pytorch.org/get-started/locally/), then `pip install rddac` without the extra. The package does not pin PyTorch, so an existing wheel is preserved.

## Preprocessing

The `pointcloud` stage of `rddac preprocess` (geometric cleaning, ICP alignment, random-forest fin classifier) needs scipy, scikit-learn and joblib; the `force`, `sheet` and `oil` stages run without them.

```bash
pip install rddac[preprocessing]
```

## Advanced

### Documentation build

```bash
pip install rddac[docs]
```

Then `mkdocs serve` from the repository root for a live preview at `http://127.0.0.1:8000`.

### Development install

```bash
git clone https://github.com/BaumSebastian/RDDAC.git
cd RDDAC
pip install -e .[dev,torch]
pre-commit install
```

`[dev]` adds `pytest`, `black`, `ruff`, `bumpver`, `pre-commit`, and the `[preprocessing]` extra so the whole test suite collects.

All runtime dependencies install automatically. This includes the [`ddacs`](https://ddacs.readthedocs.io) package, which provides the shared dataset machinery and the download of the matching simulations.
