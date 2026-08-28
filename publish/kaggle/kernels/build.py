#!/usr/bin/env python3
"""Generate Kaggle Notebook (kernel) versions of the RDDAC tutorials.

The repo notebooks in notebooks/ are the single source of truth; the only
adaptation for Kaggle is a setup cell that `pip install`s rddac and fetches the
~174 MB sample with `rddac download --small` (Kaggle auto-extracts uploaded
zips, so reading the attached dataset would need reshaping — downloading is
simpler and gives the exact layout rddac expects). The dataset is still
attached so the kernels appear on the dataset's Code tab.

Output: publish/.staging/kernels/<slug>/{notebook, kernel-metadata.json}.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_DIR = REPO / "notebooks"
OUT = REPO / "publish" / ".staging" / "kernels"

OWNER = "baumsebastian"
DATASET = f"{OWNER}/rddac-teaser"
DATA_DIR = "/kaggle/working/data"


def _slug(title: str) -> str:
    # Kaggle derives the kernel slug from the title, and the id slug MUST equal
    # it ("your kernel title does not resolve to a specific id" otherwise).
    s = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return re.sub(r"-+", "-", s).strip("-")


# file, title, pip target  (notebook 05 builds its own loose dir from the
# sample zip via tempfile, so all use the same std fetch)
NOTEBOOKS = [
    ("01_getting_started.ipynb", "RDDAC 01 Getting Started", "rddac"),
    ("02_views.ipynb", "RDDAC 02 Build Your Own View", "rddac"),
    ("03_pytorch.ipynb", "RDDAC 03 PyTorch Training", "rddac[torch]"),
    ("04_visualization.ipynb", "RDDAC 04 Visualization", "rddac"),
    ("05_loose_h5.ipynb", "RDDAC 05 Loose HDF5", "rddac"),
    # retitled: the original slug (…-numpy-export) was tombstoned by a failed create
    ("06_streaming.ipynb", "RDDAC 06 Streaming and Export", "rddac"),
    ("07_preprocessing.ipynb", "RDDAC 07 Preprocessing", "rddac[preprocessing]"),
]


_CTA = """\
---

## ⬇️ Get the full dataset — ~87 GB

This notebook ran on a **~174 MB sample** (18 of 9,000 experiments). The complete
**RDDAC** dataset — **9,000 physical experiments, ~87 GB of lossless HDF5**, with
the predefined train / validation / test split — is hosted on DaRUS with a
citable DOI:

### ➡️ [doi.org/10.18419/DARUS-5589](https://doi.org/10.18419/DARUS-5589)

Everything above scales to the full release unchanged — just fetch it with the package
(this also brings the matching DDACS FEM simulations for sim-to-real comparison):

```bash
pip install rddac
rddac download        # full ~87 GB release + ~9 GB matching simulations
```

Docs: https://rddac.readthedocs.io · Package: https://pypi.org/project/rddac · Source: https://github.com/BaumSebastian/RDDAC
"""


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def adapt(nb: dict, pip_target: str) -> dict:
    """Kernel = two quiet setup cells (pip install, small-data download) on top
    of the unmodified repo notebook (paths mapped to /kaggle/working)."""
    for c in nb["cells"]:
        c["source"] = [
            ln.replace("Path('../data')", f"Path('{DATA_DIR}')").replace("Path('./data')", f"Path('{DATA_DIR}')")
            for ln in c["source"]
            # the repo-root/notebook alternative comment is meaningless on Kaggle
            if not ln.lstrip().startswith("# DATA_DIR = ")
        ]

    fetch = (
        "from pathlib import Path\n"
        "\n"
        f"DATA_DIR = Path('{DATA_DIR}')\n"
        "\n"
        "# Fetch the ~174 MB sample once: rddac download --small\n"
        "if not (DATA_DIR / 'metadata.json').exists():\n"
        "    !rddac download --small -y --quiet --out {DATA_DIR}"
    )

    setup = [
        _md(
            "## Kaggle setup\n\nInstall `rddac` and fetch the ~174 MB sample — both quiet. "
            "Auto-generated from the repo notebook — "
            "https://github.com/BaumSebastian/RDDAC ."
        ),
        _code(f"!pip install -q {pip_target}"),
        _code(fetch),
    ]
    nb["cells"] = setup + nb["cells"] + [_md(_CTA)]
    return nb


def kernel_metadata(slug: str, title: str, code_file: str) -> dict:
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": True,  # needed for pip install + rddac download
        "dataset_sources": [DATASET],  # links the kernel to the dataset's Code tab
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)  # drop stale slugs from earlier builds
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, title, pip_target in NOTEBOOKS:
        slug = _slug(title)
        nb = adapt(json.loads((NB_DIR / fname).read_text()), pip_target)
        dest = OUT / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / fname).write_text(json.dumps(nb, indent=1))
        (dest / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(slug, title, fname), indent=2))
        print(f"built {slug}  ->  {dest}")


if __name__ == "__main__":
    main()
