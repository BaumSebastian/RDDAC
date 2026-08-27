"""Test fixtures for RDDAC.

Three tiers of data:

* ``synthetic_data_dir`` — hand-built ~few-KB Croissant dataset constructed
  programmatically. Offline, isolated, used for unit tests of the public
  surface (`load`, `add_view`, `open_h5`, `inspect_h5`, `RDDACDataset`,
  `streaming`, `views`).
* ``small_data_dir`` — `rddac download --small` against DaRUS, run once per
  pytest session. Only runs when ``RDDAC_RUN_DOWNLOAD_TESTS`` is set (the
  sample bundle is ~174 MB); picks up ``DARUS_API_TOKEN`` from ``.env`` so
  the draft dataset is reachable. Skipped when the download fails.
* ``real_data_dir`` — the full local dataset (RDDAC_FULL_DATA_DIR, default
  ``/mnt/data/datasets/rddac/upload``)
  (concave.zip, convex.zip, sample.zip, process_parameters.csv,
  metadata.json). Skipped when missing. Tests against it are guarded to a
  handful of experiment ids so the suite stays fast.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Load .env so DARUS_API_TOKEN is picked up automatically.
# ---------------------------------------------------------------------------
_ENV_FILE = _REPO_ROOT / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text().splitlines():
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Tier A — synthetic dataset
# ---------------------------------------------------------------------------

# id -> (geometry, category, blankholder_force, oil_type, split)
_SYNTHETIC_EXPERIMENTS = {
    1: ("concave", 0, 100, "coarse", "train"),
    2: ("concave", 0, 100, "coarse", "val"),
    3: ("convex", 1, 150, "fine", "test"),
}
SYNTHETIC_EXPERIMENT_IDS = sorted(_SYNTHETIC_EXPERIMENTS)

# Tiny stand-ins for the real measurement sizes.
N_FORCE = 10  # real: ~1140 rows
N_SHEET = 6  # real: ~208 rows
N_OIL = 5  # real: ~421 rows
X_SHAPE = 8  # real: 3200
Y_SHAPE = 4  # real: 2000

# All h5 field-map entries the synthetic manifest declares (mirrors the
# published manifest).
_FIELD_MAP_ENTRIES = {
    "force_data": "force/data",
    "sheet_thickness_data": "sheet_thickness/data",
    "oil_thickness_data": "oil_thickness/data",
    "pointcloud_op10_z": "pointcloud/op10/z",
    "pointcloud_op10_luminescence": "pointcloud/op10/luminescence",
    "pointcloud_op20_z": "pointcloud/op20/z",
    "pointcloud_op20_luminescence": "pointcloud/op20/luminescence",
}

_CSV_COLUMNS = [
    ("index", "sc:Integer"),
    ("experiment_id", "sc:Integer"),
    ("category", "sc:Integer"),
    ("geometry", "sc:Text"),
    ("blankholder_force", "sc:Integer"),
    ("mean_punch_temp", "sc:Float"),
    ("oil_type", "sc:Text"),
    ("has_pointcloud", "sc:Boolean"),
    ("has_oil", "sc:Boolean"),
    ("split", "sc:Text"),
]


def _make_synthetic_h5(exp_id: int) -> bytes:
    """Build a tiny in-memory h5 modelling the RDDAC raw file structure."""
    geometry, category, bhf, oil_type, _split = _SYNTHETIC_EXPERIMENTS[exp_id]
    rng = np.random.default_rng(exp_id)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["id"] = exp_id
        f.attrs["experiment_id"] = exp_id + 1
        f.attrs["geometry"] = geometry
        f.attrs["category"] = category
        f.attrs["blankholder_force"] = bhf
        f.attrs["oil_type"] = oil_type
        f.attrs["mean_punch_temp"] = 20.0 + exp_id
        f.attrs["has_oil"] = True
        f.attrs["has_pointcloud"] = True
        f.attrs["n_force_measurements"] = N_FORCE
        f.attrs["n_sheet_measurements"] = N_SHEET

        g = f.create_group("force")
        g.attrs["columns"] = [
            "time",
            "load_cell_1",
            "load_cell_2",
            "load_cell_3",
            "load_cell_4",
            "punch_temp",
            "punch_pos",
            "total_force",
        ]
        g.create_dataset("data", data=rng.random((N_FORCE, 8)).astype(np.float32))

        g = f.create_group("sheet_thickness")
        g.create_dataset("data", data=rng.random((N_SHEET, 2)).astype(np.float32))

        g = f.create_group("oil_thickness")
        g.create_dataset("data", data=rng.random((N_OIL, 2)).astype(np.float32))

        for op in ("op10", "op20"):
            g = f.create_group(f"pointcloud/{op}")
            g.attrs["x_shape"] = X_SHAPE
            g.attrs["y_shape"] = Y_SHAPE
            g.create_dataset("z", data=rng.random(X_SHAPE * Y_SHAPE).astype(np.float32))
            g.create_dataset("luminescence", data=rng.random(X_SHAPE * Y_SHAPE).astype(np.float32))
    return buf.getvalue()


def _make_synthetic_manifest() -> dict:
    """A minimal RDDAC-shaped Croissant 1.1 manifest.

    Mirrors the published layout: a CSV FileObject, geometry zips, an
    ``h5-files`` FileSet, a ``process-parameters`` RecordSet, the
    ``field-map`` RecordSet and a published-style ``force-curve`` view.
    """
    csv_id = "process_parameters_csv"
    zips = {"concave_zip": "concave.zip", "convex_zip": "convex.zip"}
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "containedIn": "cr:containedIn",
            "cr": "http://mlcommons.org/croissant/",
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "fileSet": "cr:fileSet",
            "includes": "cr:includes",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "source": "cr:source",
            "transform": "cr:transform",
            "sc": "https://schema.org/",
        },
        "@type": "sc:Dataset",
        "name": "synthetic-rddac",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "description": "synthetic rddac test dataset",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": csv_id,
                "name": "process_parameters.csv",
                "contentUrl": "process_parameters.csv",
                "encodingFormat": "text/csv",
                "md5": "0" * 32,
            },
        ]
        + [
            {
                "@type": "cr:FileObject",
                "@id": zid,
                "name": zname,
                "contentUrl": zname,
                "encodingFormat": "application/zip",
                "md5": "0" * 32,
            }
            for zid, zname in zips.items()
        ]
        + [
            {
                "@type": "cr:FileSet",
                "@id": "h5-files",
                "name": "h5-files",
                "containedIn": [{"@id": zid} for zid in zips],
                "encodingFormat": "application/x-hdf5",
                "includes": ["*.h5"],
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "process-parameters",
                "name": "process-parameters",
                "key": {"@id": "process-parameters/index"},
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": f"process-parameters/{col}",
                        "name": col,
                        "description": f"synthetic {col} column",
                        "dataType": dtype,
                        "source": {
                            "fileObject": {"@id": csv_id},
                            "extract": {"column": col},
                        },
                    }
                    for col, dtype in _CSV_COLUMNS
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "field-map",
                "name": "field-map",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": f"field-map/{name}",
                        "name": name,
                        "dataType": "sc:Float",
                        "source": {
                            "fileSet": {"@id": "h5-files"},
                            "extract": {"fileProperty": "content"},
                            "transform": [{"regex": h5_path}],
                        },
                    }
                    for name, h5_path in _FIELD_MAP_ENTRIES.items()
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "force-curve",
                "name": "force-curve",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "force-curve/force_data",
                        "name": "force_data",
                        "dataType": "sc:Float",
                        "source": {"field": {"@id": "field-map/force_data"}},
                    }
                ],
            },
        ],
    }


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """Self-contained RDDAC-shaped dataset: CSV + geometry zips + manifest.

    Zip members are zero-padded 4 digits (``RDDAC_SPEC.id_format``): experiment
    1 -> ``0001.h5``. Concave experiments (1, 2) live in ``concave.zip``,
    convex (3) in ``convex.zip`` — mirroring the published grouping.
    """
    out = tmp_path_factory.mktemp("rddac_synth")

    # process_parameters.csv — column order matches the published CSV.
    header = ",".join(col for col, _ in _CSV_COLUMNS)
    rows = [header]
    for exp_id, (geometry, category, bhf, oil_type, split) in _SYNTHETIC_EXPERIMENTS.items():
        rows.append(
            f"{exp_id},{exp_id + 1},{category},{geometry},{bhf}," f"{20.0 + exp_id},{oil_type},True,True,{split}"
        )
    (out / "process_parameters.csv").write_text("\n".join(rows) + "\n")

    # Geometry zips with zero-padded members.
    by_zip: dict[str, list[int]] = {"concave.zip": [], "convex.zip": []}
    for exp_id, (geometry, *_rest) in _SYNTHETIC_EXPERIMENTS.items():
        by_zip[f"{geometry}.zip"].append(exp_id)
    for zip_name, ids in by_zip.items():
        with zipfile.ZipFile(out / zip_name, "w") as zf:
            for exp_id in ids:
                zf.writestr(f"{exp_id:04d}.h5", _make_synthetic_h5(exp_id))

    # metadata.json — Croissant manifest
    (out / "metadata.json").write_text(json.dumps(_make_synthetic_manifest()))

    return out


# ---------------------------------------------------------------------------
# Tier B — `rddac download --small`
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_data_dir(tmp_path_factory) -> Path:
    """The small bundle in the user layout — cached, download-as-update.

    Cache location: ``RDDAC_SMALL_DATA_DIR`` (default
    ``/mnt/data/datasets/rddac_small``). When the bundle is already there the
    fixture reuses it without any network access, so this tier runs on every
    local test run. Only when the cache is missing does it download (~174 MB,
    token from ``.env`` for the draft) — and only if
    ``RDDAC_RUN_DOWNLOAD_TESTS`` is set; otherwise it skips.
    """
    cache = Path(os.environ.get("RDDAC_SMALL_DATA_DIR", "/mnt/data/datasets/rddac_small"))
    required = ["metadata.json", "process_parameters.csv", "h5/sample.zip"]
    if all((cache / name).is_file() for name in required):
        return cache

    if not os.environ.get("RDDAC_RUN_DOWNLOAD_TESTS"):
        pytest.skip(f"small bundle not cached at {cache}; set RDDAC_RUN_DOWNLOAD_TESTS=1 " "to download it (~174 MB)")

    out = cache if os.access(cache.parent, os.W_OK) else tmp_path_factory.mktemp("rddac_small")
    cmd = [sys.executable, "-m", "rddac.cli"]
    token = os.environ.get("DARUS_API_TOKEN")
    if token:
        cmd += ["--token", token]
    cmd += ["download", "--small", "-y", "--out", str(out)]
    if token:
        # The dataset is still a draft; the published version doesn't exist yet.
        cmd.append(":draft")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT))
    except subprocess.TimeoutExpired:
        pytest.skip("rddac download --small timed out")
    if res.returncode != 0:
        pytest.skip(f"rddac download --small failed: {res.stderr[-200:]}")
    return out


# ---------------------------------------------------------------------------
# Tier C — full local dataset (/mnt/data/datasets/rddac/upload)
# ---------------------------------------------------------------------------

# Experiment ids the real-data tests are allowed to touch (one per zip:
# 0 and 4500 live in sample.zip, 42 in concave.zip). Keep this list tiny —
# the full release has 9000 experiments and iterating them would take hours.
REAL_EXPERIMENT_IDS = [0, 42, 4500]


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    """The full local dataset (default /mnt/data/datasets/rddac/upload; override via RDDAC_FULL_DATA_DIR)."""
    real = Path(os.environ.get("RDDAC_FULL_DATA_DIR", "/mnt/data/datasets/rddac/upload"))
    required = ["metadata.json", "process_parameters.csv", "sample.zip"]
    missing = [name for name in required if not (real / name).is_file()]
    if missing:
        pytest.skip(f"full local dataset not available: {real} is missing {missing}")
    return real
