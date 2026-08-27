"""Croissant manifest generation for the PROCESSED layout.

The published ``metadata.json`` describes the immutable raw dataset and never
changes. This module generates a small companion manifest inside the output
directory, describing what is actually on disk after a ``rddac preprocess``
run — same dataset names as raw, new shapes — so the processed files are
consumable through the same Croissant-driven machinery::

    ds = rddac.load(source="<out>/metadata.json", data_dir="<out>")

The manifest is regenerated at the end of every run from the files present
(a partial run yields a manifest of exactly the groups that exist).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py

import rddac

from .spec_fields import CONTEXT  # noqa: F401  (re-exported for tests)

RAW_DATASET_DOI = "https://doi.org/10.18419/DARUS-5589"


def _fields_from_file(path: Path) -> list[dict]:
    """One field-map entry per dataset present in a processed file."""
    fields: list[dict] = []

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        shape = tuple(int(s) if i != 0 or name.split("/")[-1] != "z" else -1 for i, s in enumerate(obj.shape))
        if name.startswith("pointcloud/") and name.endswith("/z"):
            shape = (-1, 3)  # per-experiment point count varies
        field_id = name.replace("/", "_")
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"field-map/{field_id}",
                "name": field_id,
                "description": f"Processed dataset. HDF5 path: {name}. Shape (per file): {shape}.".replace("-1", "N"),
                "dataType": "sc:Float",
                "source": {
                    "fileSet": {"@id": "h5-files"},
                    "extract": {"fileProperty": "content"},
                    "transform": [{"regex": name}],
                },
            }
        )

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return fields


def write_manifest(out_dir: str | Path) -> Path | None:
    """Write ``<out_dir>/metadata.json`` describing the processed files on disk.

    Returns:
        The manifest path, or None when the directory holds no processed files.
    """
    out = Path(out_dir)
    files = sorted(out.glob("*.h5"))
    if not files:
        return None
    manifest = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "RDDAC-processed",
        "description": (
            "ML-ready processed layer of the RDDAC dataset, generated locally by `rddac preprocess`. "
            "Raw files are unmodified; this manifest describes the derived files in this directory."
        ),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "isBasedOn": RAW_DATASET_DOI,
        "version": rddac.__version__,
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "processed_dir",
                "name": "processed_dir",
                "description": "The directory this manifest sits in.",
                "contentUrl": ".",
                "encodingFormat": "inode/directory",
            },
            {
                "@type": "cr:FileSet",
                "@id": "h5-files",
                "name": "h5-files",
                "description": f"Per-experiment processed HDF5 files ({len(files)} present).",
                "containedIn": [{"@id": "processed_dir"}],
                "encodingFormat": "application/x-hdf5",
                "includes": ["*.h5"],
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "field-map",
                "name": "field-map",
                "description": (
                    "Every processed HDF5 dataset declared once, with its on-disk path and per-file shape "
                    "(from the first file present)."
                ),
                "field": _fields_from_file(files[0]),
            }
        ],
    }
    path = out / "metadata.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
