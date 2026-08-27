"""mkdocs-macros entry point.

This module is documentation-only. Structural constants (experiment counts,
category grid) live at the top. File names, sizes, and descriptions are read
from the committed Croissant manifest (`docs/metadata.json`, regenerated from
DaRUS by `.helper/v1/build_croissant.py`), so nothing file-related is
hand-maintained here.
"""

from __future__ import annotations

import json
from pathlib import Path

import rddac as _rddac
from rddac import croissant as _croissant
from rddac.spec import RDDAC_SPEC

METADATA_FILE = RDDAC_SPEC.metadata_file
SMALL_TEST_FILES = list(RDDAC_SPEC.small_test_files)

# ---------------------------------------------------------------------------
# Structural constants — facts of the experiment design, not of the files.
# ---------------------------------------------------------------------------
EXPERIMENT_COUNT = 9_000
EXPERIMENTS_PER_GEOMETRY = 4_500
CATEGORY_COUNT = 18
MISSING_POINTCLOUD = 10  # experiments with has_pointcloud=False
MISSING_OIL = 123  # experiments with has_oil=False

# The matching FEM simulations live in the DDACS dataset (rddac.zip there), so
# their size is not in RDDAC's manifest. The one display value maintained by
# hand; check after a DDACS release.
SIMULATION_DOWNLOAD_SIZE = "~9 GB"

# ---------------------------------------------------------------------------
# File names / sizes / descriptions from the committed Croissant manifest.
# ---------------------------------------------------------------------------
_ZIP_FILES = ("concave.zip", "convex.zip", "sample.zip")


def _file_objects() -> list[dict]:
    """FileObject entries (name, bytes, description) from docs/metadata.json."""
    manifest = json.loads(_DOCS_METADATA.read_text())
    out = []
    for d in manifest.get("distribution", []):
        if d.get("@type") != "cr:FileObject":
            continue
        size = d.get("contentSize", "")  # "12345 B"
        out.append(
            {
                "name": d["name"],
                "bytes": int(size.split()[0]) if size.endswith(" B") else None,
                "description": d.get("description", ""),
            }
        )
    return out


def _fmt_size(n: int) -> str:
    """'~87 GB' style, decimal units."""
    for unit, factor in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= factor:
            value = n / factor
            return f"~{value:.1f} {unit}" if value < 10 else f"~{value:.0f} {unit}"
    return f"{n} B"


def _size_of(*names: str) -> int:
    by_name = {f["name"]: f["bytes"] for f in _file_objects() if f["bytes"]}
    return sum(by_name.get(n, 0) for n in names)


# ---------------------------------------------------------------------------
# Croissant manifest helpers (read at build time, not coupled to the values
# above — they describe the schema, not file sizes / counts).
#
# Preference order:
#   1. docs/metadata.json   — bundled with the docs build; what RTD uses.
#                             Version-locked to the git tag; refreshed by
#                             .helper/v1/build_croissant.py on each release.
#   2. data/upload/metadata.json — local developer copy.
#   3. DaRUS URL            — last-resort network fetch.
# ---------------------------------------------------------------------------
_DOCS_METADATA = Path(__file__).resolve().parent / METADATA_FILE
_LOCAL_METADATA = Path(__file__).resolve().parent.parent / "data" / METADATA_FILE


def _dataset():
    for candidate in (_DOCS_METADATA, _LOCAL_METADATA):
        if candidate.is_file():
            return _croissant.load(source=candidate)
    return _croissant.load(source=None)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{line}\n{sep}\n{body}"


def define_env(env):
    """Plug-in entry point — mkdocs-macros calls this at build time."""

    # ----- simple value substitutions -----
    @env.macro
    def experiment_count() -> str:
        return f"{EXPERIMENT_COUNT:,}"

    @env.macro
    def experiments_per_geometry() -> str:
        return f"{EXPERIMENTS_PER_GEOMETRY:,}"

    @env.macro
    def category_count() -> str:
        return str(CATEGORY_COUNT)

    @env.macro
    def missing_pointcloud() -> str:
        return str(MISSING_POINTCLOUD)

    @env.macro
    def missing_oil() -> str:
        return str(MISSING_OIL)

    @env.macro
    def total_size() -> str:
        return _fmt_size(_size_of(*_ZIP_FILES))

    @env.macro
    def per_experiment_size() -> str:
        return _fmt_size(_size_of(*_ZIP_FILES) / EXPERIMENT_COUNT)

    @env.macro
    def small_download_size() -> str:
        # What `rddac download --small` fetches (the manifest itself adds ~20 KB).
        return _fmt_size(_size_of("sample.zip", "process_parameters.csv"))

    @env.macro
    def simulation_download_size() -> str:
        return SIMULATION_DOWNLOAD_SIZE

    @env.macro
    def darus_files_table() -> str:
        rows = [
            [f"`{f['name']}`", _fmt_size(f["bytes"]) if f["bytes"] else "", f["description"]] for f in _file_objects()
        ]
        # Files DaRUS hosts alongside the data, not part of the manifest's
        # distribution (a manifest cannot describe itself).
        rows.append(
            [
                "`metadata.json`",
                "",
                "The Croissant 1.1 manifest — the machine readable schema of this table and everything in the HDF5 files.",
            ]
        )
        rows.append(["`rddac_documentation.pdf`", "", "Standalone dataset documentation."])
        return _md_table(["File", "Size", "Contents"], rows)

    # ----- small auto-built table summarising the counts -----
    @env.macro
    def experiment_stats() -> str:
        return _md_table(
            ["", "Count"],
            [
                ["Experiments (total)", f"{EXPERIMENT_COUNT:,}"],
                ["per geometry (concave / convex)", f"{EXPERIMENTS_PER_GEOMETRY:,} each"],
                ["categories (geometry × force × oil)", str(CATEGORY_COUNT)],
                ["without point cloud", str(MISSING_POINTCLOUD)],
                ["without oil measurement", str(MISSING_OIL)],
            ],
        )

    # ----- file lists sourced from rddac.spec -----
    @env.macro
    def small_test_files() -> str:
        """Space-separated list used by `rddac download --small`."""
        return " ".join(SMALL_TEST_FILES)

    @env.macro
    def metadata_url() -> str:
        """DaRUS download URL for `metadata.json` (resolved via the API)."""
        try:
            return _croissant.metadata_url()
        except Exception:
            return "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5589"

    @env.macro
    def rddac_version() -> str:
        """Live rddac package version (resolves at docs build time)."""
        return _rddac.__version__

    # ----- schema tables sourced from metadata.json -----
    @env.macro
    def field_map_summary() -> str:
        """'7 fields: `force_data`, …' — derived from the manifest's field-map."""
        names = list(_croissant.field_map(_dataset()))
        return f"{len(names)} fields: " + ", ".join(f"`{n}`" for n in names)

    @env.macro
    def use_case_record_sets() -> str:
        """The task-specific RecordSets (everything but the two schema sets)."""
        ids = [rs.id for rs in _dataset().metadata.record_sets if rs.id not in ("process-parameters", "field-map")]
        return ", ".join(f"`{i}`" for i in ids)

    @env.macro
    def process_parameters_table() -> str:
        ds = _dataset()
        descs = _croissant.process_parameters_descriptions(ds)
        rows = [[f"`{name}`", desc] for name, desc in descs.items()]
        return _md_table(["Column", "Description"], rows)

    @env.macro
    def hdf5_field_table(prefix: str = "") -> str:
        ds = _dataset()
        fields = _croissant.field_map(ds)
        rows = []
        for name, f in fields.items():
            desc = (f.description or "").replace("\n", " ")
            if prefix and prefix not in desc:
                continue
            rows.append([f"`{name}`", desc[:120] + ("…" if len(desc) > 120 else "")])
        if not rows:
            return f"_no HDF5 fields matching prefix `{prefix}` found_"
        return _md_table(["Field", "Description"], rows)
