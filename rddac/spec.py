"""RDDAC dataset specification — single source of truth for the dataset identity.

The identity lives in :data:`RDDAC_SPEC` (a :class:`ddacs.spec.DatasetSpec`);
the machinery in the ``ddacs`` package consumes it via the ``spec=`` keyword —
see the thin wrappers in :mod:`rddac.croissant`, :mod:`rddac.h5_tools`,
:mod:`rddac.streaming` and :mod:`rddac.pytorch`. Consuming modules derive any
module-level constants they need directly from the spec.
"""

from ddacs.spec import DatasetSpec

__all__ = ["DatasetSpec", "RDDAC_SPEC"]

# ── RDDAC dataset identity ────────────────────────────────────────────────────
RDDAC_SPEC = DatasetSpec(
    name="RDDAC",
    prog="rddac",
    dataset_doi="doi:10.18419/DARUS-5589",
    default_version="1.0",
    # Experiment ids are zero-padded in the HDF5 member names: 42 -> "0042.h5".
    id_format="{:04d}",
    small_test_files=(
        "process_parameters.csv",
        "metadata.json",
        "sample.zip",
    ),
)

#: Brand colour of the documentation (docs/stylesheets/extra.css); used for CLI panels.
BRAND_COLOR = "#7F00FF"

# ── Hyperlinks (kept here so a URL change touches one file) ───────────────────
DATASET_URL = f"{RDDAC_SPEC.darus_base_url}/dataset.xhtml?persistentId={RDDAC_SPEC.dataset_doi}"
DOI_URL = f"https://doi.org/{RDDAC_SPEC.dataset_doi.replace('doi:', '')}"
GITHUB_URL = "https://github.com/BaumSebastian/RDDAC"
DOCS_URL = "https://rddac.readthedocs.io"

# ── DDACS simulation reference data (RDDAC-specific, not spec material) ───────
# RDDAC is the experimental counterpart to DDACS; `rddac download` fetches the
# matching FEM simulations alongside the measurements (skip with --no-sim) by
# delegating to the installed `ddacs` CLI. They are the RDDAC sub-study subset
# published in the DDACS dataset as a single zip.
DDACS_DATASET_DOI = "doi:10.18419/DARUS-4801"
DDACS_DATASET_URL = f"{RDDAC_SPEC.darus_base_url}/dataset.xhtml?persistentId={DDACS_DATASET_DOI}"
DDACS_SIM_FILE = "rddac.zip"
SIM_SUBDIR = "simulation"  # local subdirectory the simulations download into
