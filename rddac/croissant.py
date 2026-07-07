"""Croissant manifest access for RDDAC — thin wrappers over :mod:`ddacs.croissant`.

The machinery lives in the ``ddacs`` package (shared with the DDACS simulation
dataset); everything here just injects :data:`rddac.spec.RDDAC_SPEC` so the
manifest resolution targets the RDDAC dataset. The manifest conventions
(``field-map`` RecordSet, ``process-parameters`` join on ``index``) are
identical between the datasets, so ``add_view`` & friends pass through as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlcroissant as mlc
from ddacs import croissant as _ddacs_croissant
from ddacs.croissant import (  # noqa: F401  — shared, dataset-agnostic API
    FieldSpec,
    TimestepSpec,
    add_view,
    dataset_name,
    field_map,
    process_parameters_descriptions,
)
from ddacs.croissant import (  # noqa: F401  — internals used by tests/tools
    _build_mapping,
    _load_jsonld_dict,
    _lookup_data_type,
    _normalize_field_spec,
    _record_set,
    _resolve_field_id,
    _slicing_to_jsonpath,
)

from .spec import RDDAC_SPEC

DEFAULT_DATA_DIR = RDDAC_SPEC.default_data_dir


def metadata_url() -> str:
    """Return the DaRUS download URL for the published RDDAC ``metadata.json``.

    Resolved via the DaRUS API (numeric file id — DaRUS has no per-file
    persistent ids) and cached for the process lifetime.
    """
    return _ddacs_croissant.metadata_url(RDDAC_SPEC)


def __getattr__(name: str) -> Any:
    # Lazy module attribute (PEP 562): METADATA_URL mirrors the ddacs API.
    if name == "METADATA_URL":
        return metadata_url()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_source(source: str | Path | None = None, data_dir: str | Path | None = None) -> str:
    """Return the source string (path or URL) that :func:`load` would use.

    Resolution order:
        1. ``source`` if given (local path or HTTP(S) URL).
        2. ``<data_dir>/metadata.json`` if it exists locally.
        3. The DaRUS download URL from :func:`metadata_url`.
    """
    return _ddacs_croissant.resolve_source(source, data_dir, spec=RDDAC_SPEC)


def load(
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
) -> mlc.Dataset:
    """Return an :class:`mlcroissant.Dataset` for the RDDAC manifest.

    Local-first, URL-fallback resolution — see :func:`resolve_source`. When
    ``data_dir`` points at a directory that contains files referenced by the
    manifest (e.g. zips written by ``rddac download``), `mlcroissant` is told
    to use those local copies instead of refetching from DaRUS.

    Pass ``data_dir=None`` to opt out of local-file discovery and force
    `mlcroissant` to download via its own cache.
    """
    return _ddacs_croissant.load(source, data_dir, spec=RDDAC_SPEC)
