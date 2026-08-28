"""Model cache for the RF fin classifier.

Trained models are derived artifacts and are NOT shipped: they are retrained
deterministically from the bundled labels on first use and cached under
``<out_dir>/models/pointcloud_fin_rf/`` (the processed output directory;
the raw data directory is never written to). A cached bundle is only reused when
its fingerprint matches: schema version, scikit-learn major.minor, the sha256
of the bundled labels, the RF hyperparameters, and the sim-distance method.
The decision threshold is applied at predict time and is deliberately NOT
part of the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .. import defaults as d

#: Bump when the feature schema or bundle layout changes (forces retrain).
SCHEMA_VERSION = 2  # 2: cup-anchored two-pass alignment (sim-distance features changed)
SIMDIST_METHOD = "kd"

CACHE_SUBDIR = Path("models") / "pointcloud_fin_rf"


def labels_sha256() -> str:
    """Fingerprint of the bundled labels (names + bytes, sorted)."""
    digest = hashlib.sha256()
    root = resources.files("rddac._preprocess") / "labels"
    entries = sorted((entry.name, entry) for entry in root.iterdir() if entry.name.endswith(".npz"))
    for name, entry in entries:
        digest.update(name.encode())
        digest.update(entry.read_bytes())
    return digest.hexdigest()


def _sklearn_tag() -> str:
    return ".".join(sklearn.__version__.split(".")[:2])


def cache_dir(cache_root: str | Path) -> Path:
    """The model cache directory under ``cache_root`` (the processed output directory)."""
    return Path(cache_root) / CACHE_SUBDIR


def fingerprint(rf_n_estimators: int = d.PC_RF_N_ESTIMATORS, rf_max_depth: int = d.PC_RF_MAX_DEPTH) -> dict:
    """The current cache fingerprint (everything that invalidates a bundle)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sklearn": _sklearn_tag(),
        "labels_sha256": labels_sha256(),
        "simdist_method": SIMDIST_METHOD,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_depth": rf_max_depth,
    }


def load_bundles(cache_root: str | Path, expected_fingerprint: dict) -> dict[str, dict] | None:
    """Load all cached group bundles, or None when the cache is missing/stale.

    Args:
        data_dir: The RDDAC data directory.
        expected_fingerprint: From :func:`fingerprint`.

    Returns:
        Mapping group -> bundle (``model``, ``feature_names``, ``prior_reg``,
        ``expected_reg``, ``ref_ds``) — or None if a retrain is needed.
    """
    root = cache_dir(cache_root)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    if {k: meta.get(k) for k in expected_fingerprint} != expected_fingerprint:
        return None
    bundles: dict[str, dict] = {}
    for group in meta.get("groups", []):
        path = root / f"{group}.joblib"
        if not path.is_file():
            return None
        bundles[group] = joblib.load(path)
    return bundles or None


def save_bundles(cache_root: str | Path, bundles: dict[str, dict], meta_extra: dict, fp: dict) -> None:
    """Persist trained bundles plus the fingerprint metadata."""
    root = cache_dir(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    for group, bundle in bundles.items():
        joblib.dump(bundle, root / f"{group}.joblib")
    meta = {**fp, "groups": sorted(bundles), **meta_extra}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))


def predict_outliers(bundle: dict, x: np.ndarray, threshold: float = d.PC_RF_THRESHOLD) -> np.ndarray:
    """Outlier mask from a bundle's model at the given probability threshold.

    Prediction runs single-threaded: the runner parallelises over experiments
    (one process per worker), and nested joblib parallelism inside a worker
    would only fall back to one thread with a warning.
    """
    model = bundle["model"]
    if getattr(model, "n_jobs", 1) != 1:
        model.set_params(n_jobs=1)
    proba = model.predict_proba(x)[:, 1]
    return proba >= threshold
