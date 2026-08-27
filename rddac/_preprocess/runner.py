"""Orchestration for ``rddac preprocess`` — experiment loop, file IO, progress.

The CLI resolves arguments and calls :func:`run`; the science lives in the
modality modules. Simple modalities expose a pure ``process`` function; the
pointcloud stage owns its whole file section via the ``PROCESSES_FILE``
contract (see :mod:`.pointcloud.stage`). One raw file is opened per
experiment, all selected modalities run on it, and the results land in a NEW
``<out>/<id>.h5`` (append mode: re-runs update the selected groups only).
Raw files are opened strictly read-only and never modified.
"""

from __future__ import annotations

import importlib
import os
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

from . import force
from .h5_access import available_ids, open_raw

#: Marker file dropped into the output directory; identifies it as processed
#: output so re-runs pass the raw-directory guard in the CLI.
PROCESSED_MARKER = ".rddac-processed"

#: Modality name → submodule (imported lazily: the pointcloud stage needs the
#: optional ``rddac[preprocessing]`` dependencies, the others do not).
MODALITIES = {"force": "force", "sheet": "sheet", "oil": "oil", "pointcloud": "pointcloud.stage"}

#: Modality name → h5 group it reads and writes.
GROUPS = {"force": "force", "sheet": "sheet_thickness", "oil": "oil_thickness", "pointcloud": "pointcloud"}

#: Human-readable meaning of the per-experiment status codes in the summary.
_STATUS_TEXT = {
    "processed": "processed",
    "exists": "already in the output file, skipped (use --overwrite to recompute)",
    "no_group": "skipped, the raw file has no such measurement",
    "no_file": "skipped, no raw file for this experiment id",
    "error": "failed (see messages above)",
}
_STATUS_ORDER = {"processed": 0, "exists": 1, "no_group": 2, "no_file": 3, "error": 4}


def _status_order(item: tuple[str, int]) -> int:
    return _STATUS_ORDER.get(item[0], 9)


#: Root attrs that were dropped from the processed metadata (obsolete counts).
_OBSOLETE_ROOT_ATTRS = {
    "n_force_measurements",
    "n_sheet_measurements",
    "n_oil_measurements",
    "n_pointcloud_measurements",
}


def modality_module(name: str):
    """Import a modality module, translating missing optional deps into advice."""
    try:
        return importlib.import_module(f"rddac._preprocess.{MODALITIES[name]}")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            f"{name}: missing optional dependencies ({exc.name}) — install them with "
            "pip install 'rddac[preprocessing]'"
        ) from exc


def parse_ids(spec: str) -> set[int]:
    """Parse an id selection like ``'0-999'``, ``'42,1035'`` or ``'0-10,42'``."""
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif token:
            out.add(int(token))
    return out


def _experiment_ids(data_dir: str, ids: str | None, split: str | None) -> list[int]:
    """Experiment ids to process, from ``process_parameters.csv`` or loose files."""
    csv = os.path.join(data_dir, "process_parameters.csv")
    if os.path.isfile(csv):
        table = pd.read_csv(csv)
        if split:
            table = table[table["split"] == split]
        selected = [int(i) for i in table["index"]]
    else:
        if split:
            raise FileNotFoundError(f"--split needs {csv} (it defines the split assignment)")
        selected = sorted(int(p.stem) for p in Path(data_dir).glob("*.h5") if p.stem.isdigit())
    if ids:
        allow = parse_ids(ids)
        selected = [i for i in selected if i in allow]
    local = available_ids(data_dir)
    if local is not None:
        selected = [i for i in selected if i in local]
    return selected


def _column_names(attrs, default: tuple[str, ...]) -> list[str]:
    """Column names from a group's ``columns`` attr (bytes/str/array tolerated)."""
    columns = attrs.get("columns")
    if columns is None:
        return list(default)
    if isinstance(columns, bytes):
        columns = columns.decode()
    if isinstance(columns, str):
        return [c.strip() for c in columns.split(",")]
    return [c.decode() if isinstance(c, bytes) else str(c) for c in np.asarray(columns).tolist()]


def _process_one(job: tuple) -> dict:
    """Worker: process all selected modalities of one experiment."""
    exp_id, data_dir, out_dir, names, overwrite, cfg = job
    status: dict[str, str] = {}
    try:
        raw = open_raw(exp_id, data_dir)
    except FileNotFoundError:
        return {"id": exp_id, "status": {name: "no_file" for name in names}}
    try:
        with h5py.File(os.path.join(out_dir, f"{exp_id:04d}.h5"), "a") as out:
            for key, value in raw.attrs.items():
                if key not in _OBSOLETE_ROOT_ATTRS:
                    out.attrs[key] = value
            for name in names:
                try:
                    status[name] = _process_group(name, raw, out, overwrite, cfg.get(name, {}), data_dir)
                except Exception as exc:  # keep the other modalities going
                    status[name] = f"error: {exc}"
    finally:
        raw.close()
    return {"id": exp_id, "status": status}


def _process_group(name: str, raw: h5py.File, out: h5py.File, overwrite: bool, params: dict, data_dir: str) -> str:
    module = modality_module(name)
    if getattr(module, "PROCESSES_FILE", False):
        return module.process_experiment(raw, out, data_dir=data_dir, overwrite=overwrite, **params)

    group = GROUPS[name]
    if group not in raw:
        return "no_group"
    if group in out and not overwrite:
        return "exists"
    data = raw[f"{group}/data"][:]
    if name == "force":
        columns = _column_names(raw[group].attrs, force.COLUMNS)
        processed, attrs = module.process(data, columns, **params)
        units = force.units_for(columns)
    else:
        processed, attrs = module.process(data, **params)
        columns = list(module.COLUMNS)
        units = list(module.UNITS)
    if group in out:
        del out[group]
    g = out.create_group(group)
    g.create_dataset("data", data=processed, compression="gzip", compression_opts=4)
    g.attrs["columns"] = columns
    g.attrs["units"] = units
    g.attrs["n_measurements"] = len(processed)
    for key, value in attrs.items():
        g.attrs[key] = value
    return "processed"


def run(
    names: list[str],
    data_dir: str,
    out_dir: str,
    ids: str | None = None,
    split: str | None = None,
    workers: int = 1,
    overwrite: bool = False,
    quiet: bool = False,
    console=None,
    config: dict | None = None,
    skip_unavailable: bool = False,
    rebuild_models: bool = False,
) -> dict:
    """Process the selected modalities for all selected experiments.

    Args:
        names: Modalities to run, in order.
        data_dir: Directory holding the raw dataset (zips or loose ``.h5``).
        out_dir: Output directory for processed files (created if missing).
        ids: Optional id selection, e.g. ``'0-999'`` or ``'42,1035'``.
        split: Optional predefined split name to restrict to.
        workers: Number of parallel worker processes.
        overwrite: Recompute modality groups that already exist.
        quiet: Suppress progress bar and summary output.
        console: rich console to print to; a new one is created if None.
        config: Modality name → parameter overrides (see :mod:`.config`);
            missing entries use the defaults.
        skip_unavailable: Skip (instead of fail on) modalities whose optional
            dependencies are missing — used when the user did not name
            modalities explicitly.
        rebuild_models: Force retraining of cached models in preflight.

    Returns:
        Stats dict ``{modality: {status: count}}`` plus ``"elapsed_s"``.
    """
    if console is None:
        console = Console(quiet=quiet)
    config = config or {}

    selected: list[str] = []
    for name in names:
        try:
            module = modality_module(name)
        except RuntimeError as exc:
            if skip_unavailable:
                console.print(f"[yellow]{exc} — skipped[/yellow]")
                continue
            raise
        selected.append(name)
    names = selected
    if not names:
        return {"elapsed_s": 0.0}

    # One-time per-run hooks (main process, before forking workers). A stage
    # whose prerequisites are missing (e.g. pointcloud without simulations)
    # is skipped with a notice unless the user named it explicitly.
    ready: list[str] = []
    for name in names:
        module = modality_module(name)
        if hasattr(module, "preflight"):
            try:
                module.preflight(
                    data_dir, config.get(name, {}), console=console, rebuild_models=rebuild_models, out_dir=out_dir
                )
            except FileNotFoundError as exc:
                if not skip_unavailable:
                    raise
                console.print(f"[yellow]{name}: {exc} — skipped[/yellow]")
                continue
        ready.append(name)
    names = ready
    if not names:
        return {"elapsed_s": 0.0}

    experiment_ids = _experiment_ids(data_dir, ids, split)
    os.makedirs(out_dir, exist_ok=True)
    Path(out_dir, PROCESSED_MARKER).touch()

    jobs = [(i, data_dir, out_dir, names, overwrite, config) for i in experiment_ids]
    t0 = time.time()
    results = []

    def _iter_results():
        if workers > 1:
            with Pool(workers) as pool:
                yield from pool.imap_unordered(_process_one, jobs, chunksize=8)
        else:
            yield from map(_process_one, jobs)

    if quiet:
        results = list(_iter_results())
    else:
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"preprocess {'+'.join(names)}", total=len(jobs))
            for result in _iter_results():
                results.append(result)
                progress.advance(task)

    stats: dict = {name: {} for name in names}
    for result in results:
        for name, state in result["status"].items():
            state = state.split(":")[0]  # collapse per-file error messages
            stats[name][state] = stats[name].get(state, 0) + 1
    stats["elapsed_s"] = round(time.time() - t0, 1)

    for name, counts in stats.items():
        if name == "elapsed_s":
            continue
        parts = [
            f"{count} {_STATUS_TEXT.get(state, state)}" for state, count in sorted(counts.items(), key=_status_order)
        ]
        console.print(f"[bold]{name}[/bold]: " + ", ".join(parts))
    console.print(f"[dim]finished in {stats['elapsed_s']} s[/dim]")
    return stats
