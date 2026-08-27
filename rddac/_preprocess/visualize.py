"""Figures of the processing steps per modality (docs and notebook).

Each ``plot_*_processing`` function takes RAW arrays (as stored in the raw
h5 files) and derives the processed view by calling the modality's own
processing functions, so a figure is by construction what ``rddac
preprocess`` does — there is no second implementation of the pipeline. The
functions return a matplotlib ``Figure`` for inline use in notebooks; the
module's ``__main__`` writes the images used in the documentation::

    python -m rddac._preprocess.visualize oil          --data-dir data --id 42     --out docs/images/preprocessing
    python -m rddac._preprocess.visualize force        --data-dir data --ids 0-199 --out docs/images/preprocessing
    python -m rddac._preprocess.visualize sheet        --data-dir data --ids 0-499 --out docs/images/preprocessing
    python -m rddac._preprocess.visualize luminescence --data-dir data --id 42 --op op10 --out docs/images/preprocessing
    python -m rddac._preprocess.visualize pointcloud   --data-dir data --id 42 --op op10 --out docs/images/preprocessing

The ``pointcloud`` figure compares the raw scan with the output of ``rddac
preprocess pointcloud`` and therefore needs the processed file (default
``<data-dir>/processed/<id>.h5``); the other figures need raw data only.
``--config TOML`` applies the same parameter overrides as ``rddac preprocess``.
This is a maintainer tool, not part of the ``rddac`` CLI.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import force, oil, sheet
from .defaults import FORCE_COLUMNS

#: Colours shared by all figures: raw / processed / cut lines and removed regions.
RAW_COLOR = "C0"
PROCESSED_COLOR = "C2"
CUT_COLOR = "C3"
REMOVED_COLOR = "gray"

#: Default figure size (inches) for the two- and three-panel layouts.
FIGSIZE = (9.0, 3.0)


def _overlay_alpha(n: int) -> float:
    """Line alpha for ``n`` overlaid traces — opaque for few, faint for many."""
    return float(np.clip(3.0 / max(n, 1), 0.08, 0.9))


# ── oil ───────────────────────────────────────────────────────────────────────


def plot_oil_processing(data: np.ndarray, *, figsize: tuple[float, float] = FIGSIZE, **params) -> Figure:
    """Three panels for ONE experiment's raw ``oil_thickness/data``.

    (a) raw readings with the truncation line, (b) truncated readings with the
    Hampel-flagged outliers, (c) the processed fixed-grid profile.

    Args:
        data: ``(n, 2)`` raw array of ``[sensor_position, oil_value]``.
        figsize: Figure size in inches.
        **params: Keyword overrides for :func:`rddac._preprocess.oil.process`.
    """
    processed, attrs = oil.process(data, **params)
    max_pos = attrs["max_sensor_position"]

    df = pd.DataFrame(np.asarray(data, dtype=float), columns=list(oil.COLUMNS)).dropna(subset=["sensor_position"])
    df["pos_int"] = df["sensor_position"].round(0).astype(int)
    # Same pre-filter path as oil.process: truncate, drop dropouts, sort.
    kept = df[df["pos_int"] < max_pos].dropna(subset=["oil_value"]).sort_values("pos_int")
    outlier = oil.hampel_filter(
        kept["pos_int"].to_numpy(),
        kept["oil_value"].to_numpy(),
        window_size=attrs["hampel_window"],
        k=attrs["hampel_k"],
        quantization=attrs["value_quantization"],
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    ax = axes[0]
    ax.scatter(df["pos_int"], df["oil_value"], c=RAW_COLOR, s=10, alpha=0.5)
    ax.axvline(max_pos, color=CUT_COLOR, linestyle="--", linewidth=1.5)
    ax.set_ylabel("Oil in g/m²")
    ax.set_title("(a) Raw")

    ax = axes[1]
    ax.scatter(kept["pos_int"][~outlier], kept["oil_value"][~outlier], c=RAW_COLOR, s=10, alpha=0.5)
    ax.scatter(kept["pos_int"][outlier], kept["oil_value"][outlier], c=CUT_COLOR, s=30, marker="x", linewidths=1.5)
    ax.set_title(f"(b) Hampel filter ({int(outlier.sum())} outliers)")

    ax = axes[2]
    ax.plot(processed[:, 0], processed[:, 1], color=PROCESSED_COLOR, linewidth=1.0)
    ax.scatter(processed[:, 0], processed[:, 1], c=PROCESSED_COLOR, s=8, alpha=0.7)
    ax.set_title("(c) Processed")

    for ax in axes:
        ax.set_xlabel("Position in mm")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── force ─────────────────────────────────────────────────────────────────────


def plot_force_processing(
    tables: Sequence[np.ndarray],
    columns: Sequence[str] = FORCE_COLUMNS,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    **params,
) -> Figure:
    """Two panels with several experiments' ``force/data`` overlaid.

    (a) raw total force over time with the forming window, (b) the processed
    curves: windowed, time re-zeroed, rest offset removed.

    Args:
        tables: Raw ``(n, len(columns))`` arrays, one per experiment.
        columns: Column layout of the tables (the h5 ``columns`` attr).
        figsize: Figure size in inches.
        **params: Keyword overrides for :func:`rddac._preprocess.force.process`.
    """
    columns = list(columns)
    i_t, i_f = columns.index("time"), columns.index("total_force")
    alpha = _overlay_alpha(len(tables))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    t_start = t_end = None
    t_max = 0.0
    for table in tables:
        table = np.asarray(table, dtype=float)
        processed, attrs = force.process(table, columns, **params)
        t_start, t_end = attrs["time_window_start"], attrs["time_window_end"]
        t_max = max(t_max, float(np.nanmax(table[:, i_t])))
        axes[0].plot(table[:, i_t], table[:, i_f], color=RAW_COLOR, alpha=alpha, linewidth=1.0)
        axes[1].plot(processed[:, i_t], processed[:, i_f], color=PROCESSED_COLOR, alpha=alpha, linewidth=1.0)

    ax = axes[0]
    if t_start is not None:
        ax.axvline(t_start, color=CUT_COLOR, linestyle="--", linewidth=1.5)
        ax.axvline(t_end, color=CUT_COLOR, linestyle="--", linewidth=1.5)
        ax.axvspan(0, t_start, alpha=0.15, color=REMOVED_COLOR)
        ax.axvspan(t_end, t_max, alpha=0.15, color=REMOVED_COLOR)
        axes[1].set_xlim(0, t_end - t_start)
    ax.set_xlim(0, t_max)
    ax.set_ylabel("Force in kN")
    ax.set_title("(a) Raw with forming window")
    axes[1].set_title("(b) Processed")

    for ax in axes:
        ax.set_xlabel("Time in s")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── sheet ─────────────────────────────────────────────────────────────────────


def plot_sheet_processing(tables: Sequence[np.ndarray], *, figsize: tuple[float, float] = FIGSIZE, **params) -> Figure:
    """Two panels with several experiments' ``sheet_thickness/data`` overlaid.

    (a) raw traverses (error codes hidden) with the tail-selection cutoff,
    (b) the processed, position-normalized profiles.

    Args:
        tables: Raw ``(n, 2)`` arrays of ``[sensor_position, sheet_thickness]``.
        figsize: Figure size in inches.
        **params: Keyword overrides for :func:`rddac._preprocess.sheet.process`.
    """
    alpha = _overlay_alpha(len(tables))
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    cutoffs: list[float] = []
    starts: list[float] = []
    for table in tables:
        table = np.asarray(table, dtype=float)
        processed, attrs = sheet.process(table, **params)
        pos, thick = table[:, 0], table[:, 1]
        valid = thick > 0  # negative readings are sensor error codes
        axes[0].plot(pos[valid], thick[valid], color=RAW_COLOR, alpha=alpha, linewidth=1.0)
        axes[1].plot(processed[:, 0], processed[:, 1], color=PROCESSED_COLOR, alpha=alpha, linewidth=1.0)
        if len(pos) > attrs["last_n"]:
            cutoffs.append(float(pos[len(pos) - attrs["last_n"]]))
            starts.append(float(pos[0]))

    ax = axes[0]
    if cutoffs:
        cutoff = float(np.median(cutoffs))
        ax.axvline(cutoff, color=CUT_COLOR, linestyle="--", linewidth=1.5)
        ax.axvspan(min(starts), cutoff, alpha=0.15, color=REMOVED_COLOR)
    ax.set_ylabel("Sheet thickness in µm")
    ax.set_title("(a) Raw with cutoff")
    axes[1].set_title("(b) Processed")

    for ax in axes:
        ax.set_xlabel("Position in mm")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── luminescence (validity mask of the pointcloud stage) ──────────────────────

#: Distinct colours for connected patches (index 0 = background).
_PATCH_COLORS = (
    "#ffffff",
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
)


def plot_luminescence_processing(
    lumi_2d: np.ndarray, *, figsize: tuple[float, float] = (14.0, 3.2), **params
) -> Figure:
    """Four panels for one scan's raw luminescence grid.

    (a) raw grid, (b) connected foreground patches, (c) the validity mask
    after the patch-size filter, (d) the packed uint8 grayscale image.

    Args:
        lumi_2d: ``(H, W)`` raw luminescence grid (reshaped ``luminescence``).
        figsize: Figure size in inches.
        **params: ``[pointcloud]`` parameters; only ``lumi_min_patch_size`` is used.
    """
    from matplotlib.colors import ListedColormap
    from scipy import ndimage

    from .defaults import PC_LUMI_MIN_PATCH_SIZE
    from .pointcloud import geometry

    min_patch = params.get("lumi_min_patch_size", PC_LUMI_MIN_PATCH_SIZE)
    lumi_2d = np.asarray(lumi_2d)
    labeled, n_patches = ndimage.label(lumi_2d > 0)
    valid = geometry.lumi_valid_mask(lumi_2d, min_patch)
    packed = geometry.pack_luminescence(lumi_2d, valid)
    n_kept = len(np.unique(labeled[valid])) if valid.any() else 0

    raw_masked = np.where(lumi_2d > 0, lumi_2d.astype(float), np.nan)
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    im = axes[0].imshow(raw_masked, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].set_title("(a) Raw")

    n_colors = min(n_patches + 1, len(_PATCH_COLORS))
    axes[1].imshow(
        np.clip(labeled, 0, n_colors - 1),
        cmap=ListedColormap(_PATCH_COLORS[:n_colors]),
        aspect="auto",
        vmin=0,
        vmax=n_colors - 1,
    )
    axes[1].set_title(f"(b) Connected patches ({n_patches})")

    axes[2].imshow(np.where(valid, raw_masked, np.nan), cmap="viridis", aspect="auto")
    axes[2].set_title(f"(c) Size filter ({n_kept} kept, {n_patches - n_kept} removed)")

    im = axes[3].imshow(packed, cmap="gray", aspect="auto", vmin=0, vmax=255)
    fig.colorbar(im, ax=axes[3], shrink=0.8)
    axes[3].set_title("(d) Packed uint8")

    for ax in axes:
        ax.set_xlabel("X in px")
    axes[0].set_ylabel("Y in px")
    fig.tight_layout()
    return fig


# ── pointcloud ────────────────────────────────────────────────────────────────


def plot_pointcloud_processing(
    z_2d: np.ndarray,
    lumi_2d: np.ndarray,
    processed_points: np.ndarray,
    *,
    stats: dict | None = None,
    max_points: int = 300_000,
    figsize: tuple[float, float] = FIGSIZE,
    **params,
) -> Figure:
    """Two top-view panels: the raw calibrated scan and the processed point cloud.

    The raw panel applies only the validity mask and the calibration (steps
    1–2 of the stage); the processed panel is the stage's output (cleaned,
    ICP-aligned to the matched simulation), so gaps between the two are the
    removed artifacts. Both are coloured by height.

    Args:
        z_2d: ``(H, W)`` raw z grid (sensor units).
        lumi_2d: ``(H, W)`` raw luminescence grid (for the validity mask).
        processed_points: ``(N, 3)`` processed ``pointcloud/<op>/z``.
        stats: The processed group's attrs; when given, removal counts are
            shown in the panel title.
        max_points: Subsampling cap per panel (keeps the figure light).
        figsize: Figure size in inches.
        **params: ``[pointcloud]`` parameters; only ``lumi_min_patch_size`` is used.
    """
    from .defaults import PC_LUMI_MIN_PATCH_SIZE
    from .pointcloud import geometry

    calib = geometry.load_calibration()
    min_patch = params.get("lumi_min_patch_size", PC_LUMI_MIN_PATCH_SIZE)
    z_2d, lumi_2d = np.asarray(z_2d), np.asarray(lumi_2d)
    valid = geometry.lumi_valid_mask(lumi_2d, min_patch) & (z_2d > 0)
    if not valid.any():
        raise ValueError("no valid pixels in the raw scan (check lumi_min_patch_size)")
    y_mm_per_px = geometry.y_calibration(valid, calib["x_mm_per_pixel"])
    raw_points = geometry.extract_points(z_2d, valid, calib["x_mm_per_pixel"], y_mm_per_px, calib["z_mm_per_unit"])
    processed_points = np.asarray(processed_points, dtype=float)

    rng = np.random.default_rng(0)

    def _subsample(points: np.ndarray) -> np.ndarray:
        if len(points) <= max_points:
            return points
        return points[rng.choice(len(points), size=max_points, replace=False)]

    raw_s, proc_s = _subsample(raw_points), _subsample(processed_points)
    z_all = np.concatenate([raw_s[:, 2], proc_s[:, 2]]) if len(proc_s) else raw_s[:, 2]
    vmin, vmax = (np.percentile(z_all, 1), np.percentile(z_all, 99)) if len(z_all) else (0, 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, pts in zip(axes, (raw_s, proc_s)):
        if len(pts):
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True
            )
        ax.set_aspect("equal")
        ax.set_xlabel("X in mm")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Y in mm")
    axes[0].set_title(f"(a) Raw, calibrated ({len(raw_points):,} points)")
    title = f"(b) Processed ({len(processed_points):,} points)"
    if stats and "outlier_pct" in stats:
        title += f", {float(stats['outlier_pct']):.1f} % removed"
    axes[1].set_title(title)
    if len(raw_s):
        fig.colorbar(sc, ax=axes, shrink=0.8, label="Z in mm")
    return fig


# ── command line (maintainer tool) ────────────────────────────────────────────


def _load_raw(exp_ids: Sequence[int], data_dir: str, group: str) -> tuple[list[np.ndarray], list[str] | None]:
    """Raw tables (and the ``columns`` attr of the last one) for the given ids."""
    from .h5_access import open_raw
    from .runner import _column_names

    tables, columns = [], None
    for exp_id in exp_ids:
        try:
            raw = open_raw(exp_id, data_dir)
        except FileNotFoundError:
            print(f"skipping {exp_id:04d}: no raw file")
            continue
        with raw:
            tables.append(raw[f"{group}/data"][:])
            columns = _column_names(raw[group].attrs, FORCE_COLUMNS)
    if not tables:
        raise SystemExit("no raw experiments found for the given selection")
    return tables, columns


def _load_raw_scan(exp_id: int, data_dir: str, op: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw ``(z_2d, lumi_2d)`` grids of one scan."""
    from .h5_access import open_raw

    with open_raw(exp_id, data_dir) as raw:
        grp = raw[f"pointcloud/{op}"]
        shape = (int(grp.attrs["y_shape"]), int(grp.attrs["x_shape"]))
        return grp["z"][:].reshape(shape), grp["luminescence"][:].reshape(shape)


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``<out>/<modality>_processing.<format>`` for one modality."""
    import argparse
    import os

    from ..spec import RDDAC_SPEC
    from . import config as config_mod
    from .runner import GROUPS, parse_ids

    parser = argparse.ArgumentParser(
        prog="python -m rddac._preprocess.visualize",
        description="Render the processing-step figures used in the documentation.",
    )
    sub = parser.add_subparsers(dest="modality", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=RDDAC_SPEC.default_data_dir, help="Raw dataset directory")
    common.add_argument("--out", required=True, help="Output directory for the image")
    common.add_argument("--config", metavar="TOML", help="Parameter overrides (as for rddac preprocess)")
    common.add_argument("--format", default="png", help="Image format (default: png)")
    common.add_argument("--dpi", type=int, default=150, help="Resolution for raster formats (default: 150)")

    p = sub.add_parser("oil", parents=[common], help="One experiment: raw, Hampel outliers, processed")
    p.add_argument("--id", type=int, required=True, help="Experiment id")
    for name in ("force", "sheet"):
        p = sub.add_parser(name, parents=[common], help="Overlay of several experiments, raw vs processed")
        p.add_argument("--ids", required=True, help="Experiment ids, e.g. '0-199' or '42,1035'")
    p = sub.add_parser("luminescence", parents=[common], help="One scan: raw grid, patches, mask, packed uint8")
    p.add_argument("--id", type=int, required=True, help="Experiment id")
    p.add_argument("--op", default="op10", choices=["op10", "op20"], help="Operation (default: op10)")
    p = sub.add_parser("pointcloud", parents=[common], help="One scan: raw calibrated vs processed point cloud")
    p.add_argument("--id", type=int, required=True, help="Experiment id")
    p.add_argument("--op", default="op10", choices=["op10", "op20"], help="Operation (default: op10)")
    p.add_argument("--processed-dir", default=None, help="Processed output directory (default: <data-dir>/processed)")

    args = parser.parse_args(argv)
    cfg = config_mod.load(args.config) if args.config else config_mod.defaults_config()
    n_exp = 1

    if args.modality == "oil":
        tables, _ = _load_raw([args.id], args.data_dir, GROUPS["oil"])
        fig = plot_oil_processing(tables[0], **cfg["oil"])
    elif args.modality in ("force", "sheet"):
        tables, columns = _load_raw(sorted(parse_ids(args.ids)), args.data_dir, GROUPS[args.modality])
        n_exp = len(tables)
        if args.modality == "force":
            fig = plot_force_processing(tables, columns, **cfg["force"])
        else:
            fig = plot_sheet_processing(tables, **cfg["sheet"])
    else:
        z_2d, lumi_2d = _load_raw_scan(args.id, args.data_dir, args.op)
        if args.modality == "luminescence":
            fig = plot_luminescence_processing(lumi_2d, **cfg["pointcloud"])
        else:
            import h5py

            processed_dir = args.processed_dir or os.path.join(args.data_dir, "processed")
            path = os.path.join(processed_dir, f"{args.id:04d}.h5")
            if not os.path.isfile(path):
                raise SystemExit(f"processed file not found: {path} — run `rddac preprocess pointcloud` first")
            with h5py.File(path, "r") as f:
                grp = f[f"pointcloud/{args.op}"]
                points, stats = grp["z"][:], dict(grp.attrs)
            fig = plot_pointcloud_processing(z_2d, lumi_2d, points, stats=stats, **cfg["pointcloud"])

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{args.op}" if args.modality in ("luminescence", "pointcloud") else ""
    path = os.path.join(args.out, f"{args.modality}_processing{suffix}.{args.format}")
    fig.savefig(path, format=args.format, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path} ({n_exp} experiment{'s' if n_exp != 1 else ''})")


if __name__ == "__main__":
    main()
