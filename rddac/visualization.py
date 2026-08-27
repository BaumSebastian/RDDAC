"""Visualization utilities for RDDAC experimental data.

Plotting helpers operate on numpy arrays the caller already pulled out of an
HDF5 file. Pair them with `rddac.open_h5` for end-to-end inspection:

    >>> import rddac
    >>> with rddac.open_h5(42, data_dir="./data") as f:
    ...     z = f["pointcloud/op10/z"][:]
    ...     lumi = f["pointcloud/op10/luminescence"][:]
    ...     force = f["force/data"][:]
    >>> ax, cbar = rddac.plot_scan(z)                      # heightmap image
    >>> ax = rddac.plot_force(force)                       # force time series
    >>> pts = rddac.scan_to_pointcloud(z, lumi)            # flat buffer -> (N, 3)
    >>> ax, cbar = rddac.plot_point_cloud(pts, values=pts[:, 2])

The raw scans are flattened row-major buffers over a ``y_shape x x_shape``
(2000 x 3200) pixel grid — see the group attrs in the h5 file. Pixel values
are sensor units; the calibrated mm conversion is part of the (optional)
preprocessing step.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Default scan grid (Keyence line scanner): see pointcloud group attrs.
SCAN_X_SHAPE = 3200
SCAN_Y_SHAPE = 2000

# Force table column layout (see the `columns` attr on the h5 `force` group).
FORCE_COLUMNS = (
    "time",
    "load_cell_1",
    "load_cell_2",
    "load_cell_3",
    "load_cell_4",
    "punch_temp",
    "punch_pos",
    "total_force",
)

DEFAULT_DPI = 150


def _scan_to_2d(buffer: np.ndarray, x_shape: int, y_shape: int) -> np.ndarray:
    buffer = np.asarray(buffer)
    if buffer.ndim == 1:
        return buffer.reshape(y_shape, x_shape)
    if buffer.ndim == 2:
        return buffer
    raise ValueError(f"expected a flat or 2D scan buffer, got shape {buffer.shape}")


def plot_scan(
    z: np.ndarray,
    ax=None,
    figsize: tuple[float, float] | None = None,
    x_shape: int = SCAN_X_SHAPE,
    y_shape: int = SCAN_Y_SHAPE,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    mask_zero: bool = True,
    colorbar: bool = True,
    title: str | None = None,
):
    """Show a raw laser scan buffer (height or luminescence) as a 2D image.

    Args:
        z: Flat ``(x_shape*y_shape,)`` or already-reshaped ``(y_shape, x_shape)``
            scan buffer, e.g. ``f["pointcloud/op10/z"][:]``.
        ax: Optional existing matplotlib Axes.
        figsize: Figure size when a new figure is created.
        x_shape / y_shape: Grid dimensions (defaults match the group attrs).
        cmap: Matplotlib colormap name.
        vmin / vmax: Color range; defaults to the data range of valid pixels.
        mask_zero: Render 0-valued pixels (no measurement) as transparent.
        colorbar: Attach a colorbar.
        title: Optional axes title.

    Returns:
        ``(ax, colorbar)`` — the colorbar is ``None`` when ``colorbar=False``.
    """
    img = _scan_to_2d(z, x_shape, y_shape).astype(float)
    if mask_zero:
        img = np.where(img == 0, np.nan, img)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (12, 7.5), dpi=DEFAULT_DPI)
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel("x in px")
    ax.set_ylabel("y in px")
    if title:
        ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8) if colorbar else None
    return ax, cbar


def scan_to_pointcloud(
    z: np.ndarray,
    luminescence: np.ndarray | None = None,
    x_shape: int = SCAN_X_SHAPE,
    y_shape: int = SCAN_Y_SHAPE,
    drop_invalid: bool = True,
    stride: int = 1,
) -> np.ndarray:
    """Convert a flat scan buffer into an ``(N, 3)`` point cloud ``[x, y, z]``.

    Coordinates are pixel indices and raw z units (uncalibrated); the mm
    conversion, outlier removal and simulation alignment belong to the
    (optional) preprocessing step.

    Args:
        z: Flat or 2D height buffer.
        luminescence: Optional matching buffer; when given, pixels with zero
            luminescence are treated as invalid too.
        x_shape / y_shape: Grid dimensions.
        drop_invalid: Drop pixels with ``z == 0`` (no measurement).
        stride: Keep every ``stride``-th pixel in both axes (fast previews;
            ``stride=4`` reduces 6.4M points to ~400k).

    Returns:
        ``(N, 3)`` float32 array of ``[x_px, y_px, z_raw]``.
    """
    z2d = _scan_to_2d(z, x_shape, y_shape)[::stride, ::stride]
    valid = np.ones_like(z2d, dtype=bool)
    if drop_invalid:
        valid &= z2d != 0
    if luminescence is not None:
        lumi2d = _scan_to_2d(luminescence, x_shape, y_shape)[::stride, ::stride]
        valid &= lumi2d != 0
    yy, xx = np.nonzero(valid)
    pts = np.column_stack([xx * stride, yy * stride, z2d[yy, xx]]).astype(np.float32)
    return pts


def plot_point_cloud(
    points: np.ndarray,
    values: np.ndarray | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 1.0,
    max_points: int | None = 200_000,
    colorbar: bool = True,
    title: str | None = None,
):
    """Scatter an ``(N, 3)`` point cloud in 3D, optionally colored by ``values``.

    Signature mirrors ``ddacs.plot_point_cloud`` so DDACS code ports by
    swapping the import. Use :func:`scan_to_pointcloud` to build ``points``
    from a raw scan buffer, or plot processed clouds directly.

    Args:
        points: ``(N, 3)`` array of xyz positions.
        values: Optional per-point scalars for coloring (default: z).
        ax: Optional existing 3D Axes.
        figsize: Figure size when a new figure is created.
        cmap / vmin / vmax: Color mapping.
        point_size: Scatter marker size.
        max_points: Random-subsample cap (``None`` = plot everything).
        colorbar: Attach a colorbar.
        title: Optional axes title.

    Returns:
        ``(ax, colorbar)`` — the colorbar is ``None`` when ``colorbar=False``.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {points.shape}")
    if values is None:
        values = points[:, 2]
    values = np.asarray(values)

    if max_points is not None and len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, values = points[idx], values[idx]

    if ax is None:
        fig = plt.figure(figsize=figsize or (10, 8), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=point_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.6) if colorbar else None
    return ax, cbar


def plot_force(
    force: np.ndarray,
    columns: tuple[str, ...] = FORCE_COLUMNS,
    signals: tuple[str, ...] = ("load_cell_1", "load_cell_2", "load_cell_3", "load_cell_4", "total_force"),
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Plot press force / process signals over time from a ``force/data`` table.

    Args:
        force: ``(n, 8)`` array as stored under ``force/data``.
        columns: Column layout (defaults to the raw layout; pass the h5 group's
            ``columns`` attr when it differs, e.g. for preprocessed data).
        signals: Which columns to draw against ``time``.
        ax: Optional existing Axes.
        figsize: Figure size when a new figure is created.
        title: Optional axes title.

    Returns:
        The matplotlib Axes.
    """
    force = np.asarray(force)
    col = {name: i for i, name in enumerate(columns)}
    if "time" not in col:
        raise ValueError(f"columns must contain 'time', got {columns}")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (10, 5), dpi=DEFAULT_DPI)
    t = force[:, col["time"]]
    for name in signals:
        if name not in col:
            raise ValueError(f"unknown signal {name!r}; available: {sorted(col)}")
        ax.plot(t, force[:, col[name]], label=name, linewidth=1)
    ax.set_xlabel("time in s")
    ax.set_ylabel("force in kN")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


def plot_traverse(
    data: np.ndarray,
    ax=None,
    figsize: tuple[float, float] | None = None,
    ylabel: str = "value",
    label: str | None = None,
    title: str | None = None,
):
    """Plot a sensor traverse table (``sheet_thickness/data`` or ``oil_thickness/data``).

    Args:
        data: ``(n, 2)`` array of ``[sensor_position, value]``.
        ax: Optional existing Axes.
        figsize: Figure size when a new figure is created.
        ylabel: Y axis label (e.g. ``"sheet thickness in um"``, ``"oil in g/m^2"``).
        label: Optional line label (for overlaying multiple traverses).
        title: Optional axes title.

    Returns:
        The matplotlib Axes.
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"expected (n, 2) traverse, got {data.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (10, 4), dpi=DEFAULT_DPI)
    ax.plot(data[:, 0], data[:, 1], linewidth=1, label=label)
    ax.set_xlabel("sensor position in mm")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(loc="best", fontsize="small")
    if title:
        ax.set_title(title)
    return ax
