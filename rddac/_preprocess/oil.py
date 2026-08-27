"""Oil-film thickness preprocessing: raw ``oil_thickness/data`` → ``(200, 2) float32``.

Steps (validated in the internal pipeline on all experiments with oil):

1. Truncate to sensor positions < 200 mm (edge artifacts beyond the part).
2. Drop invalid (NaN) measurements — sensor dropouts; about two thirds of the
   experiments contain a handful of isolated ones. They are removed BEFORE the
   Hampel filter so they cannot poison its window statistics.
3. Hampel filter on the raw trace: local median ± k robust sigmas. NaN-free by
   construction; when the window MAD collapses to zero (flat, quantized trace)
   the scale falls back to the mean absolute deviation floored at the
   0.01 g/m^2 logging quantization, so isolated spikes are still caught but
   ±1 LSB wiggle never is.
4. Average duplicate positions (BF=100 experiments measure twice per position).
5. Fill the integer 0..199 mm grid, linear interpolation, nearest edge fill.
6. Cleaning statistics are returned for the h5 attributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .defaults import (
    OIL_HAMPEL_K,
    OIL_HAMPEL_WINDOW,
    OIL_MAX_SENSOR_POSITION,
    OIL_OUTPUT_LENGTH,
    OIL_VALUE_QUANTIZATION,
)

#: Column names of the processed table (also written as the h5 ``columns`` attr).
COLUMNS = ("sensor_position", "oil_value")
UNITS = ("mm", "g/m^2")


def hampel_filter(
    positions: np.ndarray,
    values: np.ndarray,
    window_size: int = OIL_HAMPEL_WINDOW,
    k: float = OIL_HAMPEL_K,
    quantization: float = OIL_VALUE_QUANTIZATION,
) -> np.ndarray:
    """Outlier mask for a 1D traverse via a position-windowed Hampel filter.

    Windows are selected by position distance (not index), so gaps from removed
    dropouts do not misalign them. In flat windows where the MAD is zero the
    scale falls back to the mean absolute deviation, floored at one logging
    quantization step.

    Args:
        positions: Integer sensor positions, one per measurement.
        values: Measured oil values (must not contain NaN).
        window_size: Half-window size in position units (mm).
        k: Threshold in robust sigmas.

    Returns:
        Boolean array; True marks an outlier.
    """
    outlier = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        window = values[np.abs(positions - positions[i]) <= window_size]
        median = np.median(window)
        abs_dev = np.abs(window - median)
        scale = 1.4826 * np.median(abs_dev)
        if scale == 0:
            # 1.2533 = sqrt(pi/2) makes the mean abs dev a consistent sigma estimate.
            scale = max(1.2533 * float(np.mean(abs_dev)), quantization)
        if np.abs(values[i] - median) > k * scale:
            outlier[i] = True
    return outlier


def process(
    data: np.ndarray,
    *,
    max_sensor_position: int = OIL_MAX_SENSOR_POSITION,
    output_length: int = OIL_OUTPUT_LENGTH,
    hampel_window: int = OIL_HAMPEL_WINDOW,
    hampel_k: float = OIL_HAMPEL_K,
    value_quantization: float = OIL_VALUE_QUANTIZATION,
) -> tuple[np.ndarray, dict]:
    """Process one experiment's raw oil table.

    Args:
        data: ``(n, 2)`` raw array of ``[sensor_position, oil_value]``.
        max_sensor_position: Positions at/after this (mm) are edge artifacts.
        output_length: Fixed output row count (one row per integer mm).
        hampel_window: Hampel half-window in mm.
        hampel_k: Hampel threshold in robust sigmas.
        value_quantization: Logging LSB; floors the flat-window fallback scale.

    Returns:
        ``(processed, attrs)`` — ``processed`` is ``(output_length, 2) float32``
        on the integer mm grid; ``attrs`` are the cleaning statistics plus the
        parameter values used (self-describing output).
    """
    df = pd.DataFrame(np.asarray(data, dtype=float), columns=list(COLUMNS))
    df = df.dropna(subset=["sensor_position"])
    df["pos_int"] = df["sensor_position"].round(0).astype(int)
    df = df[df["pos_int"] < max_sensor_position]
    n_in_range = len(df)

    n_nan = int(df["oil_value"].isna().sum())
    df = df.dropna(subset=["oil_value"]).sort_values("pos_int").reset_index(drop=True)

    outlier = hampel_filter(
        df["pos_int"].to_numpy(),
        df["oil_value"].to_numpy(),
        window_size=hampel_window,
        k=hampel_k,
        quantization=value_quantization,
    )
    n_outliers = int(outlier.sum())
    df = df[~outlier]

    df = df.groupby("pos_int", as_index=False)["oil_value"].mean()
    template = pd.DataFrame({"pos_int": range(max_sensor_position)})
    df = template.merge(df, on="pos_int", how="left")
    n_interpolated = int(df["oil_value"].isna().sum())
    df["oil_value"] = df["oil_value"].interpolate(method="linear").bfill().ffill()

    df = df.head(output_length)
    processed = df[["pos_int", "oil_value"]].to_numpy(dtype=np.float32)
    attrs = {
        "n_raw_in_range": n_in_range,
        "n_nan_removed": n_nan,
        "n_hampel_outliers": n_outliers,
        "n_positions_interpolated": n_interpolated,
        "max_sensor_position": max_sensor_position,
        "output_length": output_length,
        "hampel_window": hampel_window,
        "hampel_k": hampel_k,
        "value_quantization": value_quantization,
    }
    return processed, attrs
