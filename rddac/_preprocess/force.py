"""Force preprocessing: raw ``force/data`` → ``(600, 8) float32``.

Steps (validated in the internal pipeline):

1. Keep ALL raw columns — time, load_cell_1..4, punch_temp, punch_pos,
   total_force — so the published ``force-curve`` field mapping stays valid
   (punch_pos is the punch trajectory, informative for force-vs-position
   views; punch_temp is near-constant but costs nothing).
2. Trim to the forming window (0.25 s, 2.25 s] → fixed 600 rows at the 300 Hz
   recording rate; time is re-zeroed to the window start.
3. Remove the rest offset (blankholder preload baseline) from the load cells
   and the total force.
4. Rounding: time 4 decimals, forces 2 (kN), temperature 1 (degC); punch
   position quantized to the 0.5 mm encoder resolution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .defaults import (
    FORCE_COLUMNS,
    FORCE_KN_DECIMALS,
    FORCE_POSITION_PRECISION_MM,
    FORCE_SIGNALS,
    FORCE_TEMP_DECIMALS,
    FORCE_TIME_DECIMALS,
    FORCE_TIME_WINDOW,
)

COLUMNS = FORCE_COLUMNS

#: Physical unit per column, used for the h5 ``units`` attr.
UNITS_BY_COLUMN = {
    "time": "s",
    "load_cell_1": "kN",
    "load_cell_2": "kN",
    "load_cell_3": "kN",
    "load_cell_4": "kN",
    "punch_temp": "degC",
    "punch_pos": "mm",
    "total_force": "kN",
}


def units_for(columns) -> list[str]:
    """Units matching a column layout (empty string for unknown columns)."""
    return [UNITS_BY_COLUMN.get(name, "") for name in columns]


def process(
    data: np.ndarray,
    columns: Sequence[str] = FORCE_COLUMNS,
    *,
    time_window_start: float = FORCE_TIME_WINDOW[0],
    time_window_end: float = FORCE_TIME_WINDOW[1],
    position_precision_mm: float = FORCE_POSITION_PRECISION_MM,
) -> tuple[np.ndarray, dict]:
    """Process one experiment's raw force table.

    Args:
        data: ``(n, len(columns))`` raw array as stored under ``force/data``.
        columns: Column layout of ``data`` (pass the h5 ``columns`` attr when
            it differs from the default raw layout).
        time_window_start: Forming window start in s (exclusive).
        time_window_end: Forming window end in s (inclusive).
        position_precision_mm: Punch position quantization step.

    Returns:
        ``(processed, attrs)`` — ``processed`` is ``(m, len(columns)) float32``
        with the input column order preserved; ``attrs`` hold row counts plus
        the parameter values used (self-describing output).
    """
    df = pd.DataFrame(np.asarray(data, dtype=float), columns=list(columns))
    if "time" not in df.columns:
        raise ValueError(f"force table has no 'time' column: {list(columns)}")
    n_raw = len(df)

    df = df[(df["time"] > time_window_start) & (df["time"] <= time_window_end)].reset_index(drop=True)
    df["time"] = (df["time"] - df["time"].min()).round(FORCE_TIME_DECIMALS)

    for name in FORCE_SIGNALS:
        if name in df.columns:
            # Load cells rest at the blankholder preload, not at zero.
            df[name] -= np.abs(df[name].min())
            df[name] = df[name].round(FORCE_KN_DECIMALS)
    if "punch_temp" in df.columns:
        df["punch_temp"] = df["punch_temp"].round(FORCE_TEMP_DECIMALS)
    if "punch_pos" in df.columns:
        df["punch_pos"] = np.round(df["punch_pos"] / position_precision_mm) * position_precision_mm

    processed = df.to_numpy(dtype=np.float32)
    attrs = {
        "n_raw_rows": n_raw,
        "n_rows": len(df),
        "time_window_start": time_window_start,
        "time_window_end": time_window_end,
        "position_precision_mm": position_precision_mm,
    }
    return processed, attrs
