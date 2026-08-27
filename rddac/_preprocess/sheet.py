"""Sheet-thickness preprocessing: raw ``sheet_thickness/data`` → ``(200, 2) float32``.

Steps (validated in the internal pipeline):

1. Keep the last 200 traverse positions (the stable sensor region).
2. Normalize sensor positions to start at 0.
3. Mask negative values as NaN — the sensor reports large negative error
   codes where it lost contact. They are masked, not interpolated: consumers
   decide how to treat missing thickness.
4. Quantization: thickness to 0.01 um, position to 0.01 mm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .defaults import (
    SHEET_LAST_N,
    SHEET_POSITION_PRECISION_MM,
    SHEET_THICKNESS_PRECISION_UM,
)

COLUMNS = ("sensor_position", "sheet_thickness")
UNITS = ("mm", "um")


def process(data: np.ndarray, *, last_n: int = SHEET_LAST_N) -> tuple[np.ndarray, dict]:
    """Process one experiment's raw sheet-thickness table.

    Args:
        data: ``(n, 2)`` raw array of ``[sensor_position, sheet_thickness]``.
        last_n: How many trailing traverse positions to keep.

    Returns:
        ``(processed, attrs)`` — ``processed`` is ``(<=last_n, 2) float32``
        with NaN where the sensor reported error values; ``attrs`` hold the
        counts plus the parameter values used (self-describing output).
    """
    df = pd.DataFrame(np.asarray(data, dtype=float), columns=list(COLUMNS))
    n_raw = len(df)
    df = df.tail(last_n).reset_index(drop=True)

    df["sensor_position"] -= df["sensor_position"].min()
    n_negative = int((df["sheet_thickness"] < 0).sum())
    df.loc[df["sheet_thickness"] < 0, "sheet_thickness"] = np.nan

    df["sheet_thickness"] = (
        np.round(df["sheet_thickness"] / SHEET_THICKNESS_PRECISION_UM) * SHEET_THICKNESS_PRECISION_UM
    ).round(2)
    df["sensor_position"] = (
        np.round(df["sensor_position"] / SHEET_POSITION_PRECISION_MM) * SHEET_POSITION_PRECISION_MM
    ).round(2)

    processed = df.to_numpy(dtype=np.float32)
    attrs = {"n_raw_rows": n_raw, "n_rows": len(df), "n_negative_masked": n_negative, "last_n": last_n}
    return processed, attrs
