"""User-adjustable processing parameters: defaults, TOML load, TOML dump.

The values in :mod:`.defaults` are the citable reference recipe — running
``rddac preprocess`` without a config reproduces the published processed
data. A user who wants a variant passes ``--config my.toml`` and publishes
the TOML next to their code: raw dataset DOI + rddac version + TOML
reproduces their processed dataset exactly. ``--dump-config`` prints the
complete defaults as a ready-to-edit template.

Every value actually used is also stamped into the h5 group attrs by the
modality functions, so a processed file stays self-describing even when it
gets separated from the code that made it.
"""

from __future__ import annotations

import sys
from copy import deepcopy

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — py3.10 fallback (tomli dependency via pyproject marker)
    import tomli as tomllib

from . import defaults as d

#: The complete configurable surface: table → key → default. Keys map 1:1 to
#: the keyword arguments of the modality ``process()`` functions.
DEFAULTS: dict[str, dict] = {
    "oil": {
        "max_sensor_position": d.OIL_MAX_SENSOR_POSITION,
        "output_length": d.OIL_OUTPUT_LENGTH,
        "hampel_window": d.OIL_HAMPEL_WINDOW,
        "hampel_k": d.OIL_HAMPEL_K,
        "value_quantization": d.OIL_VALUE_QUANTIZATION,
    },
    "force": {
        "time_window_start": d.FORCE_TIME_WINDOW[0],
        "time_window_end": d.FORCE_TIME_WINDOW[1],
        "position_precision_mm": d.FORCE_POSITION_PRECISION_MM,
    },
    "sheet": {
        "last_n": d.SHEET_LAST_N,
    },
    "pointcloud": {
        "lumi_min_patch_size": d.PC_LUMI_MIN_PATCH_SIZE,
        "max_wall_angle_concave_deg": d.PC_MAX_WALL_ANGLE_CONCAVE_DEG,
        "max_wall_angle_convex_deg": d.PC_MAX_WALL_ANGLE_CONVEX_DEG,
        "z_tolerance_mm": d.PC_Z_TOLERANCE_MM,
        "min_component_size": d.PC_MIN_COMPONENT_SIZE,
        "k_angle": d.PC_K_ANGLE,
        "k_mono": d.PC_K_MONO,
        "k_closing": d.PC_K_CLOSING,
        "max_closing_iter": d.PC_MAX_CLOSING_ITER,
        "icp_max_iterations": d.PC_ICP_MAX_ITERATIONS,
        "icp_sample_size": d.PC_ICP_SAMPLE_SIZE,
        "rf_threshold": d.PC_RF_THRESHOLD,
        "rf_n_estimators": d.PC_RF_N_ESTIMATORS,
        "rf_max_depth": d.PC_RF_MAX_DEPTH,
        "keep_prepared": d.PC_KEEP_PREPARED,
    },
}


def defaults_config() -> dict[str, dict]:
    """A fresh copy of the default configuration."""
    return deepcopy(DEFAULTS)


def load(path: str) -> dict[str, dict]:
    """Merge a user TOML file over the defaults.

    Args:
        path: Path to the TOML file.

    Returns:
        The complete configuration (defaults with the file's overrides).

    Raises:
        ValueError: On unknown tables or keys — a silently ignored typo would
            mean silently NOT applying the user's intended override.
        OSError: If the file cannot be read.
    """
    with open(path, "rb") as f:
        user = tomllib.load(f)

    cfg = defaults_config()
    for table, values in user.items():
        if table not in cfg:
            raise ValueError(f"unknown config table [{table}] — valid: {', '.join(cfg)}")
        if not isinstance(values, dict):
            raise ValueError(f"[{table}] must be a table of key = value pairs")
        for key, value in values.items():
            if key not in cfg[table]:
                raise ValueError(f"unknown key '{key}' in [{table}] — valid: {', '.join(cfg[table])}")
            cfg[table][key] = value
    return cfg


def dump() -> str:
    """The complete default configuration as a ready-to-edit TOML string."""
    lines = [
        "# rddac preprocess configuration: these are the defaults (the",
        "# reference recipe). Edit values and pass the file via",
        "#   rddac preprocess --config my.toml",
        "# Publish it next to your code to make the variant reproducible.",
        "",
    ]
    for table, values in DEFAULTS.items():
        lines.append(f"[{table}]")
        for key, value in values.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)
