# Press Force

The `force` stage trims the raw press signals (`force/data`) to the forming window and removes the load-cell rest offset, producing a fixed 600-row table. It runs as part of [`rddac preprocess`](../cli.md#rddac-preprocess); parameters can be overridden via [`--config`](../cli.md#rddac-preprocess).

| Property | Raw | Processed |
| --- | --- | --- |
| Shape | variable, ~(1140, 8) or ~(720, 8) depending on press speed | fixed `(600, 8) float32` |
| Columns | `time`, `load_cell_1..4`, `punch_temp`, `punch_pos`, `total_force` | **unchanged, all 8 kept** |
| Time range | 0 to ~3.8 s | 0 to 2 s (re-zeroed) |
| Sampling rate | 300 Hz | unchanged |

## Why this stage exists

The raw recording spans the complete machine cycle (startup, forming, retraction), but only the forming phase carries process information, and its position inside the recording varies with the blankholder force (lower force → slower cycle). The load cells also rest at the blankholder preload rather than zero, so absolute values are not comparable across experiments until the baseline is removed.

**All eight columns are kept.** `punch_pos` is the punch trajectory, the natural kinematic axis for force-vs-position views. `punch_temp` is near-constant over one stroke (its mean is already a root attribute), but keeping it costs nothing and preserves the published `force-curve` field mapping unchanged.

## Processing steps

1. **Window trimming**: samples outside $(0.25\ \text{s},\ 2.25\ \text{s}]$ are startup/shutdown noise and are dropped; at 300 Hz the window yields exactly 600 rows. Time is re-zeroed to the window start.
2. **Rest-offset removal**: each load cell (and the total force) has its in-window minimum subtracted, so every curve starts from zero.
3. **Quantization**: time to 4 decimals (s), forces to 0.01 kN, temperature to 0.1 °C, punch position to the 0.5 mm encoder resolution.

<img src="../../images/preprocessing/force_processing.png" width="700">

*Raw force curves with the forming-window boundaries (left) and the processed, offset-free curves on the re-zeroed time axis (right).*

## Parameters

| TOML key (`[force]`) | Default | Meaning |
| --- | --- | --- |
| `time_window_start` | 0.25 | window start in s (exclusive) |
| `time_window_end` | 2.25 | window end in s (inclusive) |
| `position_precision_mm` | 0.5 | punch-position quantization step |

Row counts and the parameter values used are stamped into the `force` group attributes.

## Further reading

- [Preprocessing overview](index.md): quickstart, output rules, reproducibility model
- [Custom processing](custom.md): replace this stage with your own algorithm
- [HDF5 structure](../hdf5-structure.md): the raw `force` group
