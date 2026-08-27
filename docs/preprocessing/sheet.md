# Sheet Thickness

The `sheet` stage stabilizes the raw thickness traverse (`sheet_thickness/data`): it keeps the reliable tail of the scan, normalizes the position axis, and masks sensor error codes. It runs as part of [`rddac preprocess`](../cli.md#rddac-preprocess); parameters can be overridden via [`--config`](../cli.md#rddac-preprocess).

| Property | Raw | Processed |
| --- | --- | --- |
| Shape | variable, ~(208, 2) | fixed `(200, 2) float32` |
| Columns | `sensor_position`, `sheet_thickness` | unchanged |
| Position range | ~50 to ~257 mm | 0 to ~100 mm (normalized) |
| Units | mm, µm | unchanged |
| Invalid values | large negative error codes (e.g. −103697) | `NaN` |

## Why this stage exists

The traverse starts at the workpiece edge, so the first few readings are initialization artifacts, and wherever the sensor loses contact it reports **large negative error codes** instead of a thickness. Those must not leak into statistics: the mean sheet thickness of an experiment feeds the simulation matching of the [pointcloud stage](pointcloud.md).

## Processing steps

1. **Tail selection**: only the last 200 readings are kept; the discarded head is the unstable initialization region.
2. **Position normalization**: positions are shifted to start at 0, giving all experiments the same spatial reference.
3. **Error-code masking**: negative readings become `NaN`. They are masked, **not interpolated**: how to treat missing thickness is a modeling decision that belongs to the consumer, not to preprocessing.
4. **Quantization**: thickness to 0.01 µm, position to 0.01 mm (sensor datasheet resolutions).

<img src="../../images/preprocessing/sheet_processing.png" width="700">

*Raw thickness traverses with the tail-selection cutoff (left) and the processed, position-normalized profiles (right).*

## Parameters

| TOML key (`[sheet]`) | Default | Meaning |
| --- | --- | --- |
| `last_n` | 200 | trailing readings to keep |

The count of masked error codes (`n_negative_masked`) and the parameters used are stamped into the `sheet_thickness` group attributes.

## Further reading

- [Preprocessing overview](index.md): quickstart, output rules, reproducibility model
- [Custom processing](custom.md): replace this stage with your own algorithm
- [HDF5 structure](../hdf5-structure.md): the raw `sheet_thickness` group
