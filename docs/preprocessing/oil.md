# Oil-Film Thickness

The `oil` stage turns the raw oil-film traverse (`oil_thickness/data`) into a fixed-length, dropout-free profile on the integer millimeter grid, one row per position 0–199 mm. It runs as part of [`rddac preprocess`](../cli.md#rddac-preprocess); every parameter below can be overridden via [`--config`](../cli.md#rddac-preprocess).

| Property | Raw | Processed |
| --- | --- | --- |
| Shape | variable, ~(420, 2) at BF 100 kN (two readings per position), ~(210, 2) otherwise | fixed `(200, 2) float32` |
| Columns | `sensor_position`, `oil_value` | unchanged |
| Position range | 0 to ~210 mm | 0 to 199 mm (integer grid) |
| Units | mm, g/m² | unchanged |
| Invalid values | NaN dropouts | none (removed + interpolated) |

## Why this stage exists

About two thirds of the experiments contain **sensor dropouts**: isolated NaN readings, typically one to six per trace, almost always at single positions. A naive robust filter is blinded by them: a NaN inside a Hampel window makes the window median NaN, silently disabling outlier detection within ±5 mm of every dropout, precisely where spikes tend to appear. The published files carry the data exactly as recorded, so this cleaning belongs to preprocessing.

## Processing steps

1. **Truncation**: positions at/after 200 mm are edge artifacts beyond the part and are dropped.
2. **Dropout removal**: NaN readings are removed *before* filtering so they cannot poison the Hampel windows.
3. **Hampel filter**: each reading is compared against its position neighborhood (±5 mm). A reading is an outlier when

    $$ |x_i - \tilde{x}_i| > k \cdot \sigma_i, \qquad \sigma_i = 1.4826 \cdot \operatorname{MAD}_i $$

    where $\tilde{x}_i$ is the window median and $k = 3$. The sensor logs two decimals, so flat windows can collapse to $\operatorname{MAD} = 0$; the scale then falls back to the mean absolute deviation, floored at one logging quantization step $q = 0.01\ \text{g/m}^2$:

    $$ \sigma_i = \max\!\left(\sqrt{\tfrac{\pi}{2}}\; \overline{|x - \tilde{x}_i|},\; q\right) $$

    so isolated spikes are still caught while ±1 LSB wiggle is never flagged.

4. **Duplicate averaging**: BF 100 kN experiments measure each position twice; remaining readings are averaged per integer position.
5. **Grid fill**: the integer 0–199 mm grid is completed by linear interpolation, with nearest-value fill at the edges.

<img src="../../images/preprocessing/oil_processing.png" width="700">

*Oil-film processing of one experiment: raw readings with the 200 mm truncation line (left), Hampel-flagged outliers (middle), the final fixed-grid profile (right).*

## Parameters

| TOML key (`[oil]`) | Default | Meaning |
| --- | --- | --- |
| `max_sensor_position` | 200 | positions at/after this (mm) are edge artifacts |
| `output_length` | 200 | fixed output row count |
| `hampel_window` | 5 | half-window in mm |
| `hampel_k` | 3.0 | threshold in robust sigmas |
| `value_quantization` | 0.01 | logging LSB (g/m²), floors the flat-window scale |

Every value used is stamped into the `oil_thickness` group attributes together with the cleaning counts (`n_nan_removed`, `n_hampel_outliers`, `n_positions_interpolated`), so a processed file stays self-describing.

## Further reading

- [Preprocessing overview](index.md): quickstart, output rules, reproducibility model
- [Custom processing](custom.md): replace this stage with your own algorithm
- [HDF5 structure](../hdf5-structure.md): the raw `oil_thickness` group
