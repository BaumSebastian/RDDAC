# Dataset Overview

RDDAC contains **{{ experiment_count() }} physical deep-drawing and cutting experiments** on DP600 dual-phase steel, forming modified quadratic cups on an industrial press. Each experiment captures deep drawing (OP10) followed by cutting (OP20) with four raw measurement modalities. The full release is {{ total_size() }} ({{ per_experiment_size() }} per experiment, HDF5 + gzip).

| | |
|---|---|
| DOI | [10.18419/DARUS-5589](https://doi.org/10.18419/DARUS-5589) |
| Records | {{ experiment_count() }} experiments |
| Total size | {{ total_size() }} |
| File size | {{ per_experiment_size() }} per experiment |
| Format | HDF5 |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Parameter space

Three process parameters are varied on a full grid; every combination is a **category** repeated up to 500 times:

| Parameter | Values |
|-----------|--------|
| Geometry | `concave` (ids 0000-4499), `convex` (ids 4500-8999) |
| Blankholder force | 100 / 300 / 500 kN |
| Oil type | `coarse` / `medium` / `fine` lubrication pattern |

2 geometries x 3 blankholder forces x 3 oil types = **{{ category_count() }} categories** x up to 500 repetitions = {{ experiment_count() }} experiments. Unlike DDACS, where every simulation has distinct input parameters, RDDAC's value lies in the **repetitions**: hundreds of nominally identical experiments expose the natural scatter of the physical process.

{{ experiment_stats() }}

<img src="../images/agg_force_by_bhf.png" width="700">

*Total press force over time (concave geometry, mean of 25 experiments per combination) — color encodes the blankholder force, line style the oil type.*

## Operations

Each experiment runs two operations end to end. OP10 is the deep drawing step: the blank holder presses the sheet against the die, the punch travels down, and the tools are released. OP20 then cuts the formed cup. The part is laser-scanned after each operation, so both intermediate and final geometry are captured.

## Measurement modalities

Four raw modalities are recorded per experiment (see [HDF5 structure](hdf5-structure.md) for exact shapes and units). The single-experiment figures below all show experiment `0000`; the traverse heatmaps show category 2 (concave, 100 kN, fine oil).

**Press force signals** — four load cells positioned around the forming die sample the force at 300 Hz while the punch descends in OP10. The same table carries the punch temperature, the punch position, and the summed total force, so one figure shows the complete press state over the stroke:

<img src="../images/example_force.png" width="700">

*Press force signals of experiment `0000`: the four load cells, their total, and the punch position and temperature over the stroke.*

Across repetitions the force signals also expose slow process drift, such as the punch warming up over the course of a measurement campaign:

<img src="../images/agg_punch_temp.png" width="700">

*Punch temperature over repetitions, one panel per geometry — color encodes the blankholder force, line style the oil type.*

Before forming, two line measurements are taken on the flat blank — the sheet thickness on the bare blank, the oil film after the lubricant is applied:

<img src="../images/scheme_measurement_lines.png" width="700">

*One measurement line per part on the flat blank: sheet thickness before lubrication, oil film after.*

**Sheet thickness traverse** — a thickness sensor traverses the blank along that line and records the material thickness in µm, capturing the manufacturing tolerances of the sheet metal coil.

<img src="../images/example_sheet.png" width="700">

*Sheet thickness traverse of experiment `0000`.*

<img src="../images/agg_sheet_heatmap.png" width="700">

*Sheet thickness traverses of all parts of category 2 side by side — the coil-to-coil variation of the raw material.*

**Oil film traverse** — an AMEPA oil film meter traverses the lubricated blank and records the oil area density in g/m². The three oil types of the parameter grid differ in this applied pattern.

<img src="../images/example_oil.png" width="700">

*Oil film traverse of experiment `0000`.*

<img src="../images/agg_oil_heatmap.png" width="700">

*Oil film traverses of all parts of category 2 side by side — the scatter of the lubrication pattern across repetitions.*

**3D laser scans** — a Keyence LJ-X8400 laser line scanner captures the formed part after deep drawing (OP10) and again after cutting (OP20), recording a height and a luminescence buffer on a 3200 x 2000 pixel grid (6.4 million points per scan).

<img src="../images/example_scan_op20.png" width="700">

*OP20 laser scan (height buffer) of experiment `0000` — the cut cup on the magnetic gripper surface.*

The same buffers plot directly as a 3D point cloud — `rddac.scan_to_pointcloud` turns a scan into `(N, 3)` points and `rddac.plot_point_cloud` renders them (see the [visualization tutorial](tutorials/visualization.md)):

<img src="../images/example_point_cloud.png" width="700">

*OP10 scan of experiment `0000` as a 3D point cloud. The sparse bands on the steep cup walls are pixels without a laser return — raw data, no cleaning applied.*

!!! note "Raw sensor data"
    The scan `z` and `luminescence` buffers are stored in **uncalibrated sensor units**, and the number of samples `n` in the force and traverse tables varies per experiment — this is deliberately the raw data as recorded. An optional preprocessing step (calibration to mm, outlier cleaning, and alignment to the DDACS simulation frame) is planned for package v1.1 and is not part of the current release.

## Missing measurements

Not every experiment carries every modality. {{ missing_pointcloud() }} experiments lack the point cloud scans and {{ missing_oil() }} lack the oil measurement. Two boolean columns in `process_parameters.csv` (mirrored as HDF5 root attributes) flag availability:

| Flag | `False` count | Missing HDF5 group |
|------|---------------|--------------------|
| `has_pointcloud` | {{ missing_pointcloud() }} | `pointcloud/` |
| `has_oil` | {{ missing_oil() }} | `oil_thickness/` |

Filter them out before streaming a view that touches the affected groups, e.g. `where=lambda row: row["has_oil"]` — see [Process parameters](process-parameters.md#filtering-recipe).

## Relationship to DDACS

RDDAC is the experimental counterpart to the [DDACS](https://ddacs.readthedocs.io) dataset of LS-DYNA simulations: same modified quadratic cup, same DP600 steel, same two-stage OP10/OP20 process. The DDACS release contains a matching sub study (the `rddac` column of its `process_parameters.csv`) whose simulations correspond to the RDDAC parameter grid; they are published in the DDACS dataset as a single archive, `rddac.zip` ({{ simulation_download_size() }}, [doi:10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801)).

`rddac download` fetches those simulations alongside the measurements into `./data/simulation` by default; pass `--no-sim` to skip them. The accompanying paper, [*Statistical Analysis of Simulation to Reality Deviation in Deep Drawing with a Benchmark Dataset*](https://doi.org/10.1007/s12666-026-03870-5), characterises the deviation between the two.

The `rddac` package API mirrors `ddacs` one to one, so analysis code moves between the two datasets by swapping the import.

## Files on DaRUS

{{ darus_files_table() }}

The names, sizes, and descriptions above come from the [Croissant manifest](croissant.md), which is generated from the DaRUS file records. The three zips partition the {{ experiment_count() }} experiments: the 18 ids in `sample.zip` are **not** repeated in the geometry zips. HDF5 members are zero-padded to four digits (experiment 42 -> `0042.h5`).

## Further reading

- [Process parameters](process-parameters.md): categories, splits, and the columns of `process_parameters.csv`.
- [HDF5 structure](hdf5-structure.md): every field, shape, and unit inside one experiment.
- [Croissant manifest](croissant.md): the machine readable schema the package and external tools read.
