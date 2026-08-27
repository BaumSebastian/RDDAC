# Point Clouds

The `pointcloud` stage turns the raw laser-scan grids (`pointcloud/{op10,op20}/z` + `luminescence`) into cleaned, simulation-aligned point clouds: `z` becomes `(N, 3) float32` (Open3D-ready) and `luminescence` a `(2000, 3200) uint8` grayscale image (Pillow-ready). It is the only stage that needs the optional dependencies (`pip install 'rddac[preprocessing]'`) and the DDACS simulations.

!!! warning "Needs the DDACS simulations"
    Every scan is aligned to its best-matching DDACS simulation, so the stage requires the simulations fetched by `rddac download` (without `--no-sim`). Without them the command exits with an actionable error before touching any file.

| Property | Raw | Processed |
| --- | --- | --- |
| `z` | flattened `(6400000,)` grid, uncalibrated sensor units | `(N, 3) float32`, mm, cleaned + ICP-aligned |
| `luminescence` | flattened `(6400000,)` grid | `(2000, 3200) uint8`, 0 = background, valid pixels 1–255 |
| Frame | scanner table frame | matched DDACS simulation frame |

## Why this stage exists

Raw scans carry measurement artifacts, most prominently **fins**: locally smooth, wing-like surfaces at draw-in and cut edges that are not part of the physical part. They survive naive filters precisely because they are locally smooth; detecting them reliably needs cross-sample context. The scans are also uncalibrated and each sits in its own scanner frame, so cross-experiment or sim-to-real comparisons need calibration and registration first.

## Processing steps

1. **Validity mask**: connected-component filtering of the luminescence grid plus `z > 0`.

    <img src="../../images/preprocessing/luminescence_processing_op10.png" width="900">

    *Luminescence of one OP10 scan: raw grid (a), connected foreground patches (b), the validity mask after the patch-size filter (c), and the packed `uint8` image written to the processed file (d).*

2. **Calibration**: x from the sensor specification, z from the calibration-block measurement (both packaged in `calibration.json`), y per scan from the square-part assumption.
3. **Geometric outlier seeds**: three complementary detectors, then morphological closing on a kNN graph:
    - *surface angle*: local SVD plane fits; a point is seeded when its normal deviates more than the cutoff from vertical,

        $$ \theta = \arccos\left(|n_z|\right) > \theta_{\max}, \qquad \theta_{\max} = 70^\circ \text{ (concave)}, \; 80^\circ \text{ (convex)} $$

        The cutoffs sit ≥ 2° above the process limits; propagating the scanner's 5 µm z-repeatability through the gradient gives an angle uncertainty well below that margin (> 4σ).

    - *radial monotonicity*: on a formed cup, z must not increase moving outward from the part center;
    - *small 3D components*: floating clusters below the minimum size.
4. **Simulation matching + ICP**: the matching simulation is selected by geometry and blankholder force (hard) plus sheet thickness and oil-derived friction coefficient (soft, nearest); the scan is rigidly aligned to it (seeded, reproducible ICP).
5. **RF fin classifier**: a random forest over local surface features plus three cross-sample features in a registered common frame: a position prior $P(\text{outlier}\mid\text{position})$, the slope-normalized deviation from the consensus surface, and the registered coordinates. Removal uses `predict_proba` against a configurable threshold, the precision/recall knob.
6. **Final sweep**: small 3D components orphaned by the removal are dropped.

<img src="../../images/preprocessing/pointcloud_processing_op10.png" width="700">

*One OP10 scan in top view, coloured by height: the raw calibrated scan after the validity mask (left) and the processed, ICP-aligned point cloud (right); the gaps are the removed fins and outliers.*

## The classifier is retrained on your machine

Trained models are derived artifacts and are **not distributed**. The bundled labels are annotations of the dataset and, unlike the code, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribute the RDDAC dataset when you reuse them. On first use the stage retrains them deterministically (seed 42) from the labels bundled with the package: human-labeled outlier masks for 140 (experiment, operation) tasks, and caches them under `<data_dir>/models/pointcloud_fin_rf/` stamped with the scikit-learn version and the label fingerprint. The one-time cost is roughly 30–90 minutes; `--rebuild-models` forces a retrain. Held-out 5-fold cross-validation of the resulting cleaner:

| Group | Precision | Recall |
| --- | --- | --- |
| concave OP10 | 0.87 | 0.97 |
| concave OP20 | 0.86 | 0.94 |
| convex OP10 | 0.94 | 0.99 |
| convex OP20 | 0.96 | 0.96 |

## Parameters

| TOML key (`[pointcloud]`) | Default | Meaning |
| --- | --- | --- |
| `lumi_min_patch_size` | 20040 | luminescence patch filter (px) |
| `max_wall_angle_concave_deg` / `..._convex_deg` | 70 / 80 | surface-angle cutoffs |
| `z_tolerance_mm` | 1.0 | radial monotonicity tolerance |
| `min_component_size` | 50 | 3D component filter (seed stage + final sweep) |
| `k_angle` / `k_mono` / `k_closing` | 15 / 20 / 8 | kNN sizes |
| `max_closing_iter` | 15 | upper bound on morphological-closing iterations |
| `icp_max_iterations` / `icp_sample_size` | 50 / 50000 | ICP effort |
| `rf_threshold` | 0.5 | fin-classifier decision threshold (higher = more conservative removal) |
| `rf_n_estimators` / `rf_max_depth` | 150 / 18 | RF hyperparameters (changing them retrains) |
| `keep_prepared` | false | keep the prepared training grids after a successful retrain (debugging aid) |

Alignment (ICP rotation/translation), the simulation match, per-stage removal counts, and every parameter used are stamped into the `pointcloud/{op}` attributes.

## Further reading

- [Preprocessing overview](index.md): quickstart, output rules, reproducibility model
- [Custom processing](custom.md): replace this stage with your own algorithm
- [HDF5 structure](../hdf5-structure.md): the raw `pointcloud` groups
