# Point Clouds

The `pointcloud` stage turns the raw laser-scan grids (`pointcloud/{op10,op20}/z` + `luminescence`) into cleaned, simulation-aligned point clouds: `z` becomes `(N, 3) float32` (Open3D-ready) and `luminescence` a `(2000, 3200) uint8` grayscale image (Pillow-ready). It is the only stage that needs the optional dependencies (`pip install 'rddac[preprocessing]'`) and the DDACS simulations.

!!! warning "Needs the DDACS simulations"
    Every scan is aligned to its best-matching DDACS simulation, so the stage requires the simulations fetched by `rddac download` (without `--no-sim`). Without them the command exits with an actionable error before touching any file. If you only miss the simulations, fetch just those:

    ```bash
    ddacs download --files rddac.zip --out ./data/simulation
    ```

    (`./data` = your `--data-dir`; the stage looks for `simulation/rddac.zip` inside it.)

| Property | Raw | Processed |
| --- | --- | --- |
| `z` | flattened `(6400000,)` grid, uncalibrated sensor units | `(N, 3) float32`, mm, cleaned + ICP-aligned |
| `luminescence` | flattened `(6400000,)` grid | `(2000, 3200) uint8`, 0 = background, valid pixels 1–255 |
| Frame | scanner table frame | matched DDACS simulation frame |

## Why this stage exists

Raw scans carry measurement artifacts, most prominently **fins**: locally smooth, wing-like surfaces at draw-in and cut edges that are not part of the physical part. They survive naive filters precisely because they are locally smooth; detecting them reliably needs cross-sample context. The scans are also uncalibrated and each sits in its own scanner frame, so cross-experiment or sim-to-real comparisons need calibration and registration first.

## Processing steps

1. **Validity mask**: connected-component filtering of the luminescence grid plus `z > 0`; the one step shared with the [luminescence image](#the-luminescence-image-an-independent-output) below.
2. **Calibration**: all three scales from the packaged `calibration.json`: x from the sensor specification, z from the calibration-block measurement, y the constant line spacing of the measurement configuration (see [Calibration constants](#calibration-constants)).
3. **Geometric outlier seeds**: three complementary detectors, then morphological closing on a kNN graph:
    - *surface angle*: local SVD plane fits; a point is seeded when its normal deviates more than the cutoff from vertical,

        $$ \theta = \arccos\left(|n_z|\right) > \theta_{\max}, \qquad \theta_{\max} = 70^\circ \text{ (concave)}, \; 80^\circ \text{ (convex)} $$

        The cutoffs sit ≥ 2° above the process limits; propagating the scanner's 5 µm z-repeatability through the gradient gives an angle uncertainty well below that margin (> 4σ).

    - *radial monotonicity*: on a formed cup, z must not increase moving outward from the part center;
    - *small 3D components*: floating clusters below the minimum size.
4. **Simulation matching + ICP**: the matching simulation is selected by geometry and blankholder force (hard) plus sheet thickness and oil-derived friction coefficient (soft, nearest); the scan is rigidly aligned to it in two seeded, reproducible ICP passes: a first pass on all inlier points brings the scan into the simulation frame; a second pass **anchored on the cup** (bottom and walls, every point more than `icp_anchor_height_mm` above the simulation's flange level) fixes the pose. The flange, the region most affected by springback and draw-in, therefore does not bias the alignment, and the remaining wall and radius deviations belong to the part. The z offset is fixed at the cup centre after each pass; the stamped `icp_rotation`/`icp_translation` is the complete transform from the scanner frame.
5. **RF fin classifier**: a random forest over local surface features plus three cross-sample features in a registered common frame: a position prior $P(\text{outlier}\mid\text{position})$, the slope-normalized deviation from the consensus surface, and the registered coordinates. Removal uses `predict_proba` against a configurable threshold, the precision/recall knob.
6. **Final sweep**: small 3D components orphaned by the removal are dropped.

<img src="../../images/preprocessing/pointcloud_processing_concave_op10.png" width="900">
<img src="../../images/preprocessing/pointcloud_processing_concave_op20.png" width="900">
<img src="../../images/preprocessing/pointcloud_processing_convex_op10.png" width="900">
<img src="../../images/preprocessing/pointcloud_processing_convex_op20.png" width="900">

*Experiments 0 (concave) and 4500 (convex), both part of the small bundle and processed with the simulations present, in top view, OP10 (deep drawing) and OP20 (cutting) each: (a) the raw calibrated scan after the validity mask, (b) the processed, ICP-aligned point cloud (gaps are the removed fins and outliers), (c) the matched DDACS simulation in the same frame, and (d) the processed points coloured by their nearest-neighbour distance to the simulation, the same `kd` sim-distance feature the fin classifier uses.*

??? example "These figures were created with"

    ```bash
    python -m rddac._preprocess.visualize pointcloud \
        --data-dir /path/to/rddac --id 0 --op op10 --out docs/images/preprocessing
    # analogously --id 0 --op op20, and --id 4500 for the convex figures
    ```

    The `plot_pointcloud_processing` function takes the raw arrays and derives the
    processed view through the stage's own `process` function, so the figure is by
    construction what `rddac preprocess pointcloud` produces (see the
    [preprocessing tutorial](../tutorials/preprocessing.md)).


## The luminescence image: an independent output

Although the same `rddac preprocess pointcloud` command produces both outputs, the
luminescence path is its **own, independent processing**: the raw luminescence grid is
segmented into connected foreground patches, filtered by patch size into the validity
mask (the only artifact shared with the point-cloud path), and packed into a
`(2000, 3200) uint8` grayscale image: 0 reserved for background, valid pixels
normalized to 1–255, ready for Pillow / image models. None of the later point-cloud
steps touch it: outlier seeds, the fin classifier and the ICP alignment act only on
the 3D points, so the luminescence image keeps the full scanner grid.

<img src="../../images/preprocessing/luminescence_processing_concave_op10.png" width="900">

*Luminescence of one OP10 scan: raw grid (a), connected foreground patches (b), the validity mask after the patch-size filter (c), and the packed `uint8` image written to the processed file (d).*

## Calibration constants

The raw scans are in scanner units. The stage converts them with two constants that ship as package data (`rddac/_preprocess/calibration.json`, read via `rddac._preprocess.pointcloud.geometry.load_calibration()`); anyone replacing the stage needs the same two numbers:

| Constant | Value | Origin |
| --- | --- | --- |
| `x_mm_per_pixel` | 0.0769 mm (1/13) | scanner specification, median width measurement across all rows of the calibration block |
| `y_mm_per_pixel` | 0.1581 mm | line spacing of the measurement configuration (3200 x 2000), determined directly from the measurement scans: mean of the per-scan estimates, corrected by an independent simulation scale fit |
| `z_mm_per_unit` | 0.007755 mm | calibration block of 1.0 mm height: mean of the four corner height differences (RANSAC-fitted surfaces) |

All three scales are stamped into the processed `pointcloud/{op}` attributes. Up to package version 1.1.0 the y scale was *derived per scan* from a square-part assumption; the spread this produced (0.1556 to 0.1614 mm/px) was the anisotropic draw-in of the formed blank leaking into the calibration, not the scanner -- one scanner with one configuration has one line spacing.

<img src="../../images/preprocessing/pointcloud_calibration_block.png" width="700">

*The calibration scan: the magnetic gripper with the 30 x 9 x 1 mm calibration block, and the block with its measured edges. Note that this scan was recorded at the same scan speed but with a different number of points (3200 x 1600 instead of the 3200 x 2000 of the measurement scans), so its y spacing (30 mm / 151 px = 0.199 mm/px) belongs to that scan only and is **not** used for the measurements; `y_mm_per_pixel` in the table is determined directly at the measurement configuration.*

<img src="../../images/preprocessing/pointcloud_calibration_result.png" width="700">

*The calibration applied to experiment 0 (OP10): (a) the raw sensor grid, (b) the calibrated cloud in mm with everything below the flange level removed, (c) the fully processed cloud -- **after outlier and fin removal** -- on its matched DDACS simulation (182208), and (d) the same cloud coloured by the distance to it. The gaps in (c) and (d) are the removed fins and outliers.*

Beyond validating the calibration, these views carry process information directly: in (c) the scanned flange contour deviates from the simulated one differently on each side -- the material flow into the die is not symmetric -- and (d) localizes where the real part springs back from the simulated shape (walls and radii) while the cup bottom stays within a few tenths of a millimetre.

## The classifier is retrained on your machine

Trained models are derived artifacts and are **not distributed**. The bundled labels are annotations of the dataset and, unlike the code, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribute the RDDAC dataset when you reuse them. On first use the stage retrains them deterministically (seed 42) from the labels bundled with the package: human-labeled outlier masks for 140 (experiment, operation) tasks, and caches them under `<out>/models/pointcloud_fin_rf/` (the processed output directory, never the raw one) stamped with the scikit-learn version and the label fingerprint. The one-time cost is roughly 30–90 minutes; `--rebuild-models` forces a retrain. Held-out 5-fold cross-validation of the resulting cleaner:

| Group | Precision | Recall |
| --- | --- | --- |
| concave OP10 | 0.87 | 0.97 |
| concave OP20 | 0.86 | 0.94 |
| convex OP10 | 0.94 | 0.99 |
| convex OP20 | 0.96 | 0.96 |

## Runtime

Expect roughly **1.5 to 2 minutes per experiment and worker** (single core): every stage works on the ~3 million valid points of a scan with kNN graphs (surface angle, radial monotonicity, component filter, morphological closing) plus two ICP passes and the random-forest prediction, and none of these dominates. Use `--workers N` up to the number of free cores (about 2 GB of memory per worker), or split the ids across machines that share the output directory (`--ids 0-4499` / `--ids 4500-8999`). Interrupted runs resume: re-running without `--overwrite` skips finished files.

!!! note "Not implemented: grid-based outlier detection"
    The scan points lie on a regular pixel grid, so the kNN-based seed stages could be expressed as image operations on the `z` grid (gradients for the surface angle, `ndimage` labelling and closing for components), which would cut the runtime by an estimated factor of 2 to 3. This is deliberately not implemented yet: it changes the validated outlier detection and would need re-validation against the bundled labels and a retrain of the fin classifier. Contributions welcome.

## Parameters

| TOML key (`[pointcloud]`) | Default | Meaning |
| --- | --- | --- |
| `lumi_min_patch_size` | 20040 | luminescence patch filter (px) |
| `max_wall_angle_concave_deg` / `..._convex_deg` | 70 / 80 | surface-angle cutoffs |
| `z_tolerance_mm` | 1.0 | radial monotonicity tolerance |
| `min_component_size` | 50 | 3D component filter (seed stage + final sweep) |
| `k_angle` / `k_mono` / `k_closing` | 15 / 20 / 8 | kNN sizes |
| `max_closing_iter` | 15 | upper bound on morphological-closing iterations |
| `icp_max_iterations` / `icp_sample_size` | 50 / 50000 | ICP effort (per pass) |
| `icp_anchor_height_mm` | 3.0 | second ICP pass uses only points this far above the simulation's flange level (the cup) |
| `rf_threshold` | 0.5 | fin-classifier decision threshold (higher = more conservative removal) |
| `rf_n_estimators` / `rf_max_depth` | 150 / 18 | RF hyperparameters (changing them retrains) |
| `keep_prepared` | false | keep the prepared training grids after a successful retrain (debugging aid) |

Alignment (ICP rotation/translation), the simulation match, per-stage removal counts, and every parameter used are stamped into the `pointcloud/{op}` attributes.

## Further reading

- [Preprocessing overview](index.md): quickstart, output rules, reproducibility model
- [Custom processing](custom.md): replace this stage with your own algorithm
- [HDF5 structure](../hdf5-structure.md): the raw `pointcloud` groups
