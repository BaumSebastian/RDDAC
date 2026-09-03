# Changelog

All notable changes to the `rddac` package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the
`bumpver` tags. The dataset itself is versioned on DaRUS (doi:10.18419/DARUS-5589)
independently of the package.

## [Unreleased]

### Changed
- Default dataset version is now **2.0** (DaRUS doi:10.18419/DARUS-5589).
  Dataset 2.0 corrects the assignment of the convex force measurements
  (ids 4500-8999): in 1.0 the `force/data` group and the derived
  `mean_punch_temp` of these experiments were attached to the wrong ids;
  sheet thickness, oil film, point clouds and all labels were and are correct.
  `process_parameters.csv` changes in the `mean_punch_temp` column only.
- The software itself is unchanged and **backwards compatible**: both dataset
  versions share the same layout, every view and the preprocessing work on
  either, and `rddac download 1.0` still fetches the previous version.
- After switching a local copy to dataset 2.0, refresh the force-derived
  processed layer with `rddac preprocess force --overwrite`; the other
  modalities are unaffected.

## [1.1.2] - 2026-09-03

### Changed
- Pointcloud ICP now draws its sample from a density-neutral 0.35 mm voxel
  grid. The scanner grid is twice as dense in x as in y and the fin cleaner
  thins one wall, which let dense regions dominate the fit and overstate the
  deviation of sparse walls (~0.1 mm pose bias, clearly visible on convex
  OP20). Reprocess point clouds with `rddac preprocess pointcloud --overwrite`.
- Pointcloud alignment gained a third step after the cup-anchored ICP: the scan
  is centered in x/y on the deck outline of the simulation. The ICP cost is
  nearly flat in x/y for these level-topped parts, so the pose inside that
  valley was ambiguous and piled the real scan-vs-simulation width difference
  onto one side. The applied shift is stored in the new `deck_shift_x` and
  `deck_shift_y` attributes.

## [1.1.1] - 2026-09-01

### Changed
- Pointcloud preprocessing: the y scale is now the packaged constant
  `y_mm_per_pixel = 0.1581` (one scanner, one configuration, one line spacing)
  instead of a per-scan derivation from a square-part assumption, which had let
  the anisotropic draw-in of the formed blank leak into the calibration
  (spread 0.1556-0.1614 mm/px). Reprocess point clouds with
  `rddac preprocess pointcloud --overwrite --rebuild-models` to apply it.

## [1.1.0] - 2026-08-28

### Added
- `rddac preprocess`: reference preprocessing of the raw dataset into an ML-ready
  processed layer (`<data-dir>/processed`, raw files untouched): forming-window
  force curves (`force`), cleaned sheet-thickness and oil-film profiles (`sheet`,
  `oil`), and calibrated point clouds aligned to the matched DDACS simulation
  with a random-forest fin classifier (`pointcloud`, needs the `[preprocessing]`
  extra and the DDACS simulations). The alignment is a two-pass ICP anchored on
  the cup (bottom and walls), so flange springback and draw-in do not bias the
  pose. Parameters are adjustable via TOML (`--dump-config`/`--config`) and
  stamped into the output attributes; the run summary reports per modality what
  was processed, skipped or failed, in plain words.
- Fin labels (CC BY 4.0), scanner calibration and the simulation parameter table
  ship as package data; the fin classifier is retrained from the labels on first
  use and cached under `<out>/models/pointcloud_fin_rf/`.
- Processing figures per modality (`rddac._preprocess.visualize`), used in the
  documentation and in the new tutorial and notebook `07_preprocessing`; the
  point-cloud figure shows the raw scan, the processed cloud, the matched
  simulation and the distance to it.
- Documentation section *Preprocessing* (overview, one page per modality,
  custom processing) and a CI workflow (pre-commit hooks, tests on 3.10 to 3.12,
  strict docs build).

### Changed
- The processed layer needs no manifest of its own: the Croissant manifest
  downloaded with the dataset describes both layers, and every view streams the
  processed directory via `data_dir=` (pass `source=` to the local manifest).
- Documentation palette switched to violet (`#7F00FF`, softer in dark mode).

## [1.0.1] - 2026-07-13

### Fixed
- Kaggle kernel publishing: notebook slug naming and server-side rejection.

### Added
- Scripts and a workflow that publish releases to PyPI and refresh the Zenodo,
  Hugging Face and Kaggle teasers.
- LaTeX rendering of math in the documentation; links to the related repositories.

## [1.0.0] - 2026-07-07

### Added
- Initial package release: `rddac download`, Croissant access (`load`,
  `add_view`), single-experiment HDF5 access, streaming and numpy export,
  PyTorch `RDDACDataset`, plotting helpers, documentation and six tutorial
  notebooks.

[Unreleased]: https://github.com/BaumSebastian/RDDAC/compare/1.1.2...HEAD
[1.1.2]: https://github.com/BaumSebastian/RDDAC/compare/1.1.1...1.1.2
[1.1.1]: https://github.com/BaumSebastian/RDDAC/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/BaumSebastian/RDDAC/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/BaumSebastian/RDDAC/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/BaumSebastian/RDDAC/releases/tag/1.0.0
