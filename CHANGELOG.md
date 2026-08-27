# Changelog

All notable changes to the `rddac` package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the
`bumpver` tags. The dataset itself is versioned on DaRUS (doi:10.18419/DARUS-5589)
independently of the package.

## [Unreleased]

### Added
- `rddac preprocess`: reference preprocessing of the raw dataset into an ML-ready
  processed layer (`<data-dir>/processed`, raw files untouched): forming-window
  force curves (`force`), cleaned sheet-thickness and oil-film profiles (`sheet`,
  `oil`), and calibrated, simulation-aligned point clouds with a random-forest
  fin classifier (`pointcloud`, needs the `[preprocessing]` extra and the DDACS
  simulations). Parameters are adjustable via TOML (`--dump-config`/`--config`)
  and stamped into the output attributes.
- Fin labels (CC BY 4.0), scanner calibration and the simulation parameter table
  ship as package data; the fin classifier is retrained from the labels on first
  use and cached under `<out>/models/pointcloud_fin_rf/`.
- Processing figures per modality (`rddac._preprocess.visualize`), used in the
  documentation and in the new tutorial and notebook `07_preprocessing`.
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

[Unreleased]: https://github.com/BaumSebastian/RDDAC/compare/1.0.1...HEAD
[1.0.1]: https://github.com/BaumSebastian/RDDAC/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/BaumSebastian/RDDAC/releases/tag/1.0.0
