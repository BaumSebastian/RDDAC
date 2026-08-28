# CLI Reference

The `rddac` command line interface downloads the dataset from DaRUS and prints summary information. Because RDDAC is the experimental counterpart to DDACS, the full download also fetches the matching DDACS simulations (skip with `--no-sim`).

## `rddac info`

Display dataset information and the list of available versions.

```bash
rddac info
```

## `rddac download`

Download dataset files from DaRUS.

```bash
rddac download [VERSION] [OPTIONS]
```

### Arguments

{{ cli_arguments("download") }}

### Options

{{ cli_options("download") }}

`--small` fetches {{ small_download_size() }}: `sample.zip` (one experiment per category), `process_parameters.csv`, `metadata.json`.

### Default behaviour

Zip files are kept on disk by default and are not extracted. This keeps the dataset readable in place by `mlcroissant`, which references zip members through the Croissant manifest. Pass `--extract` to additionally write the HDF5 files to disk, and `--remove-zip` to delete the zip afterwards.

The `--out` directory defaults to `./data`. The same value is used by `rddac.load(data_dir=...)` and `RDDACDataset(data_dir=...)`, so files written by `rddac download` are picked up by the Python API without additional configuration.

### DDACS simulations

A full download (no `--small`, no `--files`) additionally fetches `rddac.zip` ({{ simulation_download_size() }}) from the [DDACS dataset](https://doi.org/10.18419/DARUS-4801) into `./data/simulation/`. It contains the FEM simulations matching the RDDAC parameter grid, for simulation-to-reality comparison. Pass `--no-sim` to skip this second download; the simulations are also listed and confirmed separately, so answering `n` at the prompt has the same effect.

### Examples

```bash
# Download the small test set ({{ small_download_size() }})
rddac download --small -y

# Download the full dataset: measurements + matching DDACS simulations
rddac download

# Real measurements only
rddac download --no-sim

# Download to a custom directory
rddac download --out /path/to/data

# Download specific files only
rddac download --files {{ small_test_files() }}

# Extract the zip files in place, keep the zip alongside
rddac download --extract

# Extract and remove the zip after a successful extraction
rddac download --extract --remove-zip
```

After `--extract --remove-zip`, the HDF5 files are no longer wrapped in zips and `mlcroissant` cannot resolve the FileSet. See the [Loose HDF5 recipe](tutorials/loose-h5.md) for the appropriate iteration pattern in that case.

## `rddac preprocess`

Process raw measurements into the ML-ready processed layout — see [Preprocessing](preprocessing/index.md) for what each modality does. Raw files are never modified; processed files are written separately.

```bash
rddac preprocess [MODALITY ...] [OPTIONS]
```

### Arguments

{{ cli_arguments("preprocess") }}

`pointcloud` covers both scan grids, `z` and `luminescence`. Without an explicit modality list, `pointcloud` is skipped with a notice when the DDACS simulations (or the `rddac[preprocessing]` extra) are missing; named explicitly, it fails with an actionable error instead.

### Options

{{ cli_options("preprocess") }}

### Default behaviour

Output lands in `<data-dir>/processed` as loose `<id>.h5` files. Re-runs append: modalities that already exist in an output file are skipped unless `--overwrite` is passed. The command refuses an output directory that looks like a raw dataset directory (see [output directory rules](preprocessing/index.md#output-directory-rules)).

### Examples

```bash
# Everything, reference recipe
rddac preprocess

# Only the traverse measurements, 8 workers
rddac preprocess sheet oil --workers 8

# A parameter variant, published next to your code
rddac preprocess --dump-config > my.toml
rddac preprocess oil --config my.toml

# One split, custom locations
rddac preprocess --split train --data-dir /mnt/rddac --out /mnt/rddac-processed
```

## Global Options

{{ cli_options() }}
