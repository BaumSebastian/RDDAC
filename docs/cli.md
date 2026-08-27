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

| Argument | Default | Description |
|----------|---------|-------------|
| `VERSION` | `1.0` | Dataset version to download |

### Options

| Option | Description |
|--------|-------------|
| `--small` | Download the small test set ({{ small_download_size() }}): `sample.zip` (one experiment per category), `process_parameters.csv`, `metadata.json` |
| `--no-sim` | Download only the real measurements; skip the DDACS simulations |
| `--files FILE...` | Download only the listed files |
| `--out PATH` | Output directory (default: `./data`) |
| `--extract` | Extract zip files in place after download |
| `--remove-zip` | Delete the zip file after a successful extraction (requires `--extract`) |
| `-y, --yes` | Skip the confirmation prompt |
| `-q, --quiet` | No output or progress; implies `--yes` |

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

| Argument | Default | Description |
|----------|---------|-------------|
| `MODALITY ...` | all | Subset of `force`, `sheet`, `oil`, `pointcloud`. Without an explicit list, `pointcloud` is skipped with a notice when the DDACS simulations (or the `rddac[preprocessing]` extra) are missing; named explicitly, it fails with an actionable error instead. |

### Options

| Option | Description |
|--------|-------------|
| `--data-dir PATH` | Directory holding the raw dataset — zips or loose `.h5` (default: `./data`) |
| `--out PATH` | Output directory for processed files (default: `<data-dir>/processed`) |
| `--ids IDS` | Experiment selection, e.g. `0-999` or `42,1035` |
| `--split {train,val,test}` | Restrict to one of the predefined splits |
| `--workers N` | Parallel workers (default: 1) |
| `--overwrite` | Recompute modalities that already exist in the output files |
| `--config TOML` | TOML file overriding processing parameters |
| `--dump-config` | Print the default processing parameters as TOML and exit |
| `--rebuild-models` | Force retraining of the cached fin-classifier models (pointcloud stage) |
| `-y, --yes` | Skip the confirmation prompt |
| `-q, --quiet` | No output or progress bars; implies `--yes` |

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

| Option | Description |
|--------|-------------|
| `--token TOKEN` | DaRUS API token (used to download draft versions) |
| `-V, --version` | Show the package version and exit |
