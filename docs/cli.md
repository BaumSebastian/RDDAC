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

## Global Options

| Option | Description |
|--------|-------------|
| `--token TOKEN` | DaRUS API token (used to download draft versions) |
| `-V, --version` | Show the package version and exit |
