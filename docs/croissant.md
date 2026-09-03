# Croissant Manifest

RDDAC ships a [Croissant 1.1](https://doi.org/10.1145/3650203.3663326) manifest (`metadata.json`) describing the dataset in a machine readable form. The same file is read by [Kaggle](https://www.kaggle.com/), [Hugging Face](https://huggingface.co/), and Google Dataset Search, so any tool that already speaks Croissant can consume RDDAC without RDDAC specific code.

## Why Croissant?

The Croissant manifest is the single source of truth for what is in RDDAC and where to find it.

**One schema across tooling.** Field names, shapes, units and descriptions live in the manifest only. The `rddac` Python package reads them from there; Kaggle, Hugging Face, Google Dataset Search and any other Croissant aware tool read the same file. Renaming a field, adding a new use case RecordSet, or updating a description happens once and propagates everywhere.

**No extraction step.** `mlcroissant` opens the zip archives via the [zipfile](https://docs.python.org/3/library/zipfile.html) standard library and streams individual HDF5 members on demand. With the small bundle that is one zip; with the full release it is the two geometry zips plus the sample zip totalling {{ total_size() }}. The user never has to expand the dataset on disk; `rddac download` keeps the zips as they were uploaded.

**The same schema drives the package.** `RDDACDataset` and `rddac.streaming.iter_view` use the manifest only to discover HDF5 paths and field selections; the per record read is a zip member fetch into a `BytesIO` and an `h5py.File` over it. Nothing about the data layout is hardcoded in the Python package.

## Contents

- **Distribution**: every file on DaRUS plus a FileSet for the HDF5 contents inside the zips.
- **`process-parameters`**: rows of `process_parameters.csv` with descriptions and types.
- **`field-map`**: every HDF5 dataset (path, shape, units) declared once, {{ field_map_summary() }}.
- **Use case RecordSets**: {{ use_case_record_sets() }}.

!!! warning "The `thickness` view reads the oil group"
    The published `thickness` RecordSet bundles `sheet_thickness_data` **and** `oil_thickness_data`. The {{ missing_oil() }} experiments with `has_oil=False` have no `oil_thickness/` group, so streaming them through this view raises a `KeyError`. Filter them out with `where=lambda row: row["has_oil"]` (accepted by both `rddac.streaming.iter_view` and `RDDACDataset`).

## With the `rddac` package

```python
import rddac
from rddac import croissant

ds = rddac.load()                            # default URL + data_dir="./data"
ds = rddac.load(data_dir="/mnt/big-disk")    # files were downloaded elsewhere
ds = rddac.load(source="./metadata.json")    # local manifest copy

print(croissant.dataset_name(ds))
for col, desc in croissant.process_parameters_descriptions(ds).items():
    print(col, ":", desc)
```

`croissant.METADATA_URL` is the DaRUS download URL for `metadata.json`.

## Without the `rddac` package

```python
import mlcroissant as mlc

ds = mlc.Dataset(jsonld="{{ metadata_url() }}")
for record in ds.records("force-curve"):
    ...
```
