# Custom Processing

`rddac preprocess` is the **reference implementation** of the processed layout — not the only allowed one. The raw dataset is immutable and readable through the public API, so anyone with a better algorithm can produce their own processed layer: the contract is the **schema of the output files**, not a Python interface.

## The contract

Read raw experiments with the public API, run your algorithm, and write files matching the [processed schema](index.md#what-each-modality-does) into your own output directory:

```python
import h5py
import rddac

for exp_id in range(9000):
    with rddac.open_h5(exp_id, data_dir="./data") as raw:      # immutable input
        oil_raw = raw["oil_thickness/data"][:]

    cleaned = my_better_oil(oil_raw)                            # your algorithm -> (200, 2) float32

    with h5py.File(f"./my_processed/{exp_id:04d}.h5", "a") as out:
        group = out.create_group("oil_thickness")
        group.create_dataset("data", data=cleaned)
        group.attrs["columns"] = ["sensor_position", "oil_value"]
        group.attrs["producer"] = "my_better_oil v1"            # honest provenance
```

Files that match the schema are consumable by the same downstream tooling as ours. Because raw data is canonical and pinned by the published Croissant manifest, *anyone* can reproduce your processed layer from your code — replacing an algorithm always means running it yourself; there is nothing server-side to swap.

Two conventions keep replacements honest:

- **Never write into the raw directory** — the published checksums are the dataset's identity.
- **Stamp what you did** into the group attributes (our stages record every parameter used), so a processed file documents itself even when separated from the code.

For parameter-level variants of *our* algorithms you do not need any of this — [`--config`](index.md#adjusting-parameters) covers that reproducibly.

## Internal reference

!!! warning "Internal API — may change without notice"
    The implementation lives in the private `rddac._preprocess` package. It is CLI-only by design; the signatures below are shown for orientation (e.g. for a notebook walking through the pipeline), not as a stable interface.

::: rddac._preprocess.oil.process
    options:
      show_signature: true
      show_signature_annotations: true

::: rddac._preprocess.runner.run
    options:
      show_signature: true
      show_signature_annotations: true
