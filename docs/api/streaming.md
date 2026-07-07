# Streaming

`rddac.streaming` is the offline-iteration namespace with four torch-free entry points:

- `iter_view` walks a Croissant view record by record. Shares a unified index that recognises loose `.h5` files (`rddac download --extract --remove-zip`) and zipped `*.zip` archives interchangeably.
- `export_to_numpy` materialises a view as flat `.npy` memmap shards, with optional per-field and whole-record transforms. Requires every record to share the same shape per field.
- `export_to_numpy_per_sim` writes one `.npz` per experiment instead — the escape hatch for views whose raw fields have per-experiment sample counts.
- `load_export` opens the memmap shards back as a `len + getitem + iter` protocol object that plugs into `torch.utils.data.DataLoader`, `tf.data.Dataset.from_generator`, JAX, or plain Python without any adapter.

## `rddac.streaming.iter_view`

::: rddac.streaming.iter_view
    options:
      show_signature: true
      show_signature_annotations: true

## `rddac.streaming.export_to_numpy`

::: rddac.streaming.export_to_numpy
    options:
      show_signature: true
      show_signature_annotations: true

## `rddac.streaming.export_to_numpy_per_sim`

::: rddac.streaming.export_to_numpy_per_sim
    options:
      show_signature: true
      show_signature_annotations: true

## `rddac.streaming.load_export`

::: rddac.streaming.load_export
    options:
      show_signature: true
      show_signature_annotations: true
