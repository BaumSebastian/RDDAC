# Tutorials

These tutorials walk through the typical workflows on top of the `rddac` Python surface, from a first download to a PyTorch training loop. Each page is short and code centric: read top to bottom or jump to the section that matches the task at hand.

1. [Getting Started](getting-started.md): Install, download the small bundle, load the manifest, open one experiment, render the first plot.
2. [Build your own view](views.md): Use `rddac.add_view` to compose a custom Croissant RecordSet from the field map.
3. [PyTorch training](pytorch.md): Stream a view through `RDDACDataset`, batching with `DataLoader`, multi worker sharding, DDP, and a reproducibility note.
4. [Visualization](visualization.md): Scan images, point clouds, force curves, and sensor traverses on top of `rddac.open_h5`.
5. [Loose HDF5 recipe](loose-h5.md): Iterate over loose `.h5` files after `rddac download --extract --remove-zip`.
6. [Streaming and numpy export](streaming.md): Iterate any view with `rddac.streaming.iter_view` (no PyTorch, no mlcroissant FileSet walk), and materialise it as flat `.npy` shards with `streaming.export_to_numpy` for fast training reads.

If you already know the DDACS tutorials: the flow here is identical by design. The `rddac` public surface mirrors `ddacs`, so everything you learned there ports by swapping the import.

## Prerequisites

```bash
pip install rddac
rddac download --small
```
