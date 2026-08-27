# Fin labels

Human-labeled outlier masks for the pointcloud fin cleaner. One file per
labeled (experiment, operation) task, named `<id>_<op>.npz` (e.g.
`0344_op10.npz`), each holding:

- `outlier_mask` — `(2000, 3200) bool` on the raw scan pixel grid; `True`
  marks measurement artifacts ("fins" at draw-in and cut edges) to remove.

The masks index the RAW grid, so they stay valid for any preprocessing
variant. They ship as package data (~1 MB total) because the fin classifier
is retrained from them on first use — trained models are derived artifacts
and are deliberately NOT distributed.

Labeled coverage: concave_op10/op20 (27 each), convex_op10/op20 (43 each),
all in one consistent labeling style (held-out 5-fold CV of the resulting
cleaner: precision 0.86–0.96, recall 0.94–0.99 per group).
