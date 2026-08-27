"""Preprocessing of raw RDDAC measurements into an ML-ready processed layout.

Internal package — the supported entry point is the ``rddac preprocess`` CLI;
these modules may change without notice. Anyone replacing a processing step
should read raw data via the public API and match the documented processed
schema instead of importing from here (see the "Custom processing" docs page).

STATUS: ``force``, ``sheet`` and ``oil`` are implemented (ported from the
validated internal pipeline); ``pointcloud`` follows.

Design decisions (2026-08):

- **Raw files are never modified.** Processing reads the raw experiments
  (zips or loose ``.h5``) and writes NEW files to ``<out>/<id>.h5``. The
  processed files keep the raw HDF5 paths, so the published Croissant
  manifest is the single source of truth for both layers: its views stream
  the processed directory via ``data_dir=`` (loose ``.h5`` layout). No
  manifest is generated.
- **Processed schema** per experiment::

      force/data            (600, 8) float32   # all raw columns, window-trimmed
      sheet_thickness/data  (200, 2) float32
      oil_thickness/data    (200, 2) float32   # dropout removal + robust Hampel + interpolation
      pointcloud/{op10,op20}/
        z             (N, 3)       float32     # cleaned, ICP-aligned points (Open3D-ready)
        luminescence  (2000, 3200) uint8       # grayscale [0, 255] (Pillow-ready)
        attrs: calibration, ICP rotation/translation, cleaning statistics

- Per-file cleaning statistics live in h5 attributes plus a summary log
  line; there is no separate metrics file.
- The ``pointcloud`` stage needs the DDACS simulations (``rddac download``
  without ``--no-sim``). Its fin classifier is retrained deterministically
  from the bundled labels on first use and cached under
  ``<data_dir>/models`` — trained models are not shipped.

The modality registry lives in :mod:`.runner` (the modules are imported
there, keeping this package free of import-time side effects).
"""
