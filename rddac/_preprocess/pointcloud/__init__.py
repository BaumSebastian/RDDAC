"""Pointcloud preprocessing: scan grids to cleaned, aligned points.

Outputs per op: ``z (N, 3) float32`` + ``luminescence (2000, 3200) uint8``.

Pipeline per operation (OP10 deep drawing, OP20 cutting):

1. Luminescence mask (connected-component filtering) + ``z > 0`` validity.
2. Calibration to mm (x from the sensor spec, y from the square-part
   assumption, z from the calibration block; ``calibration.json`` ships as
   package data).
3. Geometric outlier seeds — local surface angle (SVD plane fits), radial
   monotonicity, small 3D components — plus morphological closing on a kNN
   graph.
4. Simulation matching (geometry, blankholder force, sheet/oil means of the
   same run) + ICP alignment to the matched DDACS simulation. Requires the
   simulations from ``rddac download`` (without ``--no-sim``).
5. RF fin cleaner: registered position-prior + consensus-deviation features;
   retrained deterministically from the bundled labels on first use and
   cached under ``<out_dir>/models`` (models are not shipped).
6. Final small-component sweep on the cleaned cloud.

The runner drives the stage through :mod:`.stage` (``PROCESSES_FILE`` contract).
"""
