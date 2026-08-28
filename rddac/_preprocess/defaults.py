"""Processing constants — the single place numbers live (no config files).

Values were validated in the internal pipeline on all 9000 experiments; see
the modality module docstrings for what each step does.
"""

from __future__ import annotations

# ── force ─────────────────────────────────────────────────────────────────────
#: Raw column layout of ``force/data`` (mirrors the h5 ``columns`` attr and
#: ``rddac.visualization.FORCE_COLUMNS``). The processed table keeps ALL of
#: them so the published field mapping stays valid.
FORCE_COLUMNS = (
    "time",
    "load_cell_1",
    "load_cell_2",
    "load_cell_3",
    "load_cell_4",
    "punch_temp",
    "punch_pos",
    "total_force",
)
#: Signals that get the rest-offset (blankholder preload baseline) removed.
FORCE_SIGNALS = ("load_cell_1", "load_cell_2", "load_cell_3", "load_cell_4", "total_force")
#: Forming window in seconds; samples outside are press startup/shutdown noise.
FORCE_TIME_WINDOW = (0.25, 2.25)
#: Rows the window yields at the 300 Hz recording rate.
FORCE_EXPECTED_ROWS = 600
#: Decimal rounding: time in s, forces in kN, temperature in degC.
FORCE_TIME_DECIMALS = 4
FORCE_KN_DECIMALS = 2
FORCE_TEMP_DECIMALS = 1
#: Punch position quantization in mm (press position encoder resolution).
FORCE_POSITION_PRECISION_MM = 0.5

# ── sheet thickness ───────────────────────────────────────────────────────────
#: Keep the last N traverse positions (the stable sensor region).
SHEET_LAST_N = 200
#: Quantization: thickness in um, position in mm.
SHEET_THICKNESS_PRECISION_UM = 0.01
SHEET_POSITION_PRECISION_MM = 0.01

# ── oil thickness ─────────────────────────────────────────────────────────────
#: Positions at/after this sensor coordinate (mm) are edge artifacts.
OIL_MAX_SENSOR_POSITION = 200
#: Fixed output length: one row per integer position 0..199 mm.
OIL_OUTPUT_LENGTH = 200
#: Hampel filter: half-window in mm and threshold in robust sigmas.
OIL_HAMPEL_WINDOW = 5
OIL_HAMPEL_K = 3.0
#: The sensor logs 2 decimals; one LSB of wiggle is never an outlier.
OIL_VALUE_QUANTIZATION = 0.01  # g/m^2

# ── pointcloud ────────────────────────────────────────────────────────────────
#: Luminescence connected-component filter: minimum patch size in pixels.
PC_LUMI_MIN_PATCH_SIZE = 20040
#: Surface-angle seed cutoff (degrees from vertical), per part geometry.
PC_MAX_WALL_ANGLE_CONCAVE_DEG = 70.0
PC_MAX_WALL_ANGLE_CONVEX_DEG = 80.0
#: Radial monotonicity: allowed z increase (mm) moving outward.
PC_Z_TOLERANCE_MM = 1.0
#: Minimum 3D connected-component size (seed stage AND final sweep).
PC_MIN_COMPONENT_SIZE = 50
#: kNN sizes: angle plane fits, monotonicity neighbours, closing graph.
PC_K_ANGLE = 15
PC_K_MONO = 20
PC_K_CLOSING = 8
PC_MAX_CLOSING_ITER = 15
#: ICP iterations and source subsample size (seeded rng -> reproducible).
PC_ICP_MAX_ITERATIONS = 50
PC_ICP_SAMPLE_SIZE = 50000
#: Cup anchor for the second ICP pass: points higher than the simulation's flange
#: level plus this margin (mm) count as cup (bottom + walls); the flange is excluded.
PC_ICP_ANCHOR_HEIGHT_MM = 3.0
#: RF fin classifier decision threshold on predict_proba (the P/R knob).
PC_RF_THRESHOLD = 0.5
#: RF hyperparameters (part of the model-cache fingerprint).
PC_RF_N_ESTIMATORS = 150
PC_RF_MAX_DEPTH = 18
#: Keep the prepared training grids after a successful retrain.
PC_KEEP_PREPARED = False

# Fixed (not configurable — they define the deterministic reference recipe):
#: Training rng seed, inlier:outlier balance ratio, training row cap.
PC_SEED = 42
PC_BALANCE_RATIO = 10.0
PC_MAX_TRAIN_ROWS = 1_200_000
#: Nominal pixel spacings (mm) for the consensus-gradient feature; must match
#: the spacings the shipped labels were trained with.
PC_DX_MM = 0.0769
PC_DY_MM = 0.1589
