"""Tests for the oil-film preprocessing (rddac._preprocess.oil)."""

import numpy as np
import pytest

from rddac._preprocess import oil


def _trace():
    """Plausible raw trace: 0..209 mm, smooth ~1.2 g/m^2, 2-decimal quantized."""
    pos = np.arange(0, 210, dtype=float)
    val = np.round(1.2 + 0.2 * np.sin(pos / 30), 2)
    return pos, val


class TestOil:
    """Dropout removal, NaN-robust Hampel filtering, and fixed-grid output."""

    def test_removes_nan_and_spike_next_to_it(self):
        pos, val = _trace()
        val[100] = np.nan  # sensor dropout
        val[103] = 3.5  # spike inside the dropout's Hampel window
        out, attrs = oil.process(np.column_stack([pos, val]))
        assert out.shape == (200, 2) and out.dtype == np.float32
        assert not np.isnan(out[:, 1]).any()
        assert out[:, 1].max() < 2.0, "spike must not survive into the output"
        assert attrs["n_nan_removed"] == 1
        assert attrs["n_hampel_outliers"] >= 1

    def test_flat_trace_spike_caught_despite_zero_mad(self):
        pos = np.arange(0, 210, dtype=float)
        val = np.full_like(pos, 1.0)
        val[50] = 3.0
        out, attrs = oil.process(np.column_stack([pos, val]))
        assert attrs["n_hampel_outliers"] == 1
        assert out[:, 1].max() == pytest.approx(1.0)

    def test_plateau_step_is_not_flagged(self):
        pos = np.arange(0, 210, dtype=float)
        val = np.where(pos < 100, 1.10, 1.20)
        _, attrs = oil.process(np.column_stack([pos, val]))
        assert attrs["n_hampel_outliers"] == 0

    def test_attrs_are_complete_and_self_describing(self):
        pos, val = _trace()
        _, attrs = oil.process(np.column_stack([pos, val]))
        for key in (
            "n_raw_in_range",
            "n_nan_removed",
            "n_hampel_outliers",
            "n_positions_interpolated",
            "max_sensor_position",
            "output_length",
            "hampel_window",
            "hampel_k",
            "value_quantization",
        ):
            assert key in attrs, key
