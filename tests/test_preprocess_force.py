"""Tests for the force preprocessing (rddac._preprocess.force)."""

import numpy as np
import pytest

from rddac._preprocess import force


def _table():
    """300 Hz, 0..3.8 s, blankholder baseline + Gaussian forming bump, 8 columns."""
    t = np.arange(0, 3.8, 1 / 300)
    n = len(t)
    lc = 23.0 + 100.0 * np.exp(-((t - 1.2) ** 2))
    temp = np.full(n, 22.34)
    ppos = np.linspace(448.66, 289.56, n)
    return np.column_stack([t, lc, lc, lc, lc, temp, ppos, 4 * lc]).astype(np.float32)


class TestForce:
    """Window trimming, offset removal, quantization, column-layout handling."""

    def test_window_offset_and_all_columns_kept(self):
        out, attrs = force.process(_table())
        assert out.shape == (600, 8) and out.dtype == np.float32
        assert attrs["n_rows"] == 600
        assert out[0, 0] == 0.0, "time re-zeroed to window start"
        assert out[:, 1].min() == pytest.approx(0.0, abs=1e-6), "load-cell offset removed"
        pp = out[:, 6].astype(float)
        assert np.allclose(pp * 2, np.round(pp * 2), atol=1e-3), "punch_pos on 0.5 mm grid"
        assert np.all(out[:, 5] == np.float32(22.3)), "punch_temp rounded to 0.1"

    def test_respects_column_attr_order(self):
        table = _table()[:, ::-1]  # reversed layout
        out, _ = force.process(table, tuple(reversed(force.COLUMNS)))
        assert out.shape[1] == 8
        assert out[0, 7] == 0.0, "time column stays where the layout says"

    def test_missing_time_column_raises(self):
        table = _table()[:, 1:]
        with pytest.raises(ValueError, match="time"):
            force.process(table, force.COLUMNS[1:])
