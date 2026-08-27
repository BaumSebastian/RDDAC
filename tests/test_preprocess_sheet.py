"""Tests for the sheet-thickness preprocessing (rddac._preprocess.sheet)."""

import numpy as np

from rddac._preprocess import sheet


class TestSheet:
    """Error-code masking, position normalization, fixed-length output."""

    def test_masks_negatives_and_normalizes_position(self):
        pos = np.arange(10, 10 + 0.5 * 208, 0.5)
        thick = np.full(208, 995.5)
        thick[-5] = -99999.0  # sensor error code inside the kept window
        out, attrs = sheet.process(np.column_stack([pos, thick]).astype(np.float32))
        assert out.shape == (200, 2) and out.dtype == np.float32
        assert out[0, 0] == 0.0
        assert attrs["n_negative_masked"] == 1
        assert np.isnan(out[:, 1]).sum() == 1

    def test_all_negative_trace_masks_everything(self):
        pos = np.arange(0, 100, 0.5)
        thick = np.full(200, -99999.0)
        out, attrs = sheet.process(np.column_stack([pos, thick]).astype(np.float32))
        assert attrs["n_negative_masked"] == 200
        assert np.isnan(out[:, 1]).all()
