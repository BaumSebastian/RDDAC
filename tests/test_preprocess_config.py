"""Tests for the preprocessing parameter system (rddac._preprocess.config)."""

import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from rddac._preprocess import config


class TestConfig:
    """TOML defaults dump, override merging, and typo rejection."""

    def test_dump_is_valid_toml_and_roundtrips(self):
        assert tomllib.loads(config.dump()) == config.DEFAULTS

    def test_load_merges_partial_overrides(self, tmp_path):
        good = tmp_path / "good.toml"
        good.write_text("[oil]\nhampel_k = 5.0\n")
        cfg = config.load(str(good))
        assert cfg["oil"]["hampel_k"] == 5.0
        assert cfg["oil"]["hampel_window"] == config.DEFAULTS["oil"]["hampel_window"]

    def test_load_rejects_unknown_key_and_table(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("[oil]\nhampel_kk = 5.0\n")
        with pytest.raises(ValueError, match="hampel_kk"):
            config.load(str(bad))
        bad_table = tmp_path / "bad_table.toml"
        bad_table.write_text("[oill]\nhampel_k = 5.0\n")
        with pytest.raises(ValueError, match="oill"):
            config.load(str(bad_table))
