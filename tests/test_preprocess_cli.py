"""Tests for the ``rddac preprocess`` CLI plumbing (parser + output-dir guard)."""

import argparse

import pytest
from rich.console import Console

from rddac._preprocess.cli import _guard_out_dir, add_preprocess_parser


def _parse(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_preprocess_parser(sub)
    return parser.parse_args(["preprocess", *argv])


class TestParser:
    def test_defaults(self):
        args = _parse([])
        assert args.modalities == [] and args.workers == 1 and not args.overwrite

    def test_modalities_and_flags(self):
        args = _parse(["oil", "force", "--ids", "0-9", "--workers", "4", "--rebuild-models"])
        assert args.modalities == ["oil", "force"]
        assert args.ids == "0-9" and args.workers == 4 and args.rebuild_models


class TestOutputDirGuard:
    """The guard must refuse locations that could shadow or mutate raw data."""

    def test_refuses_raw_dir_itself(self, tmp_path):
        with pytest.raises(SystemExit):
            _guard_out_dir(str(tmp_path), str(tmp_path), Console(quiet=True))

    def test_refuses_raw_looking_dir(self, tmp_path):
        out = tmp_path / "other_raw"
        out.mkdir()
        (out / "metadata.json").touch()
        with pytest.raises(SystemExit):
            _guard_out_dir(str(tmp_path / "raw"), str(out), Console(quiet=True))

    def test_accepts_fresh_subdirectory(self, tmp_path):
        _guard_out_dir(str(tmp_path), str(tmp_path / "processed"), Console(quiet=True))
