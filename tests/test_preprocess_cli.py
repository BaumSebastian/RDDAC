"""Tests for the ``rddac preprocess`` CLI plumbing (parser + output-dir guard)."""

import argparse

import pytest
from rich.console import Console

from rddac._preprocess import cli as pcli
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


class TestCmdPreprocess:
    """Dispatch, prompt and error paths of ``cmd_preprocess`` (the runner is stubbed)."""

    @pytest.fixture(autouse=True)
    def _stub_run(self, monkeypatch, tmp_path):
        self.calls = []
        monkeypatch.setattr(pcli, "run", lambda names, **kw: self.calls.append((names, kw)) or {})
        self.data = tmp_path / "data"
        self.data.mkdir()
        (self.data / "0000.h5").write_bytes(b"")
        monkeypatch.setattr(pcli.console, "quiet", False)

    def test_dump_config_prints_toml_and_exits(self, capsys):
        pcli.cmd_preprocess(_parse(["--dump-config"]))
        assert "[oil]" in capsys.readouterr().out and not self.calls

    def test_bad_config_and_unknown_modality_exit_2(self, tmp_path):
        bad = tmp_path / "bad.toml"
        bad.write_text("[nope]\nx = 1\n")
        with pytest.raises(SystemExit) as exc:
            pcli.cmd_preprocess(_parse(["--data-dir", str(self.data), "--config", str(bad), "-y"]))
        assert exc.value.code == 2
        with pytest.raises(SystemExit) as exc:
            pcli.cmd_preprocess(_parse(["lidar", "--data-dir", str(self.data), "-y"]))
        assert exc.value.code == 2 and not self.calls

    def test_prompt_declined_aborts(self, monkeypatch):
        monkeypatch.setattr(pcli.Confirm, "ask", lambda *a, **k: False)
        pcli.cmd_preprocess(_parse(["oil", "--data-dir", str(self.data)]))
        assert not self.calls

    def test_yes_and_quiet_skip_prompt(self, monkeypatch):
        monkeypatch.setattr(pcli.Confirm, "ask", lambda *a, **k: pytest.fail("prompt must be skipped"))
        pcli.cmd_preprocess(_parse(["oil", "--data-dir", str(self.data), "-y"]))
        pcli.cmd_preprocess(_parse(["oil", "force", "--data-dir", str(self.data), "-q"]))
        assert [c[0] for c in self.calls] == [["oil"], ["oil", "force"]]
        assert self.calls[1][1]["quiet"] is True and pcli.console.quiet is True
        pcli.console.quiet = False

    def test_registered_in_main_cli(self):
        import rddac.cli as main_cli

        parser = main_cli.argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        main_cli.add_preprocess_parser(sub)
        assert parser.parse_args(["preprocess", "sheet", "-q"]).command == "preprocess"
