"""Tests for the RDDAC CLI.

Unit tests exercise argument parsing and the pure helpers with mocked
network access — no real DaRUS calls. The end-to-end `download --small`
integration tier at the bottom only runs when ``RDDAC_RUN_DOWNLOAD_TESTS``
is set (see the ``small_data_dir`` fixture).
"""

import argparse
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from rddac.cli import (
    __version__,
    _dataset_title,
    _file_info,
    _matches,
    cmd_download,
    cmd_info,
    main,
)
from rddac.cli import DEFAULT_VERSION, SMALL_TEST_FILES


def _download_args(**overrides) -> argparse.Namespace:
    """A cmd_download Namespace with sane defaults (mirrors the argparse dests)."""
    defaults = dict(
        version=DEFAULT_VERSION,
        token=None,
        files=None,
        small=False,
        no_sim=True,  # unit tests skip the DDACS-simulation leg by default
        out="./data",
        yes=True,
        quiet=False,
        extract=False,
        remove_zip=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _version_payload(files: list[dict]) -> dict:
    return {
        "versionNumber": 1,
        "versionMinorNumber": 0,
        "versionState": "RELEASED",
        "lastUpdateTime": "2026-01-01T00:00:00Z",
        "license": {"name": "CC BY 4.0"},
        "files": files,
    }


class TestVersion:
    """Test CLI version."""

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])


class TestFileInfo:
    """Tests for the _file_info helper function."""

    def test_basic(self):
        file_meta = {"dataFile": {"filename": "test.csv", "filesize": 1024}}
        name, size = _file_info(file_meta)
        assert name == "test.csv"
        assert size == 1024

    def test_prefers_original_name_and_size(self):
        file_meta = {
            "dataFile": {
                "filename": "test.tab",
                "filesize": 500,
                "originalFileName": "test.csv",
                "originalFileSize": 1024,
            }
        }
        name, size = _file_info(file_meta)
        assert name == "test.csv"
        assert size == 1024

    def test_original_without_size_falls_back(self):
        file_meta = {
            "dataFile": {
                "filename": "test.tab",
                "filesize": 500,
                "originalFileName": "test.csv",
            }
        }
        name, size = _file_info(file_meta)
        assert name == "test.csv"
        assert size == 500  # Falls back to filesize


class TestMatches:
    """Tests for the _matches file-selection helper."""

    def test_matches_plain_filename(self):
        meta = {"dataFile": {"filename": "sample.zip", "filesize": 1}}
        assert _matches(meta, ["sample.zip"])
        assert not _matches(meta, ["other.zip"])

    def test_matches_original_filename(self):
        meta = {
            "dataFile": {
                "filename": "process_parameters.tab",
                "filesize": 1,
                "originalFileName": "process_parameters.csv",
            }
        }
        assert _matches(meta, ["process_parameters.csv"])

    def test_matches_darus_internal_filename(self):
        """The DaRUS-internal name matches even when an original name exists."""
        meta = {
            "dataFile": {
                "filename": "process_parameters.tab",
                "filesize": 1,
                "originalFileName": "process_parameters.csv",
            }
        }
        assert _matches(meta, ["process_parameters.tab"])


class TestDatasetTitle:
    def test_extracts_title(self):
        data = {
            "metadataBlocks": {
                "citation": {
                    "fields": [
                        {"typeName": "author", "value": "x"},
                        {"typeName": "title", "value": "RDDAC Dataset"},
                    ]
                }
            }
        }
        assert _dataset_title(data) == "RDDAC Dataset"

    def test_missing_title_is_empty(self):
        assert _dataset_title({}) == ""


class TestArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_main_no_args_shows_help(self, capsys):
        """Test that running without args shows help."""
        with patch("sys.argv", ["rddac"]):
            main()
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "rddac" in captured.out

    def test_version_flag(self):
        """Test --version flag."""
        with patch("sys.argv", ["rddac", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_info_command_parsing(self):
        """Test info command is recognized."""
        with patch("sys.argv", ["rddac", "info"]):
            with patch("rddac.cli.cmd_info") as mock_info:
                main()
                mock_info.assert_called_once()

    def test_download_command_parsing(self):
        """Test download command is recognized."""
        with patch("sys.argv", ["rddac", "download"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                mock_download.assert_called_once()

    def test_download_with_version(self):
        """Test download with specific version (incl. the :draft form)."""
        with patch("sys.argv", ["rddac", "download", ":draft"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.version == ":draft"

    def test_download_default_version(self):
        """Test download uses the configured default version."""
        with patch("sys.argv", ["rddac", "download"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.version == DEFAULT_VERSION

    def test_download_small_flag(self):
        """Test --small flag."""
        with patch("sys.argv", ["rddac", "download", "--small"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.small is True

    def test_download_no_sim_flag(self):
        """Test --no-sim flag (RDDAC-specific: skips the DDACS simulations)."""
        with patch("sys.argv", ["rddac", "download", "--no-sim"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.no_sim is True

    def test_download_no_sim_default_false(self):
        with patch("sys.argv", ["rddac", "download"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.no_sim is False

    def test_download_output_dir(self):
        """Test --out flag."""
        with patch("sys.argv", ["rddac", "download", "--out", "/custom/path"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.out == "/custom/path"

    def test_download_yes_flag(self):
        """Test -y/--yes flag."""
        with patch("sys.argv", ["rddac", "download", "-y"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.yes is True

    def test_download_extract_flag(self):
        """Test --extract opt-in flag."""
        with patch("sys.argv", ["rddac", "download", "--extract"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.extract is True

    def test_download_remove_zip_flag(self):
        """Test --remove-zip flag."""
        with patch("sys.argv", ["rddac", "download", "--extract", "--remove-zip"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.remove_zip is True

    def test_download_specific_files(self):
        """Test --files flag."""
        with patch("sys.argv", ["rddac", "download", "--files", "a.csv", "b.zip"]):
            with patch("rddac.cli.cmd_download") as mock_download:
                main()
                args = mock_download.call_args[0][0]
                assert args.files == ["a.csv", "b.zip"]

    def test_token_flag(self):
        """Test --token flag."""
        with patch("sys.argv", ["rddac", "--token", "my-secret-token", "info"]):
            with patch("rddac.cli.cmd_info") as mock_info:
                main()
                args = mock_info.call_args[0][0]
                assert args.token == "my-secret-token"


class TestCmdInfo:
    """Tests for cmd_info command."""

    @patch("ddacs.cli._api_get")
    def test_cmd_info_api_failure(self, mock_api_get, capsys):
        """Test info command handles API failure gracefully."""
        mock_api_get.return_value = None

        args = argparse.Namespace(token=None)
        cmd_info(args)

        # Should not raise, just return early
        mock_api_get.assert_called_once()

    @patch("ddacs.cli._api_get")
    def test_cmd_info_success(self, mock_api_get, capsys):
        """Test info command displays version information."""
        mock_api_get.return_value = [
            {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "versionState": "RELEASED",
                "releaseTime": "2026-01-01T00:00:00Z",
                "fileCount": 5,
                "license": {"name": "CC BY 4.0"},
                "files": [{"dataFile": {"filename": "sample.zip"}}],
            }
        ]

        args = argparse.Namespace(token=None)
        cmd_info(args)

        captured = capsys.readouterr()
        # cmd_info prints the dataset URL and the version table. Asserting on
        # those guarantees the panels rendered without coupling to the
        # metadataBlocks-derived title (which the mock doesn't include).
        assert "darus.uni-stuttgart.de" in captured.out
        assert "1.0" in captured.out


class TestCmdDownload:
    """Tests for cmd_download command."""

    @patch("ddacs.cli._api_get")
    def test_cmd_download_api_failure(self, mock_api_get):
        """Test download command handles API failure gracefully."""
        mock_api_get.return_value = None

        cmd_download(_download_args())

        mock_api_get.assert_called_once()

    @patch("ddacs.cli._api_get")
    def test_cmd_download_no_matching_files(self, mock_api_get, capsys):
        """Test download when no files match criteria."""
        mock_api_get.return_value = _version_payload(
            [{"dataFile": {"filename": "other.csv", "filesize": 100}}]
        )

        cmd_download(_download_args(files=["nonexistent.csv"]))

        captured = capsys.readouterr()
        assert "No files found" in captured.out

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_success(self, mock_api_get, mock_requests_get, tmp_path):
        """Test successful file download (--no-sim: single metadata fetch)."""
        mock_api_get.return_value = _version_payload(
            [
                {
                    "dataFile": {"id": 123, "filename": "test.csv", "filesize": 100},
                    "description": "Test file",
                }
            ]
        )

        # Mock the download request
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(return_value=[b"test content"])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        cmd_download(_download_args(out=str(tmp_path)))

        # Check file was created and no DDACS metadata fetch happened (--no-sim).
        assert (tmp_path / "test.csv").exists()
        mock_api_get.assert_called_once()

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_full_delegates_ddacs_sims(
        self, mock_api_get, mock_requests_get, tmp_path, capsys
    ):
        """Without --no-sim the simulation fetch is delegated to the installed
        `ddacs` CLI (subprocess); only ONE DaRUS metadata fetch happens here."""
        rddac_payload = _version_payload(
            [{"dataFile": {"id": 1, "filename": "test.csv", "filesize": 100}}]
        )
        mock_api_get.side_effect = [rddac_payload]

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(return_value=[b"test content"])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cmd_download(_download_args(no_sim=False, out=str(tmp_path)))

        assert mock_api_get.call_count == 1  # only the RDDAC metadata fetch
        delegated = mock_run.call_args[0][0]
        assert delegated[1:8] == ["-m", "ddacs.cli", "download", "--files", "rddac.zip",
                                  "metadata.json", "process_parameters.csv"]
        captured = capsys.readouterr()
        assert "rddac.zip" in captured.out  # the sim panel rendered

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_sims_without_ddacs_installed(
        self, mock_api_get, mock_requests_get, tmp_path, capsys
    ):
        """When `ddacs` is not installed, the sim leg prints install
        instructions instead of failing."""
        rddac_payload = _version_payload(
            [{"dataFile": {"id": 1, "filename": "test.csv", "filesize": 100}}]
        )
        mock_api_get.side_effect = [rddac_payload]

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = MagicMock(return_value=[b"test content"])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        with patch("importlib.util.find_spec", return_value=None):
            cmd_download(_download_args(no_sim=False, out=str(tmp_path)))

        captured = capsys.readouterr()
        assert "pip install ddacs" in captured.out

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_small_skips_sims(self, mock_api_get, mock_requests_get, tmp_path):
        """--small implies the DDACS simulations are not fetched."""
        mock_api_get.return_value = _version_payload(
            [{"dataFile": {"id": 1, "filename": "sample.zip", "filesize": 4}}]
        )

        zip_bytes = b"PK\x05\x06" + b"\x00" * 18  # empty zip
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(zip_bytes))}
        mock_response.iter_content = MagicMock(return_value=[zip_bytes])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        cmd_download(_download_args(small=True, no_sim=False, out=str(tmp_path)))

        mock_api_get.assert_called_once()

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_with_extraction(self, mock_api_get, mock_requests_get, tmp_path):
        """Test download with zip extraction."""
        # Create a real zip file for testing
        zip_content_path = tmp_path / "content"
        zip_content_path.mkdir()
        (zip_content_path / "0042.h5").write_text("test")

        zip_path = tmp_path / "source.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(zip_content_path / "0042.h5", "0042.h5")

        zip_bytes = zip_path.read_bytes()

        mock_api_get.return_value = _version_payload(
            [
                {
                    "dataFile": {"id": 123, "filename": "test.zip", "filesize": len(zip_bytes)},
                    "description": "Test zip",
                }
            ]
        )

        # Mock the download to return our zip file
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(zip_bytes))}
        mock_response.iter_content = MagicMock(return_value=[zip_bytes])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        output_dir = tmp_path / "output"
        cmd_download(_download_args(out=str(output_dir), extract=True, remove_zip=True))

        # Check extraction worked
        assert (output_dir / "0042.h5").exists()
        # Check zip was removed (remove_zip=True)
        assert not (output_dir / "test.zip").exists()

    @patch("ddacs.cli.requests.get")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_keep_zip(self, mock_api_get, mock_requests_get, tmp_path):
        """Test download with extract but no remove-zip — zip is kept."""
        # Create a real zip file
        zip_content_path = tmp_path / "content"
        zip_content_path.mkdir()
        (zip_content_path / "0042.h5").write_text("test")

        zip_path = tmp_path / "source.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(zip_content_path / "0042.h5", "0042.h5")

        zip_bytes = zip_path.read_bytes()

        mock_api_get.return_value = _version_payload(
            [
                {
                    "dataFile": {"id": 123, "filename": "test.zip", "filesize": len(zip_bytes)},
                    "description": "Test zip",
                }
            ]
        )

        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(zip_bytes))}
        mock_response.iter_content = MagicMock(return_value=[zip_bytes])
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_requests_get.return_value = mock_response

        output_dir = tmp_path / "output"
        cmd_download(_download_args(out=str(output_dir), extract=True, remove_zip=False))

        # Check both extracted file and zip exist
        assert (output_dir / "0042.h5").exists()
        assert (output_dir / "test.zip").exists()

    @patch("ddacs.cli.Confirm.ask")
    @patch("ddacs.cli._api_get")
    def test_cmd_download_user_cancels(self, mock_api_get, mock_confirm, capsys):
        """Test download cancelled by user."""
        mock_api_get.return_value = _version_payload(
            [{"dataFile": {"id": 123, "filename": "test.csv", "filesize": 100}}]
        )
        mock_confirm.return_value = False

        cmd_download(_download_args(yes=False))  # Will prompt

        captured = capsys.readouterr()
        assert "cancelled" in captured.out.lower()


class TestSmallTestSet:
    """Tests for --small flag behavior."""

    def test_small_set_composition(self):
        """The --small bundle is CSV + manifest + sample zip."""
        assert set(SMALL_TEST_FILES) == {
            "process_parameters.csv",
            "metadata.json",
            "sample.zip",
        }

    @patch("ddacs.cli._api_get")
    def test_small_filters_files_correctly(self, mock_api_get, capsys):
        """Test that --small only selects the small-bundle files."""
        mock_api_get.return_value = _version_payload(
            [
                {"dataFile": {"filename": "process_parameters.csv", "filesize": 100, "id": 1}},
                {"dataFile": {"filename": "metadata.json", "filesize": 80, "id": 2}},
                {"dataFile": {"filename": "sample.zip", "filesize": 1000, "id": 3}},
                {"dataFile": {"filename": "concave.zip", "filesize": 10000, "id": 4}},
                {"dataFile": {"filename": "convex.zip", "filesize": 10000, "id": 5}},
            ]
        )

        args = _download_args(small=True, no_sim=False, yes=False)

        # Mock Confirm to cancel — we only care about the preview table.
        with patch("ddacs.cli.Confirm.ask", return_value=False):
            cmd_download(args)

        captured = capsys.readouterr()
        # Should show the small-set files
        assert "process_parameters.csv" in captured.out
        assert "metadata.json" in captured.out
        assert "sample.zip" in captured.out
        # Should NOT show files outside the small set
        assert "concave.zip" not in captured.out
        assert "convex.zip" not in captured.out


# ---------------------------------------------------------------------------
# End-to-end `rddac download --small` (opt-in: RDDAC_RUN_DOWNLOAD_TESTS)
# ---------------------------------------------------------------------------


class TestDownloadSmallIntegration:
    """Integration tests against a real `rddac download --small` payload.

    Skipped unless ``RDDAC_RUN_DOWNLOAD_TESTS`` is set (the sample bundle is
    ~174 MB) or the download can't succeed (no token while the dataset is a
    draft, no network, ...).
    """

    def test_load_against_real_manifest(self, small_data_dir):
        import rddac

        ds = rddac.load(data_dir=str(small_data_dir))
        assert "RDDAC" in ds.metadata.name
        # We always ship process-parameters + field-map + several views.
        rs_ids = {rs.id for rs in ds.metadata.record_sets}
        assert {"process-parameters", "field-map", "force-curve"} <= rs_ids

    def test_mapping_picks_up_small_files(self, small_data_dir):
        import rddac

        ds = rddac.load(data_dir=str(small_data_dir))
        names = {str(n).split("/")[-1] for n in (ds.mapping or {}).values()}
        assert "sample.zip" in names
        assert "process_parameters.csv" in names

    def test_open_h5_on_sample_experiment(self, small_data_dir):
        import rddac

        # sample.zip ships one experiment per category; id 0 is always there.
        with rddac.open_h5(0, data_dir=str(small_data_dir)) as f:
            assert "force/data" in f
            assert int(f.attrs["id"]) == 0

    def test_inspect_h5_runs(self, small_data_dir, capsys):
        import rddac

        with rddac.open_h5(0, data_dir=str(small_data_dir)) as f:
            rddac.inspect_h5(f)
        out = capsys.readouterr().out
        assert "force/" in out
