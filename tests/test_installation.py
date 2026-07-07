"""Test package installation and imports."""

import pytest


class TestInstallation:
    """Tests for package installation."""

    def test_package_import(self):
        """Test that the package can be imported."""
        import rddac

        assert rddac.__version__ is not None

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from rddac import cli

        assert cli.main is not None

    def test_version_format(self):
        """Test version string format."""
        import rddac

        version = rddac.__version__
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
