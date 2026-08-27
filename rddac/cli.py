"""RDDAC dataset CLI — a thin front-end over the ``ddacs`` CLI machinery.

Provides commands to view dataset information and download files from the RDDAC
(Real Deep Drawing and Cutting) dataset hosted on DaRUS. Because RDDAC is the
experimental counterpart to DDACS, the full download also fetches the matching
DDACS simulations (skip with --no-sim).

The info/download implementation is `ddacs.cli`'s, called with
``spec=RDDAC_SPEC`` (requires ddacs >= 3.2.1). Only the parser (prog,
--no-sim) and the simulation leg live here.

Usage:
    rddac info                         # Show dataset info and versions
    rddac download                     # Real measurements + DDACS simulations
    rddac download --no-sim            # Real measurements only (skip simulations)
    rddac download --small             # Small sample bundle (quick start)
    rddac download --files a.zip       # Download specific files
    rddac download --extract           # Also extract zips next to the zip
    rddac download --extract --remove-zip
    rddac download --quiet             # No output/progress; implies --yes
    rddac preprocess                   # Process all modalities into the ML-ready layout
    rddac preprocess oil force         # Process a subset of modalities

Zip files are kept by default so they remain readable in place via mlcroissant
(the Croissant manifest references zip members directly).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

from ddacs import cli as _ddacs_cli
from ddacs.cli import (  # noqa: F401  — identical helper, re-exported for tests
    _dataset_title,
)
from rich.panel import Panel

from . import __version__
from ._preprocess.cli import add_preprocess_parser, cmd_preprocess
from .spec import BRAND_COLOR, DDACS_DATASET_DOI, DDACS_SIM_FILE, RDDAC_SPEC, SIM_SUBDIR

DEFAULT_VERSION = RDDAC_SPEC.default_version
DEFAULT_DATA_DIR = RDDAC_SPEC.default_data_dir
SMALL_TEST_FILES = list(RDDAC_SPEC.small_test_files)

console = _ddacs_cli.console  # shared console: ddacs's --quiet handling applies


# ── small helpers kept for tests / tooling (dataset-agnostic) ─────────────────
def _file_info(file_meta: dict) -> tuple[str, int]:
    """Original filename + size from a DaRUS file metadata entry."""
    df = file_meta["dataFile"]
    if "originalFileName" in df:
        return df["originalFileName"], df.get("originalFileSize", df["filesize"])
    return df["filename"], df["filesize"]


def _matches(file_meta: dict, names: list[str]) -> bool:
    return _file_info(file_meta)[0] in names or file_meta["dataFile"]["filename"] in names


# ── commands ──────────────────────────────────────────────────────────────────
def cmd_info(args: argparse.Namespace) -> None:
    """Display dataset information and available versions (via ddacs.cli)."""
    _ddacs_cli.cmd_info(args, spec=RDDAC_SPEC)


def cmd_download(args: argparse.Namespace) -> None:
    """Download RDDAC measurements (and, by default, the DDACS simulations)."""
    _ddacs_cli.cmd_download(args, spec=RDDAC_SPEC)

    # The DDACS simulations come along only on a full download (not --small /
    # --files), unless explicitly skipped.
    if not args.small and not args.files and not args.no_sim:
        _download_simulations(args)


def _download_simulations(args: argparse.Namespace) -> int:
    """Fetch the matching DDACS simulations by delegating to the `ddacs` CLI.

    The download machinery is not duplicated here: if the `ddacs` package is
    installed, its own CLI downloads `rddac.zip` into ``<out>/simulation``;
    otherwise the user gets the exact command to run after installing it.

    Returns the number of files fetched (0 when skipped or delegated-and-failed).
    """
    sim_dir = os.path.join(args.out, SIM_SUBDIR)
    console.print()
    console.print(
        Panel(
            f"[bold]Source:[/bold] DDACS {DDACS_DATASET_DOI}\n[bold]File:[/bold] {DDACS_SIM_FILE}\n"
            f"[bold]Destination:[/bold] {os.path.abspath(sim_dir)}\n"
            "[dim]The matching FEM simulations. Skip with --no-sim.[/dim]",
            title="DDACS simulation reference data",
            border_style=BRAND_COLOR,
        )
    )

    if importlib.util.find_spec("ddacs") is None:
        console.print(
            "[yellow]The `ddacs` package is not installed — skipping the simulations.[/yellow]\n"
            "To fetch them later:\n"
            "  [bold]pip install ddacs[/bold]\n"
            f"  [bold]ddacs download --files {DDACS_SIM_FILE} metadata.json process_parameters.csv --out {sim_dir} -y[/bold]"
        )
        return 0

    # Also fetch DDACS's manifest + parameter table so <out>/simulation is a
    # self-contained DDACS data dir: ddacs.load(data_dir="<out>/simulation")
    # resolves the DDACS manifest locally and cannot pick up RDDAC's
    # metadata.json from the parent directory.
    cmd = [
        sys.executable,
        "-m",
        "ddacs.cli",
        "download",
        "--files",
        DDACS_SIM_FILE,
        "metadata.json",
        "process_parameters.csv",
        "--out",
        sim_dir,
    ]
    if args.yes:
        cmd.append("-y")
    if getattr(args, "quiet", False):
        cmd.append("--quiet")
    if args.extract:
        cmd.append("--extract")
    if args.remove_zip:
        cmd.append("--remove-zip")
    console.print(f"[dim]delegating to: {' '.join(cmd[2:])}[/dim]")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]ddacs download failed.[/red]")
        return 0
    return 1


def main() -> None:
    """CLI entry point for RDDAC dataset commands."""
    parser = argparse.ArgumentParser(
        prog="rddac", description="RDDAC Dataset CLI - Download experimental data from DaRUS"
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--token", help="DaRUS API token (for draft access)")
    sub = parser.add_subparsers(dest="command", help="Command")

    sub.add_parser("info", help="Show dataset info and versions")

    dl = sub.add_parser("download", help="Download dataset files")
    dl.add_argument("version", nargs="?", default=DEFAULT_VERSION, help=f"Dataset version (default: {DEFAULT_VERSION})")
    dl.add_argument("--files", nargs="+", help="Specific filenames to download")
    dl.add_argument("--small", action="store_true", help="Download the small sample bundle")
    dl.add_argument(
        "--no-sim", action="store_true", help="Download only the real measurements (skip the DDACS simulations)"
    )
    dl.add_argument("--out", default=DEFAULT_DATA_DIR, help=f"Output directory (default: {DEFAULT_DATA_DIR})")
    dl.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    dl.add_argument("-q", "--quiet", action="store_true", help="No output or progress bars; implies --yes")
    dl.add_argument("--extract", action="store_true", help="Extract downloaded zips into their directory")
    dl.add_argument("--remove-zip", action="store_true", help="Delete zips after extraction (with --extract)")

    add_preprocess_parser(sub)

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
