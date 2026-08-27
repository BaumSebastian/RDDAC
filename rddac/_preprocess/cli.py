"""The ``rddac preprocess`` subcommand — parser and dispatch.

Kept out of :mod:`rddac.cli` so the preprocessing feature lives entirely in
this package; the main CLI only attaches the parser and routes the command.
Mirrors the ``download`` UX (rich panel, ``-y``/``--quiet``, spec defaults).
"""

from __future__ import annotations

import argparse
import os

from ddacs import cli as _ddacs_cli
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm

from ..spec import BRAND_COLOR, RDDAC_SPEC
from . import config as config_mod
from .runner import MODALITIES, PROCESSED_MARKER, run

console = _ddacs_cli.console  # shared console: ddacs's --quiet handling applies
#: Errors go to stderr so they stay visible when --quiet silences the main console.
err_console = Console(stderr=True)


def add_preprocess_parser(sub: argparse._SubParsersAction) -> None:
    """Attach the ``preprocess`` subcommand to the main CLI's subparsers."""
    p = sub.add_parser("preprocess", help="Process raw measurements into the ML-ready layout")
    p.add_argument(
        "modalities", nargs="*", metavar="modality", help=f"Subset of: {', '.join(MODALITIES)} (default: all)"
    )
    p.add_argument(
        "--data-dir",
        default=RDDAC_SPEC.default_data_dir,
        help=f"Directory holding the raw dataset (default: {RDDAC_SPEC.default_data_dir})",
    )
    p.add_argument("--out", default=None, help="Output directory for processed files (default: <data-dir>/processed)")
    p.add_argument("--ids", help="Experiment id selection, e.g. '0-999' or '42,1035'")
    p.add_argument("--split", choices=["train", "val", "test"], help="Restrict to one of the predefined splits")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    p.add_argument(
        "--overwrite", action="store_true", help="Recompute modalities that already exist in the output files"
    )
    p.add_argument("--config", metavar="TOML", help="TOML file overriding processing parameters (see --dump-config)")
    p.add_argument(
        "--dump-config", action="store_true", help="Print the default processing parameters as TOML and exit"
    )
    p.add_argument(
        "--rebuild-models",
        action="store_true",
        help="Force retraining of the cached fin-classifier models (pointcloud stage)",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p.add_argument("-q", "--quiet", action="store_true", help="No output or progress bars; implies --yes")


def _guard_out_dir(data_dir: str, out_dir: str, console) -> None:
    """Refuse output locations that could shadow or mutate the raw dataset.

    Raw files are never modified — that guarantee breaks if processed files
    land in the raw directory: with extracted loose ``.h5`` the writer would
    append into the raw files themselves; with zips the loose processed files
    would shadow the raw dataset in the unified index while ``metadata.json``
    still describes raw. A subdirectory (the default ``<data-dir>/processed``)
    is always fine.
    """
    if os.path.realpath(out_dir) == os.path.realpath(data_dir):
        console.print(
            f"[red]Output directory equals the raw data directory:[/red] {os.path.abspath(out_dir)}\n"
            "Processed files must not be written into the raw dataset. Use a subdirectory, "
            "e.g. the default [bold]<data-dir>/processed[/bold]."
        )
        raise SystemExit(2)
    if os.path.isdir(out_dir) and not os.path.isfile(os.path.join(out_dir, PROCESSED_MARKER)):
        names = os.listdir(out_dir)
        raw_markers = sorted(n for n in names if n.endswith(".zip") or n == "metadata.json")
        if raw_markers:
            console.print(
                f"[red]Output directory looks like a raw dataset directory[/red] "
                f"({', '.join(raw_markers)} present): {os.path.abspath(out_dir)}\n"
                "Choose an empty or processed output directory, "
                "e.g. the default [bold]<data-dir>/processed[/bold]."
            )
            raise SystemExit(2)


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run the selected preprocessing modalities."""
    if args.dump_config:
        print(config_mod.dump())  # plain print: pipeable into a file
        return

    # --quiet silences all decorative output and progress bars and implies
    # --yes so the run proceeds unattended (errors still go to stderr).
    if args.quiet:
        args.yes = True
    console.quiet = args.quiet

    try:
        cfg = config_mod.load(args.config) if args.config else {}
    except (OSError, ValueError) as exc:
        err_console.print(f"[red]--config: {escape(str(exc))}[/red]")
        raise SystemExit(2)

    names = list(args.modalities) or list(MODALITIES)
    unknown = [n for n in names if n not in MODALITIES]
    if unknown:
        err_console.print(f"[red]Unknown modality: {', '.join(unknown)}[/red] " f"(valid: {', '.join(MODALITIES)})")
        raise SystemExit(2)

    out_dir = args.out or os.path.join(args.data_dir, "processed")
    _guard_out_dir(args.data_dir, out_dir, err_console)
    selection = args.ids or args.split or "all"
    if not args.quiet:
        console.print(
            Panel(
                f"[bold]Modalities:[/bold] {', '.join(names)}\n"
                f"[bold]Raw data:[/bold] {os.path.abspath(args.data_dir)}\n"
                f"[bold]Output:[/bold] {os.path.abspath(out_dir)}\n"
                f"[bold]Experiments:[/bold] {selection}   [bold]Workers:[/bold] {args.workers}\n"
                "[dim]Raw files are never modified; processed files are written separately.[/dim]",
                title="RDDAC preprocessing",
                border_style=BRAND_COLOR,
            )
        )
    if not args.yes and not Confirm.ask("Proceed?", default=False, console=console):
        console.print("Aborted.")
        return

    try:
        run(
            names,
            data_dir=args.data_dir,
            out_dir=out_dir,
            ids=args.ids,
            split=args.split,
            workers=args.workers,
            overwrite=args.overwrite,
            quiet=args.quiet,
            console=console,
            config=cfg,
            skip_unavailable=not args.modalities,
            rebuild_models=args.rebuild_models,
        )
    except FileNotFoundError as exc:  # explicit stage with missing prerequisites (e.g. simulations)
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        raise SystemExit(2)
