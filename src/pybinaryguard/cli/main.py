"""CLI entry point for PyBinaryGuard.

Usage::

    pybinaryguard scan
    pybinaryguard check torch
    pybinaryguard profile
    pybinaryguard doctor --error "GLIBC_2.34 not found"
    pybinaryguard inspect wheel.whl
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all commands and options."""
    # Parent parser for shared global options
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--format",
        choices=["table", "json", "minimal"],
        default="table",
        help="Output format (default: table)",
    )
    parent.add_argument(
        "--severity",
        choices=["critical", "warning", "info", "all"],
        default="all",
        help="Minimum severity to show (default: all)",
    )
    parent.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured output",
    )
    parent.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show technical details and all suggestions",
    )
    parent.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show critical findings",
    )
    parent.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: minimal output + strict exit codes",
    )
    parent.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Max scan time in seconds (default: 30)",
    )
    parent.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="RULE_ID",
        help="Rule IDs to ignore (e.g. CUDA_MINOR_MISMATCH)",
    )
    mode_group = parent.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fast",
        action="store_true",
        help="Fast scan: metadata only, skip binary analysis (<1s)",
    )
    mode_group.add_argument(
        "--deep",
        action="store_true",
        help="Deep scan: full symbol resolution + hash verification",
    )

    # Main parser
    parser = argparse.ArgumentParser(
        prog="pybinaryguard",
        description="PyBinaryGuard -- Binary Compatibility Scanner for Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent],
        epilog=(
            "Examples:\n"
            "  pybinaryguard scan               Full environment scan\n"
            "  pybinaryguard check torch         Check a specific package\n"
            "  pybinaryguard profile             Show system profile\n"
            "  pybinaryguard doctor --error 'GLIBC_2.34 not found'\n"
            "  pybinaryguard inspect wheel.whl   Analyse a wheel file\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.2",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan
    subparsers.add_parser(
        "scan", parents=[parent],
        help="Full environment scan -- check everything",
    )

    # check
    check_parser = subparsers.add_parser(
        "check", parents=[parent],
        help="Check a specific installed package",
    )
    check_parser.add_argument("package", help="Package name to check")

    # profile
    subparsers.add_parser(
        "profile", parents=[parent],
        help="Show system profile (no compatibility checks)",
    )

    # doctor
    doctor_parser = subparsers.add_parser(
        "doctor", parents=[parent],
        help="Interactive troubleshooting",
    )
    doctor_parser.add_argument("--error", help="Error message to diagnose")
    doctor_parser.add_argument("--package", help="Package that fails")

    # inspect
    inspect_parser = subparsers.add_parser(
        "inspect", parents=[parent],
        help="Analyse a .whl or .so file",
    )
    inspect_parser.add_argument("file", help="Path to .whl or .so file")

    # snapshot
    snapshot_parser = subparsers.add_parser(
        "snapshot", parents=[parent],
        help="Create environment snapshot (lockfile)",
    )
    snapshot_parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
    )
    snapshot_parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip computing binary hashes (faster)",
    )

    # verify
    verify_parser = subparsers.add_parser(
        "verify", parents=[parent],
        help="Verify environment against snapshot",
    )
    verify_parser.add_argument(
        "lockfile",
        help="Path to lockfile to verify against",
    )
    verify_parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip hash verification",
    )
    verify_parser.add_argument(
        "--no-compat-checks",
        action="store_true",
        help="Skip compatibility checks",
    )

    # export-tool-schema (agent SDK)
    schema_parser = subparsers.add_parser(
        "export-tool-schema", parents=[parent],
        help="Export tool schema for agent framework registration",
    )
    schema_parser.add_argument(
        "--schema-format",
        choices=["openai", "mcp", "json_schema"],
        default="openai",
        help="Schema format (default: openai)",
    )

    # simulate (agent SDK)
    simulate_parser = subparsers.add_parser(
        "simulate", parents=[parent],
        help="Predict package compatibility before installing",
    )
    simulate_parser.add_argument(
        "package_spec",
        help="Package specifier or wheel filename to simulate",
    )

    # validate (import testing)
    validate_parser = subparsers.add_parser(
        "validate", parents=[parent],
        help="Test actual imports in isolated subprocesses",
    )
    validate_parser.add_argument(
        "--packages",
        nargs="*",
        metavar="PKG",
        help="Specific packages to test (default: all with binaries)",
    )
    validate_parser.add_argument(
        "--import-timeout",
        type=float,
        default=10.0,
        help="Timeout per import test in seconds (default: 10)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Returns
    -------
    int
        Exit code: 0 = pass, 1 = warnings, 2 = critical, 3 = scanner error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Default to scan if no command given
    if not args.command:
        args.command = "scan"

    # Apply CI mode overrides
    if args.ci:
        args.format = "minimal"
        if args.severity == "all":
            args.severity = "critical"
        args.no_color = True

    # Apply quiet mode
    if args.quiet:
        args.severity = "critical"

    try:
        from pybinaryguard.cli.commands import dispatch

        return dispatch(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
