"""CLI command handlers for PyBinaryGuard."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

from pybinaryguard.cli.formatters import get_formatter
from pybinaryguard.models.enums import ScanMode, Severity


# Map CLI severity names to Severity enum values
_SEVERITY_MAP = {
    "critical": Severity.CRITICAL,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
    "all": Severity.INFO,
}


def dispatch(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate command handler.

    Returns the exit code.
    """
    handlers = {
        "scan": cmd_scan,
        "check": cmd_check,
        "profile": cmd_profile,
        "doctor": cmd_doctor,
        "inspect": cmd_inspect,
        "snapshot": cmd_snapshot,
        "verify": cmd_verify,
        "export-tool-schema": cmd_export_tool_schema,
        "simulate": cmd_simulate,
        "validate": cmd_validate,
    }
    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 3
    return handler(args)


def _resolve_scan_mode(args: argparse.Namespace) -> ScanMode:
    """Determine scan mode from CLI flags."""
    if getattr(args, "fast", False):
        return ScanMode.FAST
    if getattr(args, "deep", False):
        return ScanMode.DEEP
    return ScanMode.STANDARD


def cmd_scan(args: argparse.Namespace) -> int:
    """Full environment scan."""
    from pybinaryguard.scanner import Scanner

    formatter = get_formatter(args.format, no_color=args.no_color)
    severity_threshold = _SEVERITY_MAP.get(args.severity, Severity.INFO)
    ignored_rules = set(args.ignore) if args.ignore else set()
    scan_mode = _resolve_scan_mode(args)

    scanner = Scanner(
        severity_threshold=severity_threshold,
        ignored_rules=ignored_rules,
        timeout=args.timeout,
        scan_mode=scan_mode,
    )

    report = scanner.run()
    profile = scanner.get_profile()

    output = formatter.format_scan(report, profile, verbose=args.verbose)
    print(output)

    return _exit_code(report.critical_count, report.warning_count)


def cmd_check(args: argparse.Namespace) -> int:
    """Check a specific installed package."""
    from pybinaryguard.scanner import Scanner

    formatter = get_formatter(args.format, no_color=args.no_color)
    ignored_rules = set(args.ignore) if args.ignore else set()
    scan_mode = _resolve_scan_mode(args)

    scanner = Scanner(
        packages=[args.package],
        ignored_rules=ignored_rules,
        timeout=args.timeout,
        scan_mode=scan_mode,
    )

    findings = scanner.check_package(args.package)
    profile = scanner.get_profile()

    output = formatter.format_check(findings, args.package, profile, verbose=args.verbose)
    print(output)

    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    warning = sum(1 for f in findings if f.severity == Severity.WARNING)
    return _exit_code(critical, warning)


def cmd_profile(args: argparse.Namespace) -> int:
    """Show system profile."""
    from pybinaryguard.scanner import Scanner

    formatter = get_formatter(args.format, no_color=args.no_color)

    scanner = Scanner(timeout=args.timeout)
    profile = scanner.get_profile()

    output = formatter.format_profile(profile)
    print(output)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Interactive troubleshooting wizard."""
    from pybinaryguard.diagnostics.explainer import diagnose_error
    from pybinaryguard.scanner import Scanner

    formatter = get_formatter(args.format, no_color=args.no_color)

    diagnosis: Dict[str, Any] = {}

    # If an error message is provided, diagnose it
    if args.error:
        diagnosis["error"] = args.error
        result = diagnose_error(args.error)
        if result:
            diagnosis["suggestions"] = [
                result.get("explanation", ""),
                result.get("fix_hint", ""),
            ]
        else:
            diagnosis["suggestions"] = [
                "Could not match this error to a known pattern.",
                "Try: pybinaryguard scan --verbose",
            ]

    # If a package is specified, check it
    if args.package:
        diagnosis["package"] = args.package
        scanner = Scanner(
            packages=[args.package],
            timeout=args.timeout,
        )
        findings = scanner.check_package(args.package)
        if findings:
            diagnosis["findings"] = findings

    # If neither error nor package, prompt for info
    if not args.error and not args.package:
        diagnosis["suggestions"] = [
            "Usage: pybinaryguard doctor --error 'your error message'",
            "       pybinaryguard doctor --package torch",
            "Or run: pybinaryguard scan  (for a full environment check)",
        ]

    output = formatter.format_doctor(diagnosis)
    print(output)

    # Exit code based on findings
    if "findings" in diagnosis:
        critical = sum(1 for f in diagnosis["findings"] if f.severity == Severity.CRITICAL)
        warning = sum(1 for f in diagnosis["findings"] if f.severity == Severity.WARNING)
        return _exit_code(critical, warning)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a .whl or .so file."""
    from pybinaryguard.scanner import Scanner

    formatter = get_formatter(args.format, no_color=args.no_color)

    try:
        scanner = Scanner(timeout=args.timeout)
        findings = scanner.inspect_file(args.file)
        profile = scanner.get_profile()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    output = formatter.format_check(findings, args.file, profile, verbose=args.verbose)
    print(output)

    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    warning = sum(1 for f in findings if f.severity == Severity.WARNING)
    return _exit_code(critical, warning)


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Create environment snapshot."""
    from pybinaryguard.scanner import Scanner
    from pybinaryguard.snapshot import create_snapshot

    # Get system profile and packages
    scanner = Scanner(timeout=args.timeout)
    profile = scanner.get_profile()

    # Discover packages
    report = scanner.run()  # Run full scan to get packages

    # Get packages from scanner (this is a simplified approach)
    # In a full implementation, we'd need to expose packages from Scanner
    packages = []  # Would be populated from scanner

    # Create snapshot
    lockfile = create_snapshot(
        profile=profile,
        packages=packages,
        include_hashes=not args.no_hashes
    )

    # Output lockfile
    if args.output:
        lockfile.save(args.output)
        print(f"Snapshot saved to {args.output}")
    else:
        print(lockfile.to_json())

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify environment against snapshot."""
    from pybinaryguard.scanner import Scanner
    from pybinaryguard.snapshot import load_lockfile, verify_snapshot

    # Load lockfile
    try:
        lockfile = load_lockfile(args.lockfile)
    except FileNotFoundError:
        print(f"Error: Lockfile not found: {args.lockfile}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Error loading lockfile: {exc}", file=sys.stderr)
        return 3

    # Get current system state
    scanner = Scanner(timeout=args.timeout)
    profile = scanner.get_profile()

    # Run scan to get packages
    report = scanner.run()

    # Verify
    packages = []  # Would be populated from scanner
    result = verify_snapshot(
        lockfile=lockfile,
        profile=profile,
        packages=packages,
        check_hashes=not args.no_hashes,
        run_compatibility_checks=not args.no_compat_checks
    )

    # Print results
    if result.success:
        print("✅ Verification PASSED")
        print(f"   Packages verified: {result.packages_verified}")
        print(f"   Binaries verified: {result.binaries_verified}")
    else:
        print("❌ Verification FAILED")
        print(f"   Errors: {result.error_count}")
        print(f"   Warnings: {result.warning_count}")
        print(f"   Hash mismatches: {result.hash_mismatches}")

        # Print issues
        for issue in result.issues[:10]:  # Limit to first 10
            severity_symbol = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
            print(f"\n{severity_symbol} {issue.category.upper()}: {issue.message}")
            if issue.package:
                print(f"   Package: {issue.package}")
            if issue.expected:
                print(f"   Expected: {issue.expected}")
            if issue.actual:
                print(f"   Actual: {issue.actual}")

        if len(result.issues) > 10:
            print(f"\n... and {len(result.issues) - 10} more issues")

    return 0 if result.success else 1


def cmd_export_tool_schema(args: argparse.Namespace) -> int:
    """Export tool schema for agent framework registration."""
    from pybinaryguard.agent.schema import export_tool_schema

    fmt = getattr(args, "schema_format", "openai")
    print(export_tool_schema(format=fmt))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Simulate package installation compatibility."""
    from pybinaryguard.agent import simulate_install

    result = simulate_install(args.package_spec)

    fmt = getattr(args, "format", "table")
    if fmt == "json":
        print(result.to_json())
    else:
        # Human-readable output
        status = "COMPATIBLE" if result.predicted_compatible else "INCOMPATIBLE"
        icon = "+" if result.predicted_compatible else "X"
        print(f"[{icon}] {result.package_spec}: {status}")
        print(f"    Confidence: {result.confidence:.0%}")
        print(f"    Risk Level: {result.risk_level}")

        if result.blockers:
            print(f"\n    Blockers ({len(result.blockers)}):")
            for b in result.blockers:
                print(f"      - {b['message']}")

        if result.warnings:
            print(f"\n    Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"      - {w['message']}")

    return 0 if result.predicted_compatible else 2


def cmd_validate(args: argparse.Namespace) -> int:
    """Test actual imports in isolated subprocesses."""
    import json

    from pybinaryguard.scanner import Scanner
    from pybinaryguard.validators.import_validator import ImportValidator

    scanner = Scanner(timeout=args.timeout, scan_mode=ScanMode.FAST)
    report = scanner.run()

    # Discover packages to test
    validator = ImportValidator(
        timeout=getattr(args, "import_timeout", 10.0),
    )

    # Get packages with binary extensions
    test_packages = getattr(args, "packages", None)
    packages_to_test = []

    # Walk site-packages to find packages with top_level.txt
    import os
    for sp in scanner.get_profile().site_packages_paths:
        if not os.path.isdir(sp):
            continue
        for entry in os.listdir(sp):
            if entry.endswith(".dist-info"):
                pkg_name = entry.rsplit("-", 2)[0].replace("_", "-")
                if test_packages and pkg_name not in test_packages:
                    continue
                top_level = validator.get_top_level_name(
                    os.path.join(sp, entry), pkg_name
                )
                # Only test packages with binary files (unless specific list given)
                if test_packages:
                    packages_to_test.append((pkg_name, top_level))
                else:
                    # Check if package has .so files
                    pkg_dir = os.path.join(sp, top_level)
                    has_so = False
                    if os.path.isdir(pkg_dir):
                        for root, _dirs, files in os.walk(pkg_dir):
                            if any(f.endswith(".so") for f in files):
                                has_so = True
                                break
                    if has_so:
                        packages_to_test.append((pkg_name, top_level))

    if not packages_to_test:
        print("No packages with binary extensions found to validate.")
        return 0

    fmt = getattr(args, "format", "table")
    print(f"Testing {len(packages_to_test)} package imports...\n")

    results = validator.test_packages(packages_to_test)
    failures = [r for r in results if not r.success]

    if fmt == "json":
        output = json.dumps([r.as_dict() for r in results], indent=2)
        print(output)
    else:
        for r in results:
            if r.success:
                print(f"  [OK]   {r.package_name} ({r.duration_ms:.0f}ms)")
            else:
                print(f"  [FAIL] {r.package_name}: {r.error_type} — {r.error_message}")
                if r.category:
                    print(f"         Category: {r.category}")

        print(f"\n{len(results) - len(failures)} passed, {len(failures)} failed")

    return 2 if failures else 0


def _exit_code(critical: int, warning: int) -> int:
    """Compute exit code from finding counts.

    0 = all passed, 1 = warnings only, 2 = critical issues.
    """
    if critical > 0:
        return 2
    if warning > 0:
        return 1
    return 0
