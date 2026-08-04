"""Output formatters for the PyBinaryGuard CLI.

This module provides three formatters:

- :class:`TableFormatter` -- human-readable table output with optional ANSI
  colour codes (default).
- :class:`JSONFormatter` -- machine-readable JSON output.
- :class:`MinimalFormatter` -- one-line-per-finding output.

All formatters implement the :class:`FormatterBase` abstract interface.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pybinaryguard.models import Finding, ScanReport, Severity, SystemProfile


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

class _Colours:
    """ANSI escape sequences for terminal colours."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    RED_BOLD = "\033[1;31m"
    YELLOW = "\033[33m"
    YELLOW_BOLD = "\033[1;33m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    DIM = "\033[2m"


class _NoColours:
    """Drop-in replacement that produces no escape sequences."""

    RESET = ""
    BOLD = ""
    RED = ""
    RED_BOLD = ""
    YELLOW = ""
    YELLOW_BOLD = ""
    GREEN = ""
    BLUE = ""
    CYAN = ""
    DIM = ""


def _pick_colours(no_color: bool) -> Any:
    """Return colour sequences unless --no-color or non-TTY."""
    if no_color or not sys.stdout.isatty():
        return _NoColours
    return _Colours


# ---------------------------------------------------------------------------
# Severity formatting helpers
# ---------------------------------------------------------------------------

def _severity_label(severity: Severity, c: Any) -> str:
    """Format a severity label with colour."""
    labels = {
        Severity.CRITICAL: f"{c.RED_BOLD}CRITICAL{c.RESET}",
        Severity.WARNING: f"{c.YELLOW_BOLD}WARNING{c.RESET}",
        Severity.INFO: f"{c.BLUE}INFO{c.RESET}",
        Severity.PASSED: f"{c.GREEN}PASSED{c.RESET}",
    }
    return labels.get(severity, severity.value.upper())


def _plain_severity_label(severity: Severity) -> str:
    """Format a severity label without colour."""
    return severity.value.upper()


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class FormatterBase(ABC):
    """Abstract base for all output formatters."""

    @abstractmethod
    def format_scan(
        self,
        report: ScanReport,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format a full scan report.

        Parameters
        ----------
        report:
            The scan report containing findings and metadata.
        profile:
            The system profile collected during the scan.
        verbose:
            When ``True``, include technical detail and suggestions.

        Returns
        -------
        str
            The formatted output string.
        """
        ...

    @abstractmethod
    def format_check(
        self,
        findings: List[Finding],
        package: str,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format findings for a single package check.

        Parameters
        ----------
        findings:
            The list of findings for the package.
        package:
            The package name that was checked.
        profile:
            The system profile.
        verbose:
            When ``True``, include technical detail and suggestions.

        Returns
        -------
        str
            The formatted output string.
        """
        ...

    @abstractmethod
    def format_profile(self, profile: SystemProfile) -> str:
        """Format a system profile for display.

        Parameters
        ----------
        profile:
            The system profile to display.

        Returns
        -------
        str
            The formatted output string.
        """
        ...

    @abstractmethod
    def format_doctor(self, diagnosis: Dict[str, Any]) -> str:
        """Format a diagnostic/doctor result.

        Parameters
        ----------
        diagnosis:
            A dictionary with diagnosis information.

        Returns
        -------
        str
            The formatted output string.
        """
        ...


# ---------------------------------------------------------------------------
# TableFormatter
# ---------------------------------------------------------------------------

_HEADER_LINE = "\u2501" * 49  # U+2501 BOX DRAWINGS HEAVY HORIZONTAL


class TableFormatter(FormatterBase):
    """Human-readable table output with optional ANSI colours.

    When stdout is a TTY and ``no_color`` is ``False``, findings are
    colour-coded by severity:

    - CRITICAL = red bold
    - WARNING = yellow bold
    - INFO = blue
    - PASSED = green
    """

    def __init__(self, no_color: bool = False) -> None:
        self._c = _pick_colours(no_color)

    def format_scan(
        self,
        report: ScanReport,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format a full scan report as a human-readable table."""
        c = self._c
        lines: List[str] = []

        # Header
        lines.append(
            f"{c.BOLD}PyBinaryGuard v1.0.4 \u2014 Binary Compatibility Scanner{c.RESET}"
        )
        lines.append(f"{c.DIM}{_HEADER_LINE}{c.RESET}")
        lines.append("")

        # System profile summary
        lines.append(f"{c.BOLD}System Profile{c.RESET}")
        for key, value in profile.summary().items():
            lines.append(f"  {key + ':':<16}{value}")
        lines.append("")

        # Detected board (embedded intelligence)
        if report.detected_board:
            lines.append(
                f"{c.CYAN}\u2022 Detected Board: {c.BOLD}{report.detected_board}{c.RESET}"
            )
            lines.append("")

        # Scanning summary
        lines.append(
            f"Scanning {report.total_packages} packages... "
            f"{c.GREEN}done{c.RESET}"
        )
        lines.append("")

        # Results summary
        parts: List[str] = []
        if report.critical_count:
            parts.append(f"{c.RED_BOLD}{report.critical_count} critical{c.RESET}")
        if report.warning_count:
            parts.append(f"{c.YELLOW_BOLD}{report.warning_count} warning{c.RESET}")
        if report.info_count:
            parts.append(f"{c.BLUE}{report.info_count} info{c.RESET}")
        passed = report.total_packages - (
            report.critical_count + report.warning_count + report.info_count
        )
        if passed > 0:
            parts.append(f"{c.GREEN}{passed} passed{c.RESET}")

        lines.append(f"Results: {', '.join(parts)}")
        lines.append(f"{c.DIM}{_HEADER_LINE}{c.RESET}")
        lines.append("")

        # Individual findings
        for finding in report.findings:
            lines.extend(self._format_finding(finding, verbose))
            lines.append("")

        # Health score with v2 breakdown
        score = report.health_score
        label = report.health_label
        if score >= 90:
            score_colour = c.GREEN
        elif score >= 70:
            score_colour = c.YELLOW
        else:
            score_colour = c.RED
        lines.append(
            f"Health Score: {score_colour}{score} / 100{c.RESET} ({label})"
        )

        # Category breakdown (v2 scoring)
        if report.score_breakdown is not None:
            breakdown = report.score_breakdown
            lines.append("")
            lines.append(f"{c.BOLD}Score Breakdown{c.RESET}")
            for cat in breakdown.categories.values():
                if cat.weight <= 0:
                    continue
                cat_score = round(cat.score, 1)
                if cat_score >= 90:
                    cat_colour = c.GREEN
                elif cat_score >= 70:
                    cat_colour = c.YELLOW
                else:
                    cat_colour = c.RED
                weight_pct = round(cat.weight * 100)
                bar = self._score_bar(cat_score)
                lines.append(
                    f"  {cat.display_name + ':':<24}"
                    f"{cat_colour}{bar} {cat_score:>5.1f}{c.RESET}"
                    f"  {c.DIM}({weight_pct}% weight){c.RESET}"
                )
                if cat.top_issues:
                    for issue in cat.top_issues[:2]:
                        lines.append(f"    {c.DIM}- {issue}{c.RESET}")

        # Timing
        if report.scan_duration_ms > 0:
            elapsed = report.scan_duration_ms / 1000.0
            lines.append(f"{c.DIM}Scan completed in {elapsed:.1f}s{c.RESET}")

        return "\n".join(lines)

    def format_check(
        self,
        findings: List[Finding],
        package: str,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format findings for a single package check."""
        c = self._c
        lines: List[str] = []

        lines.append(f"{c.BOLD}PyBinaryGuard \u2014 Package Check: {package}{c.RESET}")
        lines.append(f"{c.DIM}{_HEADER_LINE}{c.RESET}")
        lines.append("")

        if not findings:
            lines.append(f"{c.GREEN}No issues found for '{package}'.{c.RESET}")
            return "\n".join(lines)

        for finding in findings:
            lines.extend(self._format_finding(finding, verbose))
            lines.append("")

        # Summary
        crit = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        warn = sum(1 for f in findings if f.severity == Severity.WARNING)
        info = sum(1 for f in findings if f.severity == Severity.INFO)

        parts = []
        if crit:
            parts.append(f"{c.RED_BOLD}{crit} critical{c.RESET}")
        if warn:
            parts.append(f"{c.YELLOW_BOLD}{warn} warning{c.RESET}")
        if info:
            parts.append(f"{c.BLUE}{info} info{c.RESET}")
        if not parts:
            parts.append(f"{c.GREEN}all passed{c.RESET}")

        lines.append(f"Summary: {', '.join(parts)}")

        return "\n".join(lines)

    def format_profile(self, profile: SystemProfile) -> str:
        """Format a system profile for display."""
        c = self._c
        lines: List[str] = []

        lines.append(
            f"{c.BOLD}PyBinaryGuard v1.0.4 \u2014 System Profile{c.RESET}"
        )
        lines.append(f"{c.DIM}{_HEADER_LINE}{c.RESET}")
        lines.append("")

        for key, value in profile.summary().items():
            lines.append(f"  {key + ':':<16}{value}")

        # Additional details
        lines.append("")
        lines.append(f"{c.BOLD}Library Paths{c.RESET}")
        if profile.ld_library_path:
            lines.append(f"  LD_LIBRARY_PATH:")
            for p in profile.ld_library_path:
                lines.append(f"    {p}")
        else:
            lines.append(f"  LD_LIBRARY_PATH: {c.DIM}(not set){c.RESET}")

        lines.append("")
        lines.append(f"  Site-packages:")
        if profile.site_packages_paths:
            for p in profile.site_packages_paths:
                lines.append(f"    {p}")
        else:
            lines.append(f"    {c.DIM}(none found){c.RESET}")

        # CPU flags (if verbose context)
        if profile.cpu_flags:
            lines.append("")
            lines.append(f"{c.BOLD}CPU Features{c.RESET}")
            flags_str = ", ".join(sorted(profile.cpu_flags)[:20])
            if len(profile.cpu_flags) > 20:
                flags_str += f" ... and {len(profile.cpu_flags) - 20} more"
            lines.append(f"  {flags_str}")

        return "\n".join(lines)

    def format_doctor(self, diagnosis: Dict[str, Any]) -> str:
        """Format a diagnostic result."""
        c = self._c
        lines: List[str] = []

        lines.append(
            f"{c.BOLD}PyBinaryGuard \u2014 Doctor{c.RESET}"
        )
        lines.append(f"{c.DIM}{_HEADER_LINE}{c.RESET}")
        lines.append("")

        if "error" in diagnosis:
            lines.append(f"{c.BOLD}Diagnosing error:{c.RESET}")
            lines.append(f"  {diagnosis['error']}")
            lines.append("")

        if "package" in diagnosis:
            lines.append(f"{c.BOLD}Package:{c.RESET} {diagnosis['package']}")
            lines.append("")

        if "findings" in diagnosis:
            for finding in diagnosis["findings"]:
                lines.extend(self._format_finding(finding, verbose=True))
                lines.append("")

        if "suggestions" in diagnosis:
            lines.append(f"{c.BOLD}Suggestions:{c.RESET}")
            for i, suggestion in enumerate(diagnosis["suggestions"], 1):
                lines.append(f"  {i}. {suggestion}")

        if not diagnosis.get("findings") and not diagnosis.get("suggestions"):
            lines.append(
                f"{c.GREEN}No issues detected. The environment looks healthy.{c.RESET}"
            )

        return "\n".join(lines)

    @staticmethod
    def _score_bar(score: float, width: int = 10) -> str:
        """Render a text-based score bar: [========  ]."""
        filled = round(score / 100.0 * width)
        empty = width - filled
        return "[" + "\u2588" * filled + " " * empty + "]"

    def _format_finding(self, finding: Finding, verbose: bool) -> List[str]:
        """Format a single finding for table output."""
        c = self._c
        lines: List[str] = []

        severity_str = _severity_label(finding.severity, c)

        # First line: severity + package + version
        header = severity_str
        if finding.package:
            version_part = f" {finding.package_version}" if finding.package_version else ""
            header += f"  {finding.package}{version_part}"
        lines.append(header)

        # Title
        lines.append(f"          {c.BOLD}{finding.title}{c.RESET}")

        # Explanation
        lines.append(f"          {finding.explanation}")

        # Verbose details
        if verbose and finding.technical_detail:
            lines.append(
                f"          {c.DIM}Detail: {finding.technical_detail}{c.RESET}"
            )

        # Suggestion (always shown for CRITICAL)
        if finding.suggestion and (verbose or finding.severity == Severity.CRITICAL):
            lines.append(f"          {c.CYAN}Fix: {finding.suggestion}{c.RESET}")

        return lines


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

class JSONFormatter(FormatterBase):
    """Machine-readable JSON output.

    Produces a JSON object with top-level keys ``version``, ``profile``,
    ``scan``, and ``findings``.
    """

    def format_scan(
        self,
        report: ScanReport,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format a full scan report as JSON."""
        scan_data: Dict[str, Any] = {
            "total_packages": report.total_packages,
            "packages_scanned": report.packages_scanned,
            "scan_duration_ms": round(report.scan_duration_ms, 1),
            "health_score": report.health_score,
            "health_label": report.health_label,
            "summary": {
                "critical": report.critical_count,
                "warning": report.warning_count,
                "info": report.info_count,
                "passed": report.passed_count,
            },
        }

        # Add detected board if available
        if report.detected_board:
            scan_data["detected_board"] = report.detected_board

        # Add v2 score breakdown if available
        if report.score_breakdown is not None:
            scan_data["score_breakdown"] = report.score_breakdown.as_dict()

        output: Dict[str, Any] = {
            "version": "1.0.4",
            "profile": self._profile_dict(profile),
            "scan": scan_data,
            "findings": [f.as_dict() for f in report.findings],
        }
        return json.dumps(output, indent=2, default=str)

    def format_check(
        self,
        findings: List[Finding],
        package: str,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format check findings as JSON."""
        output: Dict[str, Any] = {
            "version": "1.0.4",
            "package": package,
            "findings": [f.as_dict() for f in findings],
            "summary": {
                "critical": sum(1 for f in findings if f.severity == Severity.CRITICAL),
                "warning": sum(1 for f in findings if f.severity == Severity.WARNING),
                "info": sum(1 for f in findings if f.severity == Severity.INFO),
                "passed": sum(1 for f in findings if f.severity == Severity.PASSED),
            },
        }
        return json.dumps(output, indent=2, default=str)

    def format_profile(self, profile: SystemProfile) -> str:
        """Format a system profile as JSON."""
        output: Dict[str, Any] = {
            "version": "1.0.4",
            "profile": self._profile_dict(profile),
        }
        return json.dumps(output, indent=2, default=str)

    def format_doctor(self, diagnosis: Dict[str, Any]) -> str:
        """Format a doctor result as JSON."""
        output: Dict[str, Any] = {
            "version": "1.0.4",
            "diagnosis": {},
        }
        if "error" in diagnosis:
            output["diagnosis"]["error"] = diagnosis["error"]
        if "package" in diagnosis:
            output["diagnosis"]["package"] = diagnosis["package"]
        if "findings" in diagnosis:
            output["diagnosis"]["findings"] = [
                f.as_dict() for f in diagnosis["findings"]
            ]
        if "suggestions" in diagnosis:
            output["diagnosis"]["suggestions"] = diagnosis["suggestions"]
        return json.dumps(output, indent=2, default=str)

    @staticmethod
    def _profile_dict(profile: SystemProfile) -> Dict[str, Any]:
        """Convert a SystemProfile to a JSON-serializable dict."""
        py_ver = ".".join(str(v) for v in profile.python_version)
        result: Dict[str, Any] = {
            "python": {
                "version": py_ver,
                "abi_tag": profile.python_abi_tag,
                "implementation": profile.python_implementation,
                "executable": profile.python_executable,
                "debug_build": profile.python_debug_build,
            },
            "os": {
                "name": profile.os_name,
                "version": profile.os_version,
                "kernel": profile.kernel_version,
            },
            "architecture": profile.architecture.value,
        }
        if profile.glibc_version:
            result["glibc_version"] = (
                f"{profile.glibc_version[0]}.{profile.glibc_version[1]}"
            )
        if profile.musl_version:
            result["musl_version"] = (
                f"{profile.musl_version[0]}.{profile.musl_version[1]}"
            )
        if profile.gpu_available:
            result["gpu"] = {
                "driver_version": profile.gpu_driver_version,
                "name": profile.gpu_name,
            }
            if profile.cuda_runtime_version:
                result["gpu"]["cuda_runtime"] = (
                    f"{profile.cuda_runtime_version[0]}.{profile.cuda_runtime_version[1]}"
                )
        if profile.is_container:
            result["container"] = profile.container_runtime.value
        if profile.is_embedded_board:
            result["board"] = profile.board_name
        return result


# ---------------------------------------------------------------------------
# MinimalFormatter
# ---------------------------------------------------------------------------

class MinimalFormatter(FormatterBase):
    """One-line-per-finding output.

    Example: ``CRITICAL: torch 2.1.0 - CUDA version mismatch``
    """

    def format_scan(
        self,
        report: ScanReport,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format a scan report with one line per finding."""
        lines: List[str] = []
        for finding in report.findings:
            lines.append(self._format_finding_line(finding))

        # Summary line
        lines.append(
            f"--- {report.critical_count} critical, "
            f"{report.warning_count} warning, "
            f"{report.info_count} info | "
            f"score: {report.health_score}/100"
        )
        return "\n".join(lines)

    def format_check(
        self,
        findings: List[Finding],
        package: str,
        profile: SystemProfile,
        verbose: bool = False,
    ) -> str:
        """Format check findings with one line per finding."""
        if not findings:
            return f"OK: {package} - no issues found"
        lines: List[str] = []
        for finding in findings:
            lines.append(self._format_finding_line(finding))
        return "\n".join(lines)

    def format_profile(self, profile: SystemProfile) -> str:
        """Format a minimal profile summary."""
        parts: List[str] = []
        for key, value in profile.summary().items():
            parts.append(f"{key}: {value}")
        return " | ".join(parts)

    def format_doctor(self, diagnosis: Dict[str, Any]) -> str:
        """Format a minimal doctor result."""
        lines: List[str] = []
        if "error" in diagnosis:
            lines.append(f"Error: {diagnosis['error']}")
        if "findings" in diagnosis:
            for finding in diagnosis["findings"]:
                lines.append(self._format_finding_line(finding))
        if "suggestions" in diagnosis:
            for suggestion in diagnosis["suggestions"]:
                lines.append(f"Suggestion: {suggestion}")
        if not lines:
            lines.append("OK: no issues detected")
        return "\n".join(lines)

    @staticmethod
    def _format_finding_line(finding: Finding) -> str:
        """Format a single finding as one line."""
        label = _plain_severity_label(finding.severity)
        parts = [f"{label}:"]
        if finding.package:
            version_part = f" {finding.package_version}" if finding.package_version else ""
            parts.append(f"{finding.package}{version_part}")
            parts.append("-")
        parts.append(finding.title)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_formatter(name: str, no_color: bool = False) -> FormatterBase:
    """Return a formatter instance by name.

    Parameters
    ----------
    name:
        One of ``"table"``, ``"json"``, or ``"minimal"``.
    no_color:
        Disable ANSI colour codes (relevant for ``TableFormatter``).

    Returns
    -------
    FormatterBase
        The requested formatter.

    Raises
    ------
    ValueError
        If *name* is not a recognised formatter name.
    """
    formatters = {
        "table": lambda: TableFormatter(no_color=no_color),
        "json": lambda: JSONFormatter(),
        "minimal": lambda: MinimalFormatter(),
    }
    factory = formatters.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown formatter: {name!r}. "
            f"Available: {', '.join(sorted(formatters))}"
        )
    return factory()
