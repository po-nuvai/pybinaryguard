"""PyBinaryGuard -- Binary Compatibility Intelligence for Python.

Detect binary incompatibilities *before* they crash your program.

Quick start::

    import pybinaryguard

    report = pybinaryguard.scan()
    print(report.health_score)

    findings = pybinaryguard.check("torch")
    profile = pybinaryguard.profile()
"""

from __future__ import annotations

__version__ = "1.0.2"

from pybinaryguard.models.enums import ScanMode, Severity
from pybinaryguard.models.finding import Finding, ScanReport
from pybinaryguard.models.system import SystemProfile

__all__ = [
    "Finding",
    "ScanMode",
    "ScanReport",
    "Severity",
    "SystemProfile",
    "check",
    "inspect",
    "profile",
    "scan",
]


def scan(**kwargs: object) -> ScanReport:
    """Full environment scan.

    Returns a :class:`ScanReport` with all findings, health score, and
    package counts.  Accepts the same keyword arguments as
    :class:`~pybinaryguard.scanner.Scanner`.
    """
    from pybinaryguard.scanner import Scanner

    scanner = Scanner(**kwargs)  # type: ignore[arg-type]
    return scanner.run()


def check(package: str, **kwargs: object) -> list:
    """Check a specific installed package.

    Returns a list of :class:`Finding` objects for the given package.
    """
    from pybinaryguard.scanner import Scanner

    scanner = Scanner(packages=[package], **kwargs)  # type: ignore[arg-type]
    report = scanner.run()
    return report.findings


def profile() -> SystemProfile:
    """Collect the system profile without running compatibility checks."""
    from pybinaryguard.scanner import Scanner

    scanner = Scanner()
    return scanner.get_profile()


def inspect(file_path: str) -> list:
    """Inspect a ``.whl`` or ``.so`` file against the current system.

    Returns a list of :class:`Finding` objects.
    """
    from pybinaryguard.scanner import Scanner

    scanner = Scanner()
    return scanner.inspect_file(file_path)
