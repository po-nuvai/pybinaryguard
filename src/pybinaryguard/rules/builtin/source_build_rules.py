"""Source build detection rules.

Detects packages built from source (sdist) rather than prebuilt wheels,
and validates compiler compatibility. This addresses the blind spot that
neither pip nor Poetry catches — build-time ABI issues.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


class SourceBuildDetectionRule(Rule):
    """Detect packages that were compiled from source on this machine.

    Packages built from sdist may have been compiled with a different
    compiler version or flags than expected, leading to ABI issues.
    """

    rule_id = "SOURCE_BUILD_DETECTED"
    description = "Detect packages built from source distribution"

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        for pkg in packages:
            if pkg.is_pure_python:
                continue
            if not pkg.has_binaries:
                continue

            # If package has .so files but NO wheel tags, it was built from source
            if not pkg.wheel_tags:
                findings.append(Finding(
                    rule_id=self.rule_id,
                    severity=Severity.INFO,
                    title=f"{pkg.package_name} was built from source",
                    explanation=(
                        f"{pkg.package_name}=={pkg.package_version} has compiled "
                        f"extensions but no wheel tags, indicating it was installed "
                        f"from an sdist (source distribution) rather than a prebuilt "
                        f"wheel. Source builds may have ABI issues if the compiler "
                        f"or flags differ from what the package authors tested."
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                    suggestion=(
                        f"Consider installing a prebuilt wheel: "
                        f"pip install --only-binary :all: {pkg.package_name}"
                    ),
                    confidence=0.8,
                ))

        return findings


class SourceBuildNoCompilerRule(Rule):
    """Warn if source-built packages exist but no C compiler is available.

    If a package was built from source but gcc/clang are not currently
    present, rebuilds or updates will fail.
    """

    rule_id = "SOURCE_BUILD_NO_COMPILER"
    description = "Source-built packages with no compiler available"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return not profile.has_build_tools

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        source_built = [
            pkg for pkg in packages
            if not pkg.is_pure_python and pkg.has_binaries and not pkg.wheel_tags
        ]

        if source_built:
            names = ", ".join(p.package_name for p in source_built[:5])
            remaining = len(source_built) - 5
            suffix = f" (+{remaining} more)" if remaining > 0 else ""

            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title="Source-built packages but no C compiler found",
                explanation=(
                    f"Found {len(source_built)} package(s) built from source "
                    f"({names}{suffix}) but no C/C++ compiler (gcc, g++, clang) "
                    f"is available on this system. Upgrading or reinstalling these "
                    f"packages will fail without a compiler."
                ),
                suggestion=(
                    "Install build tools: apt install build-essential "
                    "(Debian/Ubuntu) or yum groupinstall 'Development Tools' (RHEL)"
                ),
                confidence=0.9,
            ))

        return findings


class MissingPythonHeadersRule(Rule):
    """Warn if Python dev headers are missing for source builds."""

    rule_id = "SOURCE_BUILD_NO_PYTHON_HEADERS"
    description = "Python development headers missing for source builds"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return not profile.has_python_dev_headers and profile.has_build_tools

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        source_built = [
            pkg for pkg in packages
            if not pkg.is_pure_python and pkg.has_binaries and not pkg.wheel_tags
        ]

        if source_built:
            py_ver = ".".join(str(v) for v in profile.python_version[:2])
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title="Python development headers not found",
                explanation=(
                    f"Found {len(source_built)} source-built package(s) but "
                    f"Python.h was not found. Building C extensions requires "
                    f"Python development headers."
                ),
                suggestion=(
                    f"Install Python headers: apt install python{py_ver}-dev "
                    f"(Debian/Ubuntu) or yum install python3-devel (RHEL)"
                ),
                confidence=0.85,
            ))

        return findings
