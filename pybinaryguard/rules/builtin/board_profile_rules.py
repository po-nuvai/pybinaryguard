"""Board profile-aware rules for embedded device intelligence.

These rules leverage board profiles to provide targeted diagnostics for
specific embedded hardware platforms.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.profiles import match_board_profile
from pybinaryguard.rules.base import Rule


class KNOWN_BROKEN_WHEEL(Rule):
    """Detects packages known to be broken on the detected board."""

    rule_id = "KNOWN_BROKEN_WHEEL"
    description = "Checks packages against board-specific broken wheel database"

    def is_applicable(self, profile: SystemProfile) -> bool:
        board_profile = match_board_profile(profile)
        return board_profile is not None

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []
        board_profile = match_board_profile(profile)
        if board_profile is None:
            return findings

        for package in packages:
            if not package.has_binaries:
                continue
            package_version = package.package_version or None
            broken = board_profile.is_package_known_broken(
                package.package_name, package_version
            )
            if broken:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{package.package_name} is known broken on "
                            f"{board_profile.display_name}"
                        ),
                        explanation=(
                            f"Package {package.package_name} version "
                            f"{broken.versions} is incompatible with "
                            f"{board_profile.display_name}. "
                            f"Reason: {broken.reason}"
                        ),
                        package=package.package_name,
                        suggestion=broken.recommendation,
                        confidence=1.0,
                    )
                )
        return findings


class BOARD_INCOMPATIBLE_PACKAGE(Rule):
    """Detects packages fundamentally incompatible with the board."""

    rule_id = "BOARD_INCOMPATIBLE_PACKAGE"
    description = "Checks for packages that cannot work on detected hardware"

    def is_applicable(self, profile: SystemProfile) -> bool:
        board_profile = match_board_profile(profile)
        return board_profile is not None

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []
        board_profile = match_board_profile(profile)
        if board_profile is None:
            return findings

        for package in packages:
            if board_profile.is_package_incompatible(package.package_name):
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{package.package_name} is incompatible with "
                            f"{board_profile.display_name}"
                        ),
                        explanation=(
                            f"Package {package.package_name} requires hardware "
                            f"features not available on {board_profile.display_name}."
                        ),
                        package=package.package_name,
                        suggestion=(
                            f"Remove {package.package_name} or use a board "
                            f"with required hardware features"
                        ),
                        confidence=1.0,
                    )
                )
        return findings


class BOARD_CUDA_VERSION_MISMATCH(Rule):
    """Detects CUDA version exceeding board maximum."""

    rule_id = "BOARD_CUDA_VERSION_MISMATCH"
    description = "Checks if package requires CUDA version beyond board limit"

    def is_applicable(self, profile: SystemProfile) -> bool:
        board_profile = match_board_profile(profile)
        return (
            board_profile is not None
            and board_profile.max_cuda_version is not None
        )

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []
        board_profile = match_board_profile(profile)
        if board_profile is None or board_profile.max_cuda_version is None:
            return findings

        for package in packages:
            if not package.cuda_version:
                continue
            try:
                pkg_major = int(package.cuda_version.split(".")[0])
                board_max_major = int(
                    board_profile.max_cuda_version.split(".")[0]
                )
                if pkg_major > board_max_major:
                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity=Severity.CRITICAL,
                            title=(
                                f"{package.package_name} requires CUDA "
                                f"{package.cuda_version}, but "
                                f"{board_profile.display_name} max is "
                                f"{board_profile.max_cuda_version}"
                            ),
                            explanation=(
                                f"Package {package.package_name} requires "
                                f"CUDA {package.cuda_version}, but "
                                f"{board_profile.display_name} only supports "
                                f"up to CUDA {board_profile.max_cuda_version}."
                            ),
                            package=package.package_name,
                            suggestion=(
                                f"Install a version of {package.package_name} "
                                f"compatible with CUDA "
                                f"{board_profile.max_cuda_version}"
                            ),
                            confidence=0.95,
                        )
                    )
            except (ValueError, IndexError):
                pass
        return findings


class BOARD_GLIBC_MISMATCH(Rule):
    """Detects GLIBC mismatch against board recommendations."""

    rule_id = "BOARD_GLIBC_MISMATCH"
    description = "Checks if system GLIBC differs from board recommendation"

    def is_applicable(self, profile: SystemProfile) -> bool:
        board_profile = match_board_profile(profile)
        return (
            board_profile is not None
            and board_profile.recommended_glibc is not None
            and profile.glibc_version is not None
        )

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []
        board_profile = match_board_profile(profile)
        if board_profile is None or board_profile.recommended_glibc is None:
            return findings

        system_glibc = profile.glibc_version
        if system_glibc is None:
            return findings

        recommended = board_profile.recommended_glibc
        try:
            rec_parts = tuple(int(x) for x in recommended.split(".")[:2])
            if system_glibc != rec_parts:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=(
                            f"GLIBC {system_glibc[0]}.{system_glibc[1]} "
                            f"differs from {board_profile.display_name} "
                            f"recommended {recommended}"
                        ),
                        explanation=(
                            f"Your system has GLIBC "
                            f"{system_glibc[0]}.{system_glibc[1]}, but "
                            f"{board_profile.display_name} is tested with "
                            f"GLIBC {recommended}."
                        ),
                        suggestion=(
                            f"Consider using "
                            f"{board_profile.recommended_os or 'recommended OS'} "
                            f"which provides GLIBC {recommended}"
                        ),
                        confidence=0.7,
                    )
                )
        except (ValueError, IndexError):
            pass
        return findings


class BOARD_PYTHON_VERSION_UNSUPPORTED(Rule):
    """Detects Python version not validated for the board."""

    rule_id = "BOARD_PYTHON_VERSION_UNSUPPORTED"
    description = "Checks if Python version is in board's validated list"

    def is_applicable(self, profile: SystemProfile) -> bool:
        board_profile = match_board_profile(profile)
        return (
            board_profile is not None
            and len(board_profile.python_versions) > 0
        )

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []
        board_profile = match_board_profile(profile)
        if board_profile is None or not board_profile.python_versions:
            return findings

        py_version = (
            f"{profile.python_version[0]}.{profile.python_version[1]}"
        )
        if py_version not in board_profile.python_versions:
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.WARNING,
                    title=(
                        f"Python {py_version} not validated for "
                        f"{board_profile.display_name}"
                    ),
                    explanation=(
                        f"Python {py_version} has not been validated for "
                        f"{board_profile.display_name}. Validated: "
                        f"{', '.join(board_profile.python_versions)}"
                    ),
                    suggestion=(
                        f"Use Python {board_profile.python_versions[0]} "
                        f"for best compatibility"
                    ),
                    confidence=0.6,
                )
            )
        return findings


__all__ = [
    "KNOWN_BROKEN_WHEEL",
    "BOARD_INCOMPATIBLE_PACKAGE",
    "BOARD_CUDA_VERSION_MISMATCH",
    "BOARD_GLIBC_MISMATCH",
    "BOARD_PYTHON_VERSION_UNSUPPORTED",
]
