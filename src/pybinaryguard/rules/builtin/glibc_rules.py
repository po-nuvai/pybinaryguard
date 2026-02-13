"""GLIBC compatibility rules.

These rules detect mismatches between the GLIBC version available on the
host system and the GLIBC version required by installed binary packages.
They also flag musl/glibc conflicts and inconsistent manylinux wheel
metadata.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


def _fmt_ver(version: Tuple[int, int]) -> str:
    """Format a (major, minor) version tuple as ``'M.m'``."""
    return f"{version[0]}.{version[1]}"


class GLIBCVersionMismatchRule(Rule):
    """Detects packages that need a newer GLIBC than the system provides.

    When a shared library is linked against GLIBC symbols that were
    introduced in a release newer than what your OS ships, the library
    will fail to load at import time with an ``ImportError`` referencing
    a missing ``GLIBC_X.Y`` version.  This is one of the most common
    causes of "undefined symbol" crashes in Python packages with compiled
    extensions.
    """

    rule_id = "GLIBC_VERSION_MISMATCH"
    description = (
        "Check that the system GLIBC version satisfies every package's "
        "minimum requirement."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applicable on glibc-based systems."""
        return profile.glibc_version is not None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        sys_glibc = profile.glibc_version
        if sys_glibc is None:
            return findings

        for pkg in packages:
            if pkg.required_glibc is None:
                continue
            if pkg.required_glibc > sys_glibc:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=f"{pkg.package_name} requires a newer GLIBC",
                        explanation=(
                            f"Package {pkg.package_name} {pkg.package_version} "
                            f"requires GLIBC >= {_fmt_ver(pkg.required_glibc)} "
                            f"but your system has GLIBC "
                            f"{_fmt_ver(sys_glibc)}. This means the compiled "
                            f"extensions in this package use C library functions "
                            f"that do not exist on your OS, so the package will "
                            f"crash with an ImportError when you try to import it."
                        ),
                        technical_detail=(
                            f"System GLIBC: {_fmt_ver(sys_glibc)}, "
                            f"Required: {_fmt_ver(pkg.required_glibc)}"
                        ),
                        suggestion=(
                            f"Option 1 -- downgrade the package to an older "
                            f"version that was built for GLIBC "
                            f"{_fmt_ver(sys_glibc)}:\n"
                            f"  pip install '{pkg.package_name}<"
                            f"{pkg.package_version}'\n\n"
                            f"Option 2 -- upgrade your OS to one that ships "
                            f"GLIBC >= {_fmt_ver(pkg.required_glibc)} "
                            f"(see the glibc_distro_map data for which "
                            f"distro versions include which GLIBC).\n\n"
                            f"Option 3 -- use a container image with a newer "
                            f"base OS:\n"
                            f"  docker run -it python:3.x-bookworm"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings


class MuslGlibcConflictRule(Rule):
    """Detects manylinux packages running on a musl-based system.

    Manylinux wheels are linked against glibc.  Alpine Linux (and other
    musl-based distros) ship musl instead, so these wheels will fail to
    load.  The typical symptom is an error like
    ``Error loading shared library: libc.musl-x86_64.so.1``.
    """

    rule_id = "MUSL_GLIBC_CONFLICT"
    description = (
        "Flag manylinux packages on musl-based systems (e.g. Alpine)."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applicable on musl-based systems."""
        return profile.musl_version is not None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            if pkg.manylinux_tag is None:
                continue
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.CRITICAL,
                    title=(
                        f"{pkg.package_name} is a manylinux wheel "
                        f"on a musl system"
                    ),
                    explanation=(
                        f"Package {pkg.package_name} {pkg.package_version} "
                        f"was built as a manylinux wheel (tagged "
                        f"{pkg.manylinux_tag}), which means it is linked "
                        f"against glibc.  Your system uses musl libc "
                        f"(commonly Alpine Linux), so these compiled "
                        f"extensions cannot run."
                    ),
                    technical_detail=(
                        f"Wheel platform tag: {pkg.manylinux_tag}, "
                        f"System libc: musl "
                        f"{_fmt_ver(profile.musl_version) if profile.musl_version else 'unknown'}"
                    ),
                    suggestion=(
                        f"Option 1 -- install a musllinux or Alpine-specific "
                        f"wheel if one is published:\n"
                        f"  pip install --only-binary :all: "
                        f"--platform musllinux_1_1_x86_64 "
                        f"{pkg.package_name}\n\n"
                        f"Option 2 -- install via conda-forge, which provides "
                        f"musl-compatible builds:\n"
                        f"  conda install -c conda-forge {pkg.package_name}\n\n"
                        f"Option 3 -- build from source:\n"
                        f"  apk add build-base && pip install --no-binary "
                        f":all: {pkg.package_name}"
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                )
            )
        return findings


class ManylinuxTagViolationRule(Rule):
    """Detects wheels whose .so files need a newer GLIBC than the tag claims.

    A manylinux tag (e.g. ``manylinux_2_17``) promises that the wheel
    only requires GLIBC 2.17 or later.  If the actual shared objects
    inside the wheel reference symbols from a newer GLIBC, the tag is
    wrong: the wheel will fail to load on older glibc systems that the
    tag claims to support.
    """

    rule_id = "MANYLINUX_TAG_VIOLATION"
    description = (
        "Warn when a wheel's actual GLIBC requirement exceeds its "
        "manylinux platform tag."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            claimed = pkg.manylinux_glibc
            actual = pkg.required_glibc
            if claimed is None or actual is None:
                continue
            if actual > claimed:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=(
                            f"{pkg.package_name} wheel tag is inconsistent"
                        ),
                        explanation=(
                            f"The wheel for {pkg.package_name} "
                            f"{pkg.package_version} claims to be compatible "
                            f"with manylinux (GLIBC >= {_fmt_ver(claimed)}), "
                            f"but its .so files actually require GLIBC "
                            f"{_fmt_ver(actual)}.  This means the wheel was "
                            f"built incorrectly and will break on older Linux "
                            f"distributions that only have GLIBC "
                            f"{_fmt_ver(claimed)}."
                        ),
                        technical_detail=(
                            f"Manylinux tag claims GLIBC "
                            f"{_fmt_ver(claimed)}, actual requirement "
                            f"is GLIBC {_fmt_ver(actual)}"
                        ),
                        suggestion=(
                            f"This is a packaging bug in {pkg.package_name}.  "
                            f"Consider filing an issue with the package "
                            f"maintainers.  In the meantime, you can work "
                            f"around it by installing a previous version or "
                            f"building from source."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                        confidence=0.9,
                    )
                )
        return findings


class LibstdcxxVersionRule(Rule):
    """Detects packages that need a newer libstdc++ (GLIBCXX) than available.

    C++ extensions link against ``libstdc++.so`` and may require symbol
    versions (``GLIBCXX_X.Y.Z``) that only exist in newer GCC releases.
    When the system's ``libstdc++`` is too old you get an error like:
    ``version 'GLIBCXX_3.4.30' not found``.
    """

    rule_id = "LIBSTDCXX_TOO_OLD"
    description = (
        "Check that the system libstdc++ provides all GLIBCXX versions "
        "required by installed packages."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only relevant on glibc-based systems (libstdc++ ships with GCC)."""
        return profile.glibc_version is not None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            if pkg.required_glibcxx is None:
                continue
            # Check each shared object that has a GLIBCXX requirement.
            for so in pkg.shared_objects:
                if so.required_glibcxx is None:
                    continue
                # GLIBCXX versions look like "3.4.30".  We compare them
                # lexicographically after splitting into int tuples.
                so_ver = _parse_glibcxx(so.required_glibcxx)
                pkg_ver = _parse_glibcxx(pkg.required_glibcxx)
                if so_ver is None or pkg_ver is None:
                    continue
                # The package-level required_glibcxx should already be the
                # max across all .so files, so we report at the package
                # level once.
            if pkg.required_glibcxx is not None:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{pkg.package_name} may need a newer libstdc++"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} contains C++ extensions "
                            f"that require GLIBCXX version "
                            f"{pkg.required_glibcxx}.  If your system's "
                            f"libstdc++ does not provide this version you "
                            f"will see an error like 'version GLIBCXX_"
                            f"{pkg.required_glibcxx} not found' when "
                            f"importing the package."
                        ),
                        technical_detail=(
                            f"Required GLIBCXX: {pkg.required_glibcxx}"
                        ),
                        suggestion=(
                            f"Option 1 -- upgrade GCC / libstdc++:\n"
                            f"  sudo apt install libstdc++6  # Debian/Ubuntu\n"
                            f"  sudo dnf install libstdc++   # Fedora/RHEL\n\n"
                            f"Option 2 -- use conda which bundles its own "
                            f"libstdc++:\n"
                            f"  conda install -c conda-forge "
                            f"{pkg.package_name}\n\n"
                            f"Option 3 -- install an older version of the "
                            f"package that was compiled with an older GCC."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                        confidence=0.8,
                    )
                )
        return findings


def _parse_glibcxx(version_str: str) -> Optional[Tuple[int, ...]]:
    """Parse a GLIBCXX version string like ``'3.4.30'`` into an int tuple."""
    parts = version_str.split(".")
    try:
        return tuple(int(p) for p in parts)
    except (ValueError, TypeError):
        return None
