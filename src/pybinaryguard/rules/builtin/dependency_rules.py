"""Dependency conflict detection rules.

Reads installed packages' METADATA to detect version conflicts between
what's actually installed. Unlike pip/Poetry which resolve dependencies
before install, this catches conflicts that exist right now in the
live environment — including conflicts caused by manual pip installs,
conda mixing, or stale venvs.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a PEP 440 version into a comparable tuple."""
    # Strip pre/post/dev suffixes for comparison
    clean = re.split(r"[^0-9.]", version_str.strip())[0]
    parts = []
    for part in clean.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def _version_matches(installed: str, spec: str) -> bool:
    """Check if installed version matches a version specifier.

    Handles ==, >=, <=, !=, >, <, ~=, and bare versions.
    """
    spec = spec.strip()
    if not spec:
        return True

    installed_t = _parse_version(installed)

    # Handle compound specs (>=1.0,<2.0)
    if "," in spec:
        return all(_version_matches(installed, s.strip()) for s in spec.split(","))

    # Parse operator and version
    match = re.match(r"(~=|==|!=|>=|<=|>|<)\s*(.+)", spec)
    if not match:
        # Bare version = equality
        return installed_t == _parse_version(spec)

    op, ver_str = match.group(1), match.group(2)

    # Handle wildcard ==1.0.*
    if ver_str.endswith(".*"):
        prefix = _parse_version(ver_str[:-2])
        if op == "==":
            return installed_t[:len(prefix)] == prefix
        elif op == "!=":
            return installed_t[:len(prefix)] != prefix

    ver_t = _parse_version(ver_str)

    if op == "==":
        return installed_t == ver_t
    elif op == "!=":
        return installed_t != ver_t
    elif op == ">=":
        return installed_t >= ver_t
    elif op == "<=":
        return installed_t <= ver_t
    elif op == ">":
        return installed_t > ver_t
    elif op == "<":
        return installed_t < ver_t
    elif op == "~=":
        # ~=1.4.2 means >=1.4.2,<1.5.0
        return installed_t >= ver_t and installed_t[:len(ver_t) - 1] == ver_t[:len(ver_t) - 1]

    return True


def _read_requires(dist_info_path: str) -> List[str]:
    """Read Requires-Dist from METADATA file."""
    metadata_path = os.path.join(dist_info_path, "METADATA")
    if not os.path.isfile(metadata_path):
        return []

    requires = []
    try:
        with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Requires-Dist:"):
                    req = line[len("Requires-Dist:"):].strip()
                    requires.append(req)
                elif line == "" and requires:
                    # End of headers
                    break
    except OSError:
        pass
    return requires


def _parse_requirement(req_str: str) -> Optional[Tuple[str, str, str]]:
    """Parse a Requires-Dist string into (name, version_spec, extras/markers).

    Returns None if the requirement has markers that exclude current env.
    """
    # Strip inline comments
    req_str = req_str.split("#")[0].strip()

    # Handle markers (e.g., ; python_version >= "3.8")
    marker = ""
    if ";" in req_str:
        req_str, marker = req_str.split(";", 1)
        req_str = req_str.strip()
        marker = marker.strip()

        # Skip requirements with extra markers (optional deps)
        if 'extra ==' in marker or 'extra ==' in marker:
            return None

    # Handle extras (e.g., package[extra])
    extras = ""
    if "[" in req_str:
        base, rest = req_str.split("[", 1)
        extras = rest.split("]")[0]
        req_str = base + rest.split("]")[1] if "]" in rest else base

    # Parse name and version spec
    match = re.match(r"([A-Za-z0-9][\w.-]*)\s*(.*)", req_str.strip())
    if not match:
        return None

    name = match.group(1).strip().lower().replace("-", "_").replace(".", "_")
    version_spec = match.group(2).strip().strip("()")

    return (name, version_spec, marker)


class DependencyConflictRule(Rule):
    """Detect version conflicts between installed packages.

    Reads each package's Requires-Dist metadata and checks whether
    the actually installed version satisfies the requirement. Catches
    conflicts that pip missed, or that were introduced by manual installs.
    """

    rule_id = "DEPENDENCY_VERSION_CONFLICT"
    description = "Detect version conflicts between installed packages"

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        # Build installed package index: normalized_name -> version
        installed: Dict[str, str] = {}
        dist_info_dirs: Dict[str, str] = {}

        for pkg in packages:
            norm_name = pkg.package_name.lower().replace("-", "_").replace(".", "_")
            installed[norm_name] = pkg.package_version
            dist_info_dirs[norm_name] = pkg.install_path

        # Check each package's requirements
        seen_conflicts: Set[str] = set()

        for pkg in packages:
            dist_info = pkg.install_path
            if not dist_info or not os.path.isdir(dist_info):
                continue

            requires = _read_requires(dist_info)

            for req_str in requires:
                parsed = _parse_requirement(req_str)
                if parsed is None:
                    continue

                dep_name, version_spec, marker = parsed

                if dep_name not in installed:
                    continue  # Not installed — a different kind of issue

                if not version_spec:
                    continue  # No version constraint

                actual_version = installed[dep_name]

                if not _version_matches(actual_version, version_spec):
                    conflict_key = f"{pkg.package_name}->{dep_name}"
                    if conflict_key in seen_conflicts:
                        continue
                    seen_conflicts.add(conflict_key)

                    findings.append(Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=f"Version conflict: {pkg.package_name} requires "
                              f"{dep_name} {version_spec}",
                        explanation=(
                            f"{pkg.package_name}=={pkg.package_version} requires "
                            f"{dep_name}{version_spec} but "
                            f"{dep_name}=={actual_version} is installed. "
                            f"This version mismatch may cause ImportError or "
                            f"unexpected behavior at runtime."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                        suggestion=(
                            f"pip install '{dep_name}{version_spec}' "
                            f"or pip install --upgrade {pkg.package_name}"
                        ),
                        confidence=0.95,
                    ))

        return findings


class MissingDependencyRule(Rule):
    """Detect required dependencies that are not installed at all."""

    rule_id = "DEPENDENCY_MISSING"
    description = "Detect required packages that are not installed"

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        # Build installed index
        installed: Set[str] = set()
        for pkg in packages:
            norm = pkg.package_name.lower().replace("-", "_").replace(".", "_")
            installed.add(norm)

        seen: Set[str] = set()

        for pkg in packages:
            dist_info = pkg.install_path
            if not dist_info or not os.path.isdir(dist_info):
                continue

            requires = _read_requires(dist_info)

            for req_str in requires:
                parsed = _parse_requirement(req_str)
                if parsed is None:
                    continue

                dep_name, version_spec, marker = parsed

                if dep_name not in installed and dep_name not in seen:
                    seen.add(dep_name)
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=f"Missing dependency: {pkg.package_name} requires "
                              f"{dep_name}",
                        explanation=(
                            f"{pkg.package_name}=={pkg.package_version} requires "
                            f"{dep_name} but it is not installed. This will cause "
                            f"an ImportError when {pkg.package_name} tries to use it."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                        suggestion=f"pip install {dep_name}",
                        confidence=0.9,
                    ))

        return findings
