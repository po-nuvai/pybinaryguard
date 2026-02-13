"""Python ABI and version compatibility rules.

These rules detect mismatches between the Python interpreter running on the
host and the Python ABI / version that installed binary packages were built
for.  Such mismatches typically result in ``ImportError`` or segfaults at
import time.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo, WheelTag
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


def _extract_interpreter_version(tag: str) -> Optional[Tuple[int, int]]:
    """Extract (major, minor) from an interpreter tag like ``'cp312'``.

    Returns ``None`` if the tag cannot be parsed.
    """
    match = re.match(r"^(?:cp|pp|ip|jy)(\d)(\d+)$", tag)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def _system_interpreter_tag(profile: SystemProfile) -> str:
    """Derive the expected CPython interpreter tag from the system profile.

    For CPython 3.12 this is ``'cp312'``.
    """
    major, minor = profile.python_version[0], profile.python_version[1]
    return f"cp{major}{minor}"


class PythonABIMismatchRule(Rule):
    """Detects packages built for a different CPython ABI tag.

    A CPython extension module is compiled against a specific C API and
    ABI, identified by a tag like ``cp312``.  Loading an extension built
    for ``cp310`` into a ``cp312`` interpreter may segfault or raise an
    ``ImportError`` because the internal struct layouts differ.
    """

    rule_id = "PYTHON_ABI_MISMATCH"
    description = (
        "Check that each package's wheel ABI tag matches the running "
        "Python interpreter's ABI."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        if not profile.python_abi_tag:
            return findings

        sys_abi = profile.python_abi_tag  # e.g. "cp312"

        for pkg in packages:
            if pkg.is_pure_python or not pkg.wheel_tags:
                continue
            for tag in pkg.wheel_tags:
                # Skip universal tags that work on any interpreter.
                if tag.abi in ("none", "abi3"):
                    continue
                if tag.abi != sys_abi:
                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity=Severity.CRITICAL,
                            title=(
                                f"{pkg.package_name} was built for a "
                                f"different Python ABI"
                            ),
                            explanation=(
                                f"Package {pkg.package_name} "
                                f"{pkg.package_version} was compiled for "
                                f"the '{tag.abi}' ABI but your Python "
                                f"interpreter uses '{sys_abi}'.  Compiled "
                                f"extensions are not interchangeable between "
                                f"Python ABI versions because internal data "
                                f"structures change.  Importing this package "
                                f"will raise an ImportError or cause a "
                                f"segmentation fault."
                            ),
                            technical_detail=(
                                f"Wheel ABI tag: {tag.abi}, "
                                f"System ABI tag: {sys_abi}"
                            ),
                            suggestion=(
                                f"Reinstall the package so pip fetches the "
                                f"correct wheel for your interpreter:\n"
                                f"  pip install --force-reinstall "
                                f"{pkg.package_name}=={pkg.package_version}"
                            ),
                            package=pkg.package_name,
                            package_version=pkg.package_version,
                        )
                    )
                    # One finding per package is enough.
                    break
        return findings


class PythonVersionMismatchRule(Rule):
    """Detects packages whose interpreter tag targets a different Python.

    Even without a strict ABI mismatch, a wheel tagged ``cp310`` that is
    loaded into Python 3.12 is suspect.  This rule checks the interpreter
    field of wheel tags against the running Python version.
    """

    rule_id = "PYTHON_VERSION_MISMATCH"
    description = (
        "Check that each package's wheel interpreter tag matches the "
        "running Python version."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        sys_tag = _system_interpreter_tag(profile)
        sys_ver = (profile.python_version[0], profile.python_version[1])

        for pkg in packages:
            if pkg.is_pure_python or not pkg.wheel_tags:
                continue
            for tag in pkg.wheel_tags:
                # Universal / stable-ABI wheels work everywhere.
                if tag.interpreter in ("py3", "py2.py3"):
                    continue
                if tag.abi == "abi3":
                    continue
                wheel_ver = _extract_interpreter_version(tag.interpreter)
                if wheel_ver is None:
                    continue
                if wheel_ver != sys_ver:
                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity=Severity.CRITICAL,
                            title=(
                                f"{pkg.package_name} targets Python "
                                f"{wheel_ver[0]}.{wheel_ver[1]}"
                            ),
                            explanation=(
                                f"Package {pkg.package_name} "
                                f"{pkg.package_version} was built for "
                                f"Python {wheel_ver[0]}.{wheel_ver[1]} "
                                f"(tag '{tag.interpreter}') but you are "
                                f"running Python {sys_ver[0]}.{sys_ver[1]}.  "
                                f"CPython compiled extensions are version-"
                                f"specific and cannot be shared across "
                                f"different Python minor versions."
                            ),
                            technical_detail=(
                                f"Wheel interpreter tag: {tag.interpreter}, "
                                f"System Python: {sys_tag}"
                            ),
                            suggestion=(
                                f"Reinstall with the correct Python version:\n"
                                f"  pip install --force-reinstall "
                                f"{pkg.package_name}=={pkg.package_version}\n\n"
                                f"If you need a specific Python version, "
                                f"create a virtual environment:\n"
                                f"  python{wheel_ver[0]}.{wheel_ver[1]} -m "
                                f"venv .venv && source .venv/bin/activate"
                            ),
                            package=pkg.package_name,
                            package_version=pkg.package_version,
                        )
                    )
                    break
        return findings


class DebugReleaseMixRule(Rule):
    """Detects mixing debug and release Python builds.

    CPython can be compiled with ``--with-pydebug``, which changes the
    sizes of internal objects (adding ref-count fields, etc.).  Loading a
    release-mode extension into a debug interpreter -- or vice versa --
    will corrupt memory and is likely to segfault.
    """

    rule_id = "DEBUG_RELEASE_MIX"
    description = (
        "Warn when the Python build type (debug/release) does not match "
        "the installed extensions."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        if not profile.python_debug_build:
            # Release builds are the common case; extensions are almost
            # always built in release mode so there is nothing to flag.
            return findings

        for pkg in packages:
            if pkg.is_pure_python or not pkg.wheel_tags:
                continue
            for tag in pkg.wheel_tags:
                # A debug-built Python uses an ABI tag ending with 'd'
                # (e.g. "cp312d").  If the wheel's abi tag does NOT end
                # with 'd', it is a release wheel.
                if tag.abi in ("none", "abi3"):
                    continue
                if not tag.abi.endswith("d"):
                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity=Severity.WARNING,
                            title=(
                                f"{pkg.package_name} is a release build "
                                f"on a debug interpreter"
                            ),
                            explanation=(
                                f"Your Python interpreter was compiled "
                                f"with --with-pydebug (debug build), but "
                                f"{pkg.package_name} {pkg.package_version} "
                                f"is a release-mode extension (ABI tag "
                                f"'{tag.abi}').  Debug and release builds "
                                f"use different internal struct layouts, so "
                                f"mixing them can cause crashes or "
                                f"memory corruption."
                            ),
                            technical_detail=(
                                f"System: debug build, "
                                f"Wheel ABI: {tag.abi} (release)"
                            ),
                            suggestion=(
                                f"Either rebuild the package from source "
                                f"under your debug Python:\n"
                                f"  pip install --no-binary :all: "
                                f"{pkg.package_name}\n\n"
                                f"Or switch to a release build of Python for "
                                f"production workloads."
                            ),
                            package=pkg.package_name,
                            package_version=pkg.package_version,
                        )
                    )
                    break
        return findings
