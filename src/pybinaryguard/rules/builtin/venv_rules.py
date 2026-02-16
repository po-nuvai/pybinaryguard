"""Virtual environment misconfiguration rules.

Detects common venv issues that cause packages to misbehave:
system/venv package mixing, pip user-site leaks, stale environments.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


class SystemPythonWarningRule(Rule):
    """Warn when running on system Python without a virtual environment."""

    rule_id = "VENV_SYSTEM_PYTHON"
    description = "Detect use of system Python without isolation"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.is_system_python

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        # Only warn if there are a significant number of packages
        if len(packages) > 10:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.INFO,
                title="Running on system Python without a virtual environment",
                explanation=(
                    f"Found {len(packages)} packages installed on system Python. "
                    f"Installing packages system-wide can cause conflicts with "
                    f"OS-managed packages and may break system tools. This is "
                    f"especially risky on Debian/Ubuntu where Python is externally "
                    f"managed (PEP 668)."
                ),
                suggestion=(
                    "Create a virtual environment: python3 -m venv .venv && "
                    "source .venv/bin/activate"
                ),
                confidence=0.7,
            ))

        return findings


class MixedEnvironmentRule(Rule):
    """Detect packages from multiple environments leaking into sys.path."""

    rule_id = "VENV_MIXED_ENVIRONMENT"
    description = "Detect mixed package sources on sys.path"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.mixed_env_risk

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        return [Finding(
            rule_id=self.rule_id,
            severity=Severity.WARNING,
            title="Mixed package environments detected",
            explanation=(
                "Packages from multiple environment roots are on sys.path. "
                "This happens when system packages leak into a virtual environment "
                "or when multiple venvs overlap. Mixed environments cause "
                "unpredictable import behavior and version conflicts."
            ),
            suggestion=(
                "Create a clean virtual environment: python3 -m venv --clear .venv"
            ),
            confidence=0.85,
        )]


class UserSiteLeakRule(Rule):
    """Detect pip user-site packages leaking into a virtual environment."""

    rule_id = "VENV_USER_SITE_LEAK"
    description = "Detect user-site packages leaking into venv"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.is_virtual_env and profile.pip_user_site_enabled

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        return [Finding(
            rule_id=self.rule_id,
            severity=Severity.WARNING,
            title="User site-packages leaking into virtual environment",
            explanation=(
                f"Running in a {profile.venv_type} environment but pip user "
                f"site-packages directory is on sys.path. Packages installed "
                f"with 'pip install --user' outside the venv will shadow "
                f"venv-installed packages, causing version confusion."
            ),
            suggestion=(
                "Set PYTHONNOUSERSITE=1 or create venv with: "
                "python3 -m venv --system-site-packages=off .venv"
            ),
            confidence=0.9,
        )]


class CondaPipMixingRule(Rule):
    """Detect pip-installed packages in a conda environment."""

    rule_id = "VENV_CONDA_PIP_MIXING"
    description = "Detect pip packages in conda environment"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.venv_type == "conda"

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        # In conda envs, packages without wheel tags that have binaries
        # were likely pip-installed (conda packages have different metadata)
        pip_binary_pkgs = [
            pkg for pkg in packages
            if pkg.has_binaries and pkg.wheel_tags
        ]

        if len(pip_binary_pkgs) > 5:
            names = ", ".join(p.package_name for p in pip_binary_pkgs[:5])
            remaining = len(pip_binary_pkgs) - 5
            suffix = f" (+{remaining} more)" if remaining > 0 else ""

            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.INFO,
                title=f"Pip-installed binary packages in conda environment",
                explanation=(
                    f"Found {len(pip_binary_pkgs)} pip-installed package(s) with "
                    f"compiled extensions in a conda environment ({names}{suffix}). "
                    f"Mixing pip and conda can cause library conflicts when both "
                    f"provide different versions of shared libraries (e.g., libstdc++, "
                    f"MKL, OpenSSL)."
                ),
                suggestion=(
                    "Prefer conda install for packages with C extensions: "
                    "conda install <package>. Use pip only for packages not "
                    "available via conda."
                ),
                confidence=0.7,
            ))

        return findings
