"""Architecture compatibility rules.

These rules detect when a package contains shared objects compiled for a
different CPU architecture than the one running on the host.  For example,
an x86_64 ``.so`` file will not load on an ARM system.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Architecture, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


class ArchMismatchRule(Rule):
    """Detects packages containing binaries for the wrong CPU architecture.

    Every ELF shared object records the target architecture in its header.
    If that does not match the host CPU, the dynamic linker will refuse to
    load it, resulting in an ``OSError`` or ``ImportError``.
    """

    rule_id = "ARCH_MISMATCH"
    description = (
        "Check that each package's binary architecture matches the host CPU."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        sys_arch = profile.architecture

        if sys_arch == Architecture.UNKNOWN:
            return findings

        for pkg in packages:
            if pkg.target_architecture is None:
                continue
            if pkg.target_architecture == Architecture.UNKNOWN:
                continue
            if pkg.target_architecture != sys_arch:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{pkg.package_name} is built for "
                            f"{pkg.target_architecture.value}"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} contains binaries "
                            f"compiled for {pkg.target_architecture.value} "
                            f"but your system is {sys_arch.value}.  "
                            f"Binary code is architecture-specific: an "
                            f"x86_64 library cannot run on an ARM CPU and "
                            f"vice versa.  The package will fail to import "
                            f"with an error like 'wrong ELF class' or "
                            f"'cannot open shared object file'."
                        ),
                        technical_detail=(
                            f"Package arch: {pkg.target_architecture.value}, "
                            f"System arch: {sys_arch.value}"
                        ),
                        suggestion=(
                            f"Reinstall the package so pip selects the "
                            f"correct wheel for your architecture:\n"
                            f"  pip install --force-reinstall "
                            f"{pkg.package_name}=={pkg.package_version}\n\n"
                            f"If no pre-built wheel exists for "
                            f"{sys_arch.value}, build from source:\n"
                            f"  pip install --no-binary :all: "
                            f"{pkg.package_name}"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings
