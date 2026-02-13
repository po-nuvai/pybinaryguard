"""CPU instruction-set rules.

These rules detect packages that require advanced CPU instruction sets
(AVX, AVX2, AVX-512, etc.) which may not be available on the host.  Running
such a package on an unsupported CPU triggers a ``SIGILL``
(illegal instruction) signal, causing the process to crash instantly without
a meaningful Python traceback.
"""

from __future__ import annotations

from typing import Dict, List, Set

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule

# Mapping of well-known packages to the instruction sets they are known
# to require.  An empty set means the package *may* use advanced
# instructions but does not always require them (build-dependent).
KNOWN_INSTRUCTION_REQUIREMENTS: Dict[str, Set[str]] = {
    "tensorflow": {"avx"},
    "tensorflow-cpu": {"avx"},
    "tensorflow-gpu": {"avx"},
    "numpy": set(),  # modern numpy may need AVX2 on some builds
    "faiss-cpu": {"avx2"},
    "faiss-gpu": {"avx2"},
    "pytorch": set(),
    "torch": set(),
    "scipy": set(),
    "scikit-learn": set(),
    "xgboost": set(),
    "lightgbm": set(),
    "onnxruntime": {"avx"},
    "onnxruntime-gpu": {"avx"},
}

_INSTRUCTION_FRIENDLY_NAMES: Dict[str, str] = {
    "avx": "AVX (Advanced Vector Extensions)",
    "avx2": "AVX2 (Advanced Vector Extensions 2)",
    "avx512": "AVX-512",
    "sse42": "SSE 4.2",
    "neon": "ARM NEON",
}


def _system_has_instruction(profile: SystemProfile, instruction: str) -> bool:
    """Check whether the system supports the given instruction set."""
    mapping: Dict[str, bool] = {
        "avx": profile.has_avx,
        "avx2": profile.has_avx2,
        "avx512": profile.has_avx512,
        "sse42": profile.has_sse42,
        "neon": profile.has_neon,
    }
    return mapping.get(instruction, False)


class AVX2RequiredRule(Rule):
    """Detects packages that require AVX2 on a CPU without it.

    AVX2 is commonly used in high-performance numerical libraries to
    speed up vector and matrix operations.  Older or low-power CPUs
    (virtual machines on older hosts, Atom processors, etc.) may lack
    AVX2 support.
    """

    rule_id = "AVX2_REQUIRED"
    description = (
        "Flag packages that require AVX2 when the host CPU lacks it."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Skip if the system already has AVX2."""
        return not profile.has_avx2

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            pkg_lower = pkg.package_name.lower()
            requirements = KNOWN_INSTRUCTION_REQUIREMENTS.get(pkg_lower)
            if requirements is None:
                continue
            if "avx2" not in requirements:
                continue
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.CRITICAL,
                    title=(
                        f"{pkg.package_name} requires AVX2 instructions"
                    ),
                    explanation=(
                        f"Package {pkg.package_name} "
                        f"{pkg.package_version} is known to require "
                        f"AVX2 (Advanced Vector Extensions 2) CPU "
                        f"instructions, but your CPU ({profile.cpu_model or 'unknown'}) "
                        f"does not support AVX2.  When the package tries "
                        f"to execute an AVX2 instruction your process "
                        f"will be killed by an 'Illegal instruction' "
                        f"signal (SIGILL) with no Python traceback."
                    ),
                    technical_detail=(
                        f"CPU model: {profile.cpu_model or 'unknown'}, "
                        f"has_avx2: False"
                    ),
                    suggestion=(
                        f"Option 1 -- look for a build that does not "
                        f"require AVX2 (some packages publish 'noavx' "
                        f"variants).\n\n"
                        f"Option 2 -- use conda-forge, which often provides "
                        f"builds for older CPUs:\n"
                        f"  conda install -c conda-forge "
                        f"{pkg.package_name}\n\n"
                        f"Option 3 -- build from source with AVX2 "
                        f"disabled (if the package supports it)."
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                    confidence=0.9,
                )
            )
        return findings


class IllegalInstructionRiskRule(Rule):
    """General check for packages that may trigger SIGILL.

    Consults the ``KNOWN_INSTRUCTION_REQUIREMENTS`` table to determine
    whether the host CPU provides all instruction sets a package needs.
    """

    rule_id = "ILLEGAL_INSTRUCTION_RISK"
    description = (
        "Detect packages whose instruction-set requirements may not "
        "be met by the host CPU."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            pkg_lower = pkg.package_name.lower()
            requirements = KNOWN_INSTRUCTION_REQUIREMENTS.get(pkg_lower)
            if requirements is None:
                continue
            missing: List[str] = []
            for instr in sorted(requirements):
                if not _system_has_instruction(profile, instr):
                    missing.append(instr)
            if not missing:
                continue
            friendly = ", ".join(
                _INSTRUCTION_FRIENDLY_NAMES.get(m, m.upper()) for m in missing
            )
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.CRITICAL,
                    title=(
                        f"{pkg.package_name} may crash with "
                        f"'Illegal instruction'"
                    ),
                    explanation=(
                        f"Package {pkg.package_name} "
                        f"{pkg.package_version} is known to require "
                        f"the following CPU instruction set(s): "
                        f"{friendly}.  Your CPU "
                        f"({profile.cpu_model or 'unknown'}) does not "
                        f"support {'them' if len(missing) > 1 else 'it'}.  "
                        f"This will cause the process to be killed with "
                        f"a SIGILL (Illegal instruction) signal the "
                        f"moment the package tries to use the missing "
                        f"instruction."
                    ),
                    technical_detail=(
                        f"Missing instructions: {', '.join(missing)}, "
                        f"CPU model: {profile.cpu_model or 'unknown'}, "
                        f"CPU flags: {', '.join(sorted(profile.cpu_flags)) if profile.cpu_flags else 'unknown'}"
                    ),
                    suggestion=(
                        f"Option 1 -- install an alternative build that "
                        f"does not require {friendly}.\n\n"
                        f"Option 2 -- use conda-forge:\n"
                        f"  conda install -c conda-forge "
                        f"{pkg.package_name}\n\n"
                        f"Option 3 -- run inside a VM or container on "
                        f"hardware that supports these instructions."
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                    confidence=0.85,
                )
            )
        return findings
