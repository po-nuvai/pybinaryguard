"""Jetson platform plugin for PyBinaryGuard.

Activates when ``/etc/nv_tegra_release`` exists, indicating an NVIDIA
Jetson (Tegra) system running Linux for Tegra (L4T).

Provides
--------
- **JetsonProbe** -- Enhanced Jetson detection: L4T version, JetPack
  mapping, and Tegra SoC model.
- **JetPackCUDARule** -- Verifies that the installed CUDA version matches
  the expected version for the detected JetPack release.
- **JetsonX86WheelRule** -- Detects x86-only wheels installed on an ARM
  Jetson board, which will fail at import time.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pybinaryguard.models.enums import Architecture, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.probes.base import ProbeBase
from pybinaryguard.rules.base import Rule

if TYPE_CHECKING:
    from pybinaryguard.plugins.hooks import HookRegistry

logger = logging.getLogger(__name__)

# -- Mapping tables ---------------------------------------------------------

L4T_TO_JETPACK: Dict[str, str] = {
    "32.7.1": "4.6.1",
    "32.7.2": "4.6.2",
    "32.7.3": "4.6.3",
    "32.7.4": "4.6.4",
    "35.1.0": "5.0.2",
    "35.2.1": "5.1",
    "35.3.1": "5.1.1",
    "35.4.1": "5.1.2",
    "35.5.0": "5.1.3",
    "36.2.0": "6.0",
    "36.3.0": "6.0.1",
    "36.4.0": "6.1",
}

JETPACK_CUDA: Dict[str, Tuple[int, int]] = {
    "4.6.1": (10, 2),
    "4.6.2": (10, 2),
    "4.6.3": (10, 2),
    "4.6.4": (10, 2),
    "5.0.2": (11, 4),
    "5.1": (11, 4),
    "5.1.1": (11, 4),
    "5.1.2": (11, 4),
    "5.1.3": (11, 4),
    "6.0": (12, 2),
    "6.0.1": (12, 2),
    "6.1": (12, 6),
}

# Known Tegra SoC chip IDs from /sys/module/tegra_fuse/parameters/tegra_chip_id
_TEGRA_CHIP_NAMES: Dict[str, str] = {
    "0x18": "TX2 (Parker)",
    "0x19": "Xavier (Carmel)",
    "0x21": "Nano/TX1 (Erista)",
    "0x23": "Orin (Ampere)",
}

_TEGRA_RELEASE_PATH = "/etc/nv_tegra_release"
_TEGRA_CHIP_ID_PATH = "/sys/module/tegra_fuse/parameters/tegra_chip_id"


# -- Probe ------------------------------------------------------------------


class JetsonProbe(ProbeBase):
    """Enhanced Jetson/Tegra probe.

    Reads ``/etc/nv_tegra_release`` to determine the L4T version, maps it
    to a JetPack version, and identifies the Tegra SoC model.  All data is
    returned as ``SystemProfile`` field values.

    This probe is read-only and does not modify any system state.
    """

    name = "jetson"

    def is_applicable(self) -> bool:
        """Only run on systems where ``/etc/nv_tegra_release`` exists."""
        return os.path.isfile(_TEGRA_RELEASE_PATH)

    def collect(self) -> Dict[str, Any]:
        """Collect Jetson platform information.

        Returns
        -------
        Dict[str, Any]
            Keys include ``is_embedded_board``, ``board_name``,
            ``jetpack_version``, and -- when determinable --
            ``cuda_toolkit_version``.
        """
        data: Dict[str, Any] = {
            "is_embedded_board": True,
            "architecture": Architecture.AARCH64,
        }

        l4t_version = self._parse_l4t_version()
        tegra_model = self._read_tegra_model()

        board_name = "NVIDIA Jetson"
        if tegra_model:
            board_name = f"NVIDIA Jetson ({tegra_model})"
        data["board_name"] = board_name

        if l4t_version:
            jetpack = L4T_TO_JETPACK.get(l4t_version)
            if jetpack:
                data["jetpack_version"] = jetpack
                expected_cuda = JETPACK_CUDA.get(jetpack)
                if expected_cuda:
                    data["cuda_toolkit_version"] = expected_cuda

        data["gpu_available"] = True

        return data

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _parse_l4t_version() -> Optional[str]:
        """Parse the L4T version string from ``/etc/nv_tegra_release``.

        The first line typically looks like::

            # R35 (release), REVISION: 4.1, ...

        This method extracts ``"35.4.1"`` from that line.
        """
        try:
            with open(_TEGRA_RELEASE_PATH, "r") as fh:
                content = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        # Match "R<major> (release), REVISION: <minor>.<patch>"
        match = re.search(
            r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*(\d+\.\d+)",
            content,
        )
        if match:
            major = match.group(1)
            minor_patch = match.group(2)
            return f"{major}.{minor_patch}"
        return None

    @staticmethod
    def _read_tegra_model() -> Optional[str]:
        """Read the Tegra chip ID and map it to a human-readable name.

        The chip ID is read from
        ``/sys/module/tegra_fuse/parameters/tegra_chip_id``.
        """
        try:
            with open(_TEGRA_CHIP_ID_PATH, "r") as fh:
                raw = fh.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        # Normalise to hex string
        try:
            chip_id = int(raw, 0)
            hex_id = f"0x{chip_id:02x}"
        except ValueError:
            return None

        return _TEGRA_CHIP_NAMES.get(hex_id)


# -- Rules ------------------------------------------------------------------


class JetPackCUDARule(Rule):
    """Verify that installed CUDA matches the expected JetPack CUDA version.

    On a Jetson board, the JetPack release pins a specific CUDA toolkit
    version.  If the detected CUDA version does not match, libraries
    compiled against the expected CUDA may crash or produce incorrect
    results.
    """

    rule_id = "JETPACK_CUDA_MISMATCH"
    description = (
        "Check that the installed CUDA version matches the expected "
        "version for the detected JetPack release."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applies on Jetson boards with a known JetPack version."""
        return (
            profile.is_embedded_board
            and profile.jetpack_version is not None
            and profile.jetpack_version in JETPACK_CUDA
        )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Compare detected CUDA with JetPack-expected CUDA."""
        findings: List[Finding] = []

        if profile.jetpack_version is None:
            return findings

        expected_cuda = JETPACK_CUDA.get(profile.jetpack_version)
        if expected_cuda is None:
            return findings

        actual_cuda = profile.cuda_runtime_version or profile.cuda_toolkit_version
        if actual_cuda is None:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title="CUDA not detected on Jetson board",
                explanation=(
                    f"JetPack {profile.jetpack_version} expects CUDA "
                    f"{expected_cuda[0]}.{expected_cuda[1]}, but no CUDA "
                    f"installation was detected on this system."
                ),
                suggestion=(
                    "Install the CUDA toolkit that matches your JetPack "
                    "version, or ensure that nvidia-smi / nvcc is accessible."
                ),
            ))
            return findings

        if actual_cuda[0] != expected_cuda[0] or actual_cuda[1] != expected_cuda[1]:
            severity = Severity.CRITICAL if actual_cuda[0] != expected_cuda[0] else Severity.WARNING
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=severity,
                title="CUDA version does not match JetPack expectation",
                explanation=(
                    f"JetPack {profile.jetpack_version} expects CUDA "
                    f"{expected_cuda[0]}.{expected_cuda[1]}, but CUDA "
                    f"{actual_cuda[0]}.{actual_cuda[1]} was detected."
                ),
                technical_detail=(
                    f"Expected CUDA {expected_cuda[0]}.{expected_cuda[1]} "
                    f"(JetPack {profile.jetpack_version}), "
                    f"found {actual_cuda[0]}.{actual_cuda[1]}."
                ),
                suggestion=(
                    "Re-flash the Jetson with the correct JetPack SDK, or "
                    "install the matching CUDA toolkit version. Mixing CUDA "
                    "versions on Jetson is not recommended."
                ),
            ))
        else:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.PASSED,
                title="CUDA version matches JetPack expectation",
                explanation=(
                    f"CUDA {actual_cuda[0]}.{actual_cuda[1]} matches "
                    f"JetPack {profile.jetpack_version} requirements."
                ),
            ))

        return findings


class JetsonX86WheelRule(Rule):
    """Detect x86-only wheels installed on an ARM Jetson board.

    Wheels built for ``manylinux_*_x86_64`` or ``linux_x86_64`` will
    contain shared objects compiled for the wrong architecture and will
    fail with ``ImportError`` or ``OSError`` when loaded on the Jetson's
    AArch64 CPU.
    """

    rule_id = "JETSON_X86_WHEEL"
    description = (
        "Detect x86-compiled wheels installed on an ARM Jetson board."
    )

    _X86_PLATFORM_FRAGMENTS = ("x86_64", "i686", "i386", "amd64")

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applies on Jetson boards (AArch64 embedded)."""
        return (
            profile.is_embedded_board
            and profile.architecture in (Architecture.AARCH64, Architecture.ARMV7L)
        )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Check each package's wheel tags and shared objects for x86 targets."""
        findings: List[Finding] = []

        for pkg in packages:
            if pkg.is_pure_python:
                continue

            # Check wheel tags
            for tag in pkg.wheel_tags:
                platform_lower = tag.platform.lower()
                if any(frag in platform_lower for frag in self._X86_PLATFORM_FRAGMENTS):
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=f"x86 wheel installed on ARM Jetson: {pkg.package_name}",
                        explanation=(
                            f"Package {pkg.package_name}=={pkg.package_version} "
                            f"was installed from an x86 wheel (platform tag: "
                            f"{tag.platform!r}), but this system is {profile.architecture.value}."
                        ),
                        technical_detail=(
                            f"Wheel tag: {tag.interpreter}-{tag.abi}-{tag.platform}"
                        ),
                        suggestion=(
                            f"Reinstall {pkg.package_name} with a wheel built "
                            f"for aarch64, or build from source: "
                            f"pip install --no-binary {pkg.package_name} {pkg.package_name}"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    ))
                    break  # One finding per package is sufficient

            # Also check shared objects for architecture mismatch
            if not findings or findings[-1].package != pkg.package_name:
                for so in pkg.shared_objects:
                    if so.architecture in (Architecture.X86_64, Architecture.I686):
                        findings.append(Finding(
                            rule_id=self.rule_id,
                            severity=Severity.CRITICAL,
                            title=f"x86 binary in package on ARM Jetson: {pkg.package_name}",
                            explanation=(
                                f"Shared object {so.filename} in "
                                f"{pkg.package_name}=={pkg.package_version} is "
                                f"compiled for {so.architecture.value}, but this "
                                f"Jetson runs {profile.architecture.value}."
                            ),
                            technical_detail=f"Binary: {so.path}",
                            suggestion=(
                                f"Reinstall {pkg.package_name} from an ARM-compatible "
                                f"wheel or build from source."
                            ),
                            package=pkg.package_name,
                            package_version=pkg.package_version,
                        ))
                        break  # One finding per package is sufficient

        return findings


# -- Plugin entry point -----------------------------------------------------


def register(registry: HookRegistry) -> None:
    """Register Jetson extensions if running on a Tegra platform.

    This function is called by the plugin loader.  It only activates
    when ``/etc/nv_tegra_release`` exists, indicating an L4T-based
    Jetson system.
    """
    if not os.path.exists(_TEGRA_RELEASE_PATH):
        logger.debug("Jetson plugin: /etc/nv_tegra_release not found; not activating")
        return

    registry.add_probe(JetsonProbe())
    registry.add_rule(JetPackCUDARule())
    registry.add_rule(JetsonX86WheelRule())
    logger.info("Jetson plugin activated")
