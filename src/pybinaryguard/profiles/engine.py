"""Board profile engine for embedded device intelligence.

This module provides the core profile system that loads board-specific
configuration and provides intelligent recommendations based on detected hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pybinaryguard.models.system import SystemProfile


@dataclass
class BrokenWheel:
    """Information about a known broken wheel for a specific board."""

    package: str
    versions: List[str]
    reason: str
    recommendation: str


@dataclass
class ValidatedStack:
    """A validated combination of packages known to work together."""

    name: str
    packages: Dict[str, str]
    build_opencv: bool = False


@dataclass
class BoardProfile:
    """Comprehensive profile for a specific embedded board.

    Contains hardware specifications, compatibility matrices, known issues,
    and validated software stacks for embedded devices.
    """

    board_id: str
    display_name: str
    vendor: str
    architecture: str

    # Detection patterns
    dt_model_patterns: List[str] = field(default_factory=list)
    tegra_version: Optional[str] = None
    cpu_model_patterns: List[str] = field(default_factory=list)
    revision_codes: List[str] = field(default_factory=list)

    # Hardware specs
    cuda_compute_capability: Optional[str] = None
    max_cuda_version: Optional[str] = None
    recommended_jetpack: Optional[str] = None
    recommended_os: Optional[str] = None
    gpu_model: Optional[str] = None
    tpu: Optional[str] = None
    ram_gb: Optional[int] = None
    recommended_glibc: Optional[str] = None
    kernel_version_min: Optional[str] = None

    # Compatibility
    python_versions: List[str] = field(default_factory=list)
    manylinux_tags: List[str] = field(default_factory=list)
    cuda_versions: List[str] = field(default_factory=list)
    tensorrt_versions: List[str] = field(default_factory=list)
    opencv_backends: List[str] = field(default_factory=list)
    tflite_versions: List[str] = field(default_factory=list)
    required_packages: List[str] = field(default_factory=list)

    # Known issues
    broken_wheels: List[BrokenWheel] = field(default_factory=list)
    incompatible_packages: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: Dict[str, Any] = field(default_factory=dict)

    # Validated stacks
    validated_stacks: List[ValidatedStack] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BoardProfile:
        """Create a BoardProfile from a dictionary (loaded from JSON)."""
        # Extract top-level fields
        board_id = data["board_id"]
        display_name = data["display_name"]
        vendor = data["vendor"]
        architecture = data["architecture"]

        # Extract detection patterns
        detection = data.get("detection", {})
        dt_model_patterns = detection.get("dt_model_patterns", [])
        tegra_version = detection.get("tegra_version")
        cpu_model_patterns = detection.get("cpu_model_patterns", [])
        revision_codes = detection.get("revision_codes", [])

        # Extract specs
        specs = data.get("specs", {})
        cuda_compute_capability = specs.get("cuda_compute_capability")
        max_cuda_version = specs.get("max_cuda_version")
        recommended_jetpack = specs.get("recommended_jetpack")
        recommended_os = specs.get("recommended_os")
        gpu_model = specs.get("gpu_model")
        tpu = specs.get("tpu")
        ram_gb = specs.get("ram_gb")
        recommended_glibc = specs.get("recommended_glibc")
        kernel_version_min = specs.get("kernel_version_min")

        # Extract compatibility
        compat = data.get("compatibility", {})
        python_versions = compat.get("python_versions", [])
        manylinux_tags = compat.get("manylinux_tags", [])
        cuda_versions = compat.get("cuda_versions", [])
        tensorrt_versions = compat.get("tensorrt_versions", [])
        opencv_backends = compat.get("opencv_backends", [])
        tflite_versions = compat.get("tflite_versions", [])
        required_packages = compat.get("required_packages", [])

        # Extract known issues
        issues = data.get("known_issues", {})
        broken_wheels_data = issues.get("broken_wheels", [])
        broken_wheels = [
            BrokenWheel(
                package=bw["package"],
                versions=bw["versions"],
                reason=bw["reason"],
                recommendation=bw["recommendation"]
            )
            for bw in broken_wheels_data
        ]
        incompatible_packages = issues.get("incompatible_packages", [])

        # Extract recommendations
        recommendations = data.get("recommendations", {})

        # Extract validated stacks
        stacks_data = data.get("validated_stacks", [])
        validated_stacks = [
            ValidatedStack(
                name=stack["name"],
                packages=stack["packages"],
                build_opencv=stack.get("build_opencv", False)
            )
            for stack in stacks_data
        ]

        return cls(
            board_id=board_id,
            display_name=display_name,
            vendor=vendor,
            architecture=architecture,
            dt_model_patterns=dt_model_patterns,
            tegra_version=tegra_version,
            cpu_model_patterns=cpu_model_patterns,
            revision_codes=revision_codes,
            cuda_compute_capability=cuda_compute_capability,
            max_cuda_version=max_cuda_version,
            recommended_jetpack=recommended_jetpack,
            recommended_os=recommended_os,
            gpu_model=gpu_model,
            tpu=tpu,
            ram_gb=ram_gb,
            recommended_glibc=recommended_glibc,
            kernel_version_min=kernel_version_min,
            python_versions=python_versions,
            manylinux_tags=manylinux_tags,
            cuda_versions=cuda_versions,
            tensorrt_versions=tensorrt_versions,
            opencv_backends=opencv_backends,
            tflite_versions=tflite_versions,
            required_packages=required_packages,
            broken_wheels=broken_wheels,
            incompatible_packages=incompatible_packages,
            recommendations=recommendations,
            validated_stacks=validated_stacks,
        )

    def matches_system(self, profile: SystemProfile) -> bool:
        """Check if this board profile matches the given system profile.

        Uses getattr for board-specific fields that may not be present on
        every SystemProfile (dt_model, tegra_release, rpi_revision are
        populated only by the BoardProbe on embedded hardware).
        """
        # Check device tree model
        dt_model = getattr(profile, "dt_model", None) or ""
        if dt_model:
            for pattern in self.dt_model_patterns:
                if pattern.lower() in dt_model.lower():
                    return True

        # Check Tegra version (Jetson-specific)
        tegra_release = getattr(profile, "tegra_release", None) or ""
        if self.tegra_version and tegra_release:
            if self.tegra_version in tegra_release:
                return True

        # Check CPU model
        if profile.cpu_model:
            for pattern in self.cpu_model_patterns:
                if pattern.lower() in profile.cpu_model.lower():
                    # Additional architecture check
                    if profile.architecture.value == self.architecture:
                        return True

        # Check revision codes (Raspberry Pi-specific)
        rpi_revision = getattr(profile, "rpi_revision", None) or ""
        if rpi_revision:
            if rpi_revision.lower() in [rc.lower() for rc in self.revision_codes]:
                return True

        return False

    def is_package_known_broken(self, package_name: str, version: Optional[str] = None) -> Optional[BrokenWheel]:
        """Check if a package version is known to be broken on this board."""
        for broken in self.broken_wheels:
            if broken.package.lower() == package_name.lower():
                if version is None or "*" in broken.versions:
                    return broken
                # Simple version matching (could be enhanced)
                for ver_pattern in broken.versions:
                    if ver_pattern.startswith(">=") and version:
                        min_ver = ver_pattern[2:]
                        if self._version_gte(version, min_ver):
                            return broken
                    elif ver_pattern.startswith("<") and version:
                        max_ver = ver_pattern[1:]
                        if self._version_lt(version, max_ver):
                            return broken
                    elif version and ver_pattern == version:
                        return broken
        return None

    def is_package_incompatible(self, package_name: str) -> bool:
        """Check if a package is fundamentally incompatible with this board."""
        for pkg in self.incompatible_packages:
            if pkg.endswith("*"):
                prefix = pkg[:-1]
                if package_name.lower().startswith(prefix.lower()):
                    return True
            elif pkg.lower() == package_name.lower():
                return True
        return False

    @staticmethod
    def _version_gte(v1: str, v2: str) -> bool:
        """Compare versions (simple numeric comparison)."""
        try:
            parts1 = [int(p) for p in v1.split(".")[:3]]
            parts2 = [int(p) for p in v2.split(".")[:3]]
            return parts1 >= parts2
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _version_lt(v1: str, v2: str) -> bool:
        """Compare versions (simple numeric comparison)."""
        try:
            parts1 = [int(p) for p in v1.split(".")[:3]]
            parts2 = [int(p) for p in v2.split(".")[:3]]
            return parts1 < parts2
        except (ValueError, IndexError):
            return False


class ProfileEngine:
    """Engine for loading and matching board profiles."""

    def __init__(self, profile_dir: Optional[Path] = None):
        """Initialize the profile engine.

        Args:
            profile_dir: Directory containing profile JSON files.
                        If None, uses the built-in profiles directory.
        """
        if profile_dir is None:
            # Use built-in profiles
            profile_dir = Path(__file__).parent

        self.profile_dir = Path(profile_dir)
        self.profiles: Dict[str, BoardProfile] = {}
        self._load_all_profiles()

    def _load_all_profiles(self) -> None:
        """Load all board profiles from the profile directory."""
        if not self.profile_dir.exists():
            return

        for json_file in self.profile_dir.glob("*.json"):
            try:
                profile = self._load_profile_file(json_file)
                self.profiles[profile.board_id] = profile
            except Exception:
                # Skip invalid profiles silently
                pass

    def _load_profile_file(self, path: Path) -> BoardProfile:
        """Load a single profile from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return BoardProfile.from_dict(data)

    def match_profile(self, system: SystemProfile) -> Optional[BoardProfile]:
        """Find the board profile that matches the given system profile."""
        for profile in self.profiles.values():
            if profile.matches_system(system):
                return profile
        return None

    def get_profile(self, board_id: str) -> Optional[BoardProfile]:
        """Get a specific board profile by ID."""
        return self.profiles.get(board_id)

    def list_profiles(self) -> List[str]:
        """List all available board profile IDs."""
        return list(self.profiles.keys())


# Module-level convenience functions

_engine: Optional[ProfileEngine] = None


def _get_engine() -> ProfileEngine:
    """Get or create the global profile engine."""
    global _engine
    if _engine is None:
        _engine = ProfileEngine()
    return _engine


def load_profile(board_id: str) -> Optional[BoardProfile]:
    """Load a board profile by ID."""
    return _get_engine().get_profile(board_id)


def match_board_profile(system: SystemProfile) -> Optional[BoardProfile]:
    """Match a system profile to a board profile."""
    return _get_engine().match_profile(system)
