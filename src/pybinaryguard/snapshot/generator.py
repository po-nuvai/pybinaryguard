"""Environment snapshot generator.

Creates lockfiles with binary hashes, GPU stack info, and system configuration.
"""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pybinaryguard.models.system import SystemProfile
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.snapshot.lockfile import Lockfile, PackageSnapshot, SystemSnapshot
from pybinaryguard.profiles import match_board_profile


class SnapshotGenerator:
    """Generates environment snapshots with binary fingerprints."""

    def __init__(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo]
    ):
        """Initialize snapshot generator.

        Args:
            profile: System profile
            packages: List of installed packages
        """
        self.profile = profile
        self.packages = packages

    def generate(self, include_hashes: bool = True) -> Lockfile:
        """Generate a complete environment snapshot.

        Args:
            include_hashes: Whether to compute binary hashes (slower but more secure)

        Returns:
            Lockfile with complete environment state
        """
        # Build system snapshot
        system_snapshot = self._build_system_snapshot()

        # Build package snapshots
        package_snapshots = []
        for pkg in self.packages:
            pkg_snapshot = self._build_package_snapshot(pkg, include_hashes)
            package_snapshots.append(pkg_snapshot)

        # Sort packages by name for consistency
        package_snapshots.sort(key=lambda p: p.name.lower())

        # Create lockfile
        lockfile = Lockfile(
            version="1.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            system=system_snapshot,
            packages=package_snapshots,
            metadata={
                "generated_by": "pybinaryguard",
                "total_packages": len(package_snapshots),
                "packages_with_binaries": sum(
                    1 for p in package_snapshots if p.binary_hashes
                ),
            },
        )

        return lockfile

    def _build_system_snapshot(self) -> SystemSnapshot:
        """Build system configuration snapshot."""
        # Detect board profile
        board_profile = match_board_profile(self.profile)
        detected_board = board_profile.display_name if board_profile else None

        # Get CPU flags
        cpu_flags = []
        if self.profile.cpu_flags:
            # Extract key flags
            flags_str = self.profile.cpu_flags.lower()
            for flag in ["avx", "avx2", "avx512", "sse4_2", "neon", "fma"]:
                if flag in flags_str:
                    cpu_flags.append(flag)

        # Build Python version string
        py_ver = self.profile.python_version
        python_version = f"{py_ver[0]}.{py_ver[1]}.{py_ver[2]}" if py_ver else ""

        return SystemSnapshot(
            python_version=python_version,
            platform=platform.system(),
            architecture=self.profile.architecture.value if self.profile.architecture else "",
            glibc_version=self.profile.glibc_version,
            cuda_runtime=self.profile.cuda_runtime_version,
            cuda_driver=self.profile.cuda_driver_version,
            cuda_compute_capability=self.profile.cuda_compute_capability,
            tensorrt_version=self.profile.tensorrt_version,
            cudnn_version=self.profile.cudnn_version,
            detected_board=detected_board,
            cpu_flags=cpu_flags,
            container_runtime=self.profile.container_runtime.value if self.profile.container_runtime else None,
        )

    def _build_package_snapshot(
        self,
        package: PackageBinaryInfo,
        include_hashes: bool
    ) -> PackageSnapshot:
        """Build snapshot for a single package."""
        binary_hashes = {}

        if include_hashes and package.has_binaries:
            for so in package.shared_objects:
                if so.path and Path(so.path).exists():
                    # Compute SHA256 hash
                    hash_value = self._compute_sha256(so.path)
                    # Use relative path as key
                    rel_path = so.path.replace(package.install_path, "").lstrip("/")
                    binary_hashes[rel_path] = hash_value

        # Get wheel tags if available
        wheel_tags = []
        if hasattr(package, 'wheel_tags') and package.wheel_tags:
            for tag in package.wheel_tags:
                wheel_tags.append(f"{tag.python_tag}-{tag.abi_tag}-{tag.platform_tag}")

        return PackageSnapshot(
            name=package.name,
            version=package.version or "unknown",
            cuda_version=package.cuda_version,
            binary_hashes=binary_hashes,
            manylinux_tag=package.manylinux_tag,
            wheel_tags=wheel_tags,
        )

    @staticmethod
    def _compute_sha256(file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def create_snapshot(
    profile: SystemProfile,
    packages: List[PackageBinaryInfo],
    include_hashes: bool = True
) -> Lockfile:
    """Convenience function to create an environment snapshot.

    Args:
        profile: System profile
        packages: List of installed packages
        include_hashes: Whether to compute binary hashes

    Returns:
        Generated lockfile
    """
    generator = SnapshotGenerator(profile, packages)
    return generator.generate(include_hashes)
