"""Environment snapshot verifier.

Verifies current environment against a lockfile and runs compatibility checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pybinaryguard.models.system import SystemProfile
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.snapshot.lockfile import Lockfile
from pybinaryguard.predictor import predict_import_failures


@dataclass
class VerificationIssue:
    """An issue found during verification."""

    severity: str  # "error", "warning", "info"
    category: str  # "system", "package", "binary_hash", "compatibility"
    message: str
    package: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of environment verification."""

    success: bool
    issues: List[VerificationIssue] = field(default_factory=list)
    packages_verified: int = 0
    binaries_verified: int = 0
    hash_mismatches: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


class SnapshotVerifier:
    """Verifies environment against a snapshot lockfile."""

    def __init__(
        self,
        lockfile: Lockfile,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo]
    ):
        """Initialize verifier.

        Args:
            lockfile: Lockfile to verify against
            profile: Current system profile
            packages: Currently installed packages
        """
        self.lockfile = lockfile
        self.profile = profile
        self.packages = packages

        # Build package lookup
        self.package_map = {pkg.name.lower(): pkg for pkg in packages}

    def verify(
        self,
        check_hashes: bool = True,
        run_compatibility_checks: bool = True
    ) -> VerificationResult:
        """Verify current environment against lockfile.

        Args:
            check_hashes: Whether to verify binary hashes
            run_compatibility_checks: Whether to run predictive compatibility checks

        Returns:
            VerificationResult with issues found
        """
        result = VerificationResult(success=True)

        # Verify system configuration
        self._verify_system(result)

        # Verify packages
        self._verify_packages(result, check_hashes)

        # Run compatibility checks
        if run_compatibility_checks:
            self._run_compatibility_checks(result)

        # Determine overall success
        result.success = result.error_count == 0

        return result

    def _verify_system(self, result: VerificationResult) -> None:
        """Verify system configuration matches lockfile."""
        if not self.lockfile.system:
            return

        snapshot_sys = self.lockfile.system

        # Check Python version
        py_ver = self.profile.python_version
        current_py = f"{py_ver[0]}.{py_ver[1]}.{py_ver[2]}" if py_ver else ""

        if snapshot_sys.python_version and current_py != snapshot_sys.python_version:
            result.issues.append(
                VerificationIssue(
                    severity="error",
                    category="system",
                    message="Python version mismatch",
                    expected=snapshot_sys.python_version,
                    actual=current_py,
                )
            )

        # Check GLIBC version
        if snapshot_sys.glibc_version and self.profile.glibc_version:
            if self.profile.glibc_version != snapshot_sys.glibc_version:
                result.issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="system",
                        message="GLIBC version differs from snapshot",
                        expected=snapshot_sys.glibc_version,
                        actual=self.profile.glibc_version,
                    )
                )

        # Check CUDA runtime
        if snapshot_sys.cuda_runtime and self.profile.cuda_runtime_version:
            if self.profile.cuda_runtime_version != snapshot_sys.cuda_runtime:
                result.issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="system",
                        message="CUDA runtime version differs",
                        expected=snapshot_sys.cuda_runtime,
                        actual=self.profile.cuda_runtime_version,
                    )
                )

        # Check detected board
        from pybinaryguard.profiles import match_board_profile
        board_profile = match_board_profile(self.profile)
        current_board = board_profile.display_name if board_profile else None

        if snapshot_sys.detected_board and current_board != snapshot_sys.detected_board:
            result.issues.append(
                VerificationIssue(
                    severity="error",
                    category="system",
                    message="Board mismatch - environment was created for different hardware",
                    expected=snapshot_sys.detected_board,
                    actual=current_board or "Unknown",
                )
            )

    def _verify_packages(self, result: VerificationResult, check_hashes: bool) -> None:
        """Verify packages match lockfile."""
        for pkg_snapshot in self.lockfile.packages:
            result.packages_verified += 1

            # Check if package is installed
            current_pkg = self.package_map.get(pkg_snapshot.name.lower())

            if not current_pkg:
                result.issues.append(
                    VerificationIssue(
                        severity="error",
                        category="package",
                        message=f"Package missing from environment",
                        package=pkg_snapshot.name,
                        expected=pkg_snapshot.version,
                        actual=None,
                    )
                )
                continue

            # Check version
            if current_pkg.version != pkg_snapshot.version:
                result.issues.append(
                    VerificationIssue(
                        severity="error",
                        category="package",
                        message=f"Package version mismatch",
                        package=pkg_snapshot.name,
                        expected=pkg_snapshot.version,
                        actual=current_pkg.version,
                    )
                )

            # Check CUDA version for GPU packages
            if pkg_snapshot.cuda_version and current_pkg.cuda_version:
                if current_pkg.cuda_version != pkg_snapshot.cuda_version:
                    result.issues.append(
                        VerificationIssue(
                            severity="warning",
                            category="package",
                            message=f"CUDA build version differs",
                            package=pkg_snapshot.name,
                            expected=pkg_snapshot.cuda_version,
                            actual=current_pkg.cuda_version,
                        )
                    )

            # Verify binary hashes
            if check_hashes and pkg_snapshot.binary_hashes:
                self._verify_hashes(result, pkg_snapshot, current_pkg)

    def _verify_hashes(
        self,
        result: VerificationResult,
        pkg_snapshot,
        current_pkg: PackageBinaryInfo
    ) -> None:
        """Verify binary file hashes."""
        for rel_path, expected_hash in pkg_snapshot.binary_hashes.items():
            result.binaries_verified += 1

            # Find matching shared object
            full_path = None
            for so in current_pkg.shared_objects:
                if so.path and rel_path in so.path:
                    full_path = so.path
                    break

            if not full_path or not Path(full_path).exists():
                result.issues.append(
                    VerificationIssue(
                        severity="error",
                        category="binary_hash",
                        message=f"Binary file missing or moved",
                        package=current_pkg.name,
                        expected=rel_path,
                    )
                )
                result.hash_mismatches += 1
                continue

            # Compute current hash
            current_hash = self._compute_sha256(full_path)

            if current_hash != expected_hash:
                result.issues.append(
                    VerificationIssue(
                        severity="error",
                        category="binary_hash",
                        message=f"Binary hash mismatch - file may be corrupted or tampered",
                        package=current_pkg.name,
                        expected=expected_hash[:16] + "...",
                        actual=current_hash[:16] + "...",
                    )
                )
                result.hash_mismatches += 1

    def _run_compatibility_checks(self, result: VerificationResult) -> None:
        """Run predictive compatibility checks."""
        for pkg in self.packages:
            if not pkg.has_binaries:
                continue

            # Run predictive failure checks
            failures = predict_import_failures(pkg, self.profile)

            for failure in failures:
                result.issues.append(
                    VerificationIssue(
                        severity="warning",
                        category="compatibility",
                        message=failure.error_message,
                        package=pkg.name,
                    )
                )

    @staticmethod
    def _compute_sha256(file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def verify_snapshot(
    lockfile: Lockfile,
    profile: SystemProfile,
    packages: List[PackageBinaryInfo],
    check_hashes: bool = True,
    run_compatibility_checks: bool = True
) -> VerificationResult:
    """Convenience function to verify environment against lockfile.

    Args:
        lockfile: Lockfile to verify against
        profile: Current system profile
        packages: Currently installed packages
        check_hashes: Whether to verify binary hashes
        run_compatibility_checks: Whether to run compatibility checks

    Returns:
        VerificationResult
    """
    verifier = SnapshotVerifier(lockfile, profile, packages)
    return verifier.verify(check_hashes, run_compatibility_checks)
