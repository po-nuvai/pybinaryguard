"""CUDA / GPU compatibility rules.

These rules verify that the GPU driver, CUDA runtime, cuDNN, and GPU
compute capability are compatible with installed GPU-accelerated packages
such as PyTorch and TensorFlow.

Compatibility data is loaded from JSON files shipped in the
``pybinaryguard/rules/data/`` directory.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_data_cache: Dict[str, Any] = {}


def _load_json(filename: str) -> Any:
    """Load and cache a JSON file from the ``rules/data/`` directory.

    Args:
        filename: Name of the JSON file (e.g. ``'cuda_compat_matrix.json'``).

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if filename in _data_cache:
        return _data_cache[filename]
    filepath = os.path.join(_DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    _data_cache[filename] = data
    return data


def _is_cuda_package(pkg: PackageBinaryInfo) -> bool:
    """Heuristic: does this package look like a GPU-accelerated build?

    Checks for a ``+cuXXX`` suffix in the version string or the
    presence of ``libcudart`` in required libraries.
    """
    if "+cu" in pkg.package_version:
        return True
    for lib in pkg.required_libraries:
        if "libcudart" in lib:
            return True
    return False


def _extract_cuda_version_from_version(version: str) -> Optional[Tuple[int, int]]:
    """Extract CUDA version from a package version string like ``2.1.0+cu118``.

    Returns ``(11, 8)`` for ``+cu118``, ``(12, 1)`` for ``+cu121``, etc.
    """
    match = re.search(r"\+cu(\d{2,3})", version)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]))
    # 3 digits: e.g. "118" -> (11, 8), "121" -> (12, 1)
    return (int(digits[:2]), int(digits[2:]))


def _driver_major(driver_version: str) -> Optional[int]:
    """Extract the major version from a driver string like ``'535.104.05'``."""
    parts = driver_version.split(".")
    if not parts:
        return None
    try:
        return int(parts[0])
    except (ValueError, TypeError):
        return None


def _max_cuda_for_driver(driver_major: int) -> Optional[Tuple[int, int]]:
    """Look up the maximum CUDA version supported by a given driver major.

    Returns ``None`` if the driver is not in the compatibility matrix.
    The lookup finds the highest driver key that is <= ``driver_major``.
    """
    data = _load_json("cuda_compat_matrix.json")
    mapping: Dict[str, list] = data.get("driver_to_max_cuda", {})
    best: Optional[Tuple[int, int]] = None
    best_key = -1
    for key_str, cuda_ver in mapping.items():
        try:
            key_int = int(key_str)
        except (ValueError, TypeError):
            continue
        if key_int <= driver_major and key_int > best_key:
            best_key = key_int
            best = (int(cuda_ver[0]), int(cuda_ver[1]))
    return best


def _fmt_ver(version: Tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _fmt_ver3(version: Tuple[int, int, int]) -> str:
    return f"{version[0]}.{version[1]}.{version[2]}"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class CUDADriverTooOldRule(Rule):
    """Detects when the GPU driver does not support the installed CUDA runtime.

    NVIDIA GPU drivers have a maximum CUDA version they can support.
    If the installed CUDA runtime is newer than the driver supports, GPU
    operations will fail with ``CUDA error: no kernel image is available``.
    """

    rule_id = "CUDA_DRIVER_TOO_OLD"
    description = (
        "Check that the GPU driver supports the installed CUDA runtime "
        "version."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return (
            profile.gpu_available
            and profile.gpu_driver_version is not None
            and profile.cuda_runtime_version is not None
        )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        if profile.gpu_driver_version is None or profile.cuda_runtime_version is None:
            return findings

        drv_major = _driver_major(profile.gpu_driver_version)
        if drv_major is None:
            return findings

        max_cuda = _max_cuda_for_driver(drv_major)
        if max_cuda is None:
            return findings

        cuda_rt = profile.cuda_runtime_version
        if cuda_rt > max_cuda:
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.CRITICAL,
                    title="GPU driver is too old for the installed CUDA runtime",
                    explanation=(
                        f"Your NVIDIA GPU driver ({profile.gpu_driver_version}) "
                        f"supports CUDA up to {_fmt_ver(max_cuda)}, but your "
                        f"installed CUDA runtime is {_fmt_ver(cuda_rt)}.  "
                        f"This means GPU operations will fail because the "
                        f"driver cannot execute CUDA {_fmt_ver(cuda_rt)} "
                        f"programs."
                    ),
                    technical_detail=(
                        f"Driver: {profile.gpu_driver_version} "
                        f"(major {drv_major}), "
                        f"max CUDA: {_fmt_ver(max_cuda)}, "
                        f"CUDA runtime: {_fmt_ver(cuda_rt)}"
                    ),
                    suggestion=(
                        f"Option 1 -- upgrade your NVIDIA driver to one "
                        f"that supports CUDA {_fmt_ver(cuda_rt)}:\n"
                        f"  sudo apt install nvidia-driver-XXX  # Debian/Ubuntu\n"
                        f"  sudo dnf install nvidia-driver       # Fedora/RHEL\n\n"
                        f"Option 2 -- downgrade CUDA to "
                        f"{_fmt_ver(max_cuda)} and reinstall GPU packages."
                    ),
                )
            )
        return findings


class CUDARuntimeMismatchRule(Rule):
    """Detects CUDA major version mismatch between framework and runtime.

    GPU-accelerated packages like PyTorch and TensorFlow are compiled
    against a specific CUDA version.  The major version must match the
    system CUDA runtime; a CUDA 11.x package will not work with a
    CUDA 12.x runtime.
    """

    rule_id = "CUDA_RUNTIME_MISMATCH"
    description = (
        "Check that GPU packages' CUDA build version major matches the "
        "system CUDA runtime major version."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.cuda_runtime_version is not None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        cuda_rt = profile.cuda_runtime_version
        if cuda_rt is None:
            return findings

        for pkg in packages:
            pkg_cuda = pkg.cuda_build_version
            if pkg_cuda is None:
                # Try to extract from version string.
                pkg_cuda = _extract_cuda_version_from_version(
                    pkg.package_version
                )
            if pkg_cuda is None:
                continue
            if pkg_cuda[0] != cuda_rt[0]:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{pkg.package_name} CUDA major version mismatch"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} was built for "
                            f"CUDA {_fmt_ver(pkg_cuda)} but your system "
                            f"has CUDA {_fmt_ver(cuda_rt)}.  Because the "
                            f"CUDA major version differs ({pkg_cuda[0]} vs "
                            f"{cuda_rt[0]}), the compiled GPU kernels are "
                            f"incompatible and will fail to load."
                        ),
                        technical_detail=(
                            f"Package CUDA: {_fmt_ver(pkg_cuda)}, "
                            f"System CUDA: {_fmt_ver(cuda_rt)}"
                        ),
                        suggestion=(
                            f"Install the correct variant for your CUDA "
                            f"version:\n"
                            f"  pip install {pkg.package_name}=="
                            f"{_strip_cuda_suffix(pkg.package_version)}"
                            f"+cu{cuda_rt[0]}{cuda_rt[1]}\n\n"
                            f"Or see the package's installation page for "
                            f"CUDA {_fmt_ver(cuda_rt)} wheels."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings


class CUDAMinorMismatchRule(Rule):
    """Warns when CUDA minor versions differ between package and runtime.

    CUDA minor version mismatches are not always fatal (NVIDIA provides
    some forward compatibility), but they can cause subtle issues or
    missing features.
    """

    rule_id = "CUDA_MINOR_MISMATCH"
    description = (
        "Warn when CUDA minor versions differ between a GPU package "
        "and the system runtime."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.cuda_runtime_version is not None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        cuda_rt = profile.cuda_runtime_version
        if cuda_rt is None:
            return findings

        for pkg in packages:
            pkg_cuda = pkg.cuda_build_version
            if pkg_cuda is None:
                pkg_cuda = _extract_cuda_version_from_version(
                    pkg.package_version
                )
            if pkg_cuda is None:
                continue
            # Only warn if major matches but minor differs.
            if pkg_cuda[0] == cuda_rt[0] and pkg_cuda[1] != cuda_rt[1]:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=(
                            f"{pkg.package_name} CUDA minor version differs"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} was built for "
                            f"CUDA {_fmt_ver(pkg_cuda)} but your system "
                            f"has CUDA {_fmt_ver(cuda_rt)}.  While CUDA "
                            f"minor versions are sometimes compatible, "
                            f"there may be missing features or subtle "
                            f"runtime differences."
                        ),
                        technical_detail=(
                            f"Package CUDA: {_fmt_ver(pkg_cuda)}, "
                            f"System CUDA: {_fmt_ver(cuda_rt)}"
                        ),
                        suggestion=(
                            f"For best compatibility, install the wheel "
                            f"that matches your CUDA version:\n"
                            f"  pip install {pkg.package_name}=="
                            f"{_strip_cuda_suffix(pkg.package_version)}"
                            f"+cu{cuda_rt[0]}{cuda_rt[1]}"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings


class CUDNNVersionMismatchRule(Rule):
    """Detects cuDNN version incompatibilities with frameworks.

    TensorFlow and other frameworks are compiled against a specific
    cuDNN version.  A mismatch (especially a major version mismatch)
    leads to errors like ``Could not load library 'libcudnn.so.8'``.
    """

    rule_id = "CUDNN_VERSION_MISMATCH"
    description = (
        "Check that the system cuDNN version is compatible with GPU "
        "framework requirements."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.cudnn_version is not None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        if profile.cudnn_version is None:
            return findings

        sys_cudnn = profile.cudnn_version  # (major, minor, patch)

        # Check TensorFlow requirements from the compatibility matrix.
        tf_matrix = _try_load_json("tensorflow_cuda_matrix.json")
        for pkg in packages:
            pkg_lower = pkg.package_name.lower()
            if pkg_lower not in ("tensorflow", "tensorflow-gpu"):
                continue
            if tf_matrix is None:
                continue
            # Find the closest matching TF version in the matrix.
            tf_info = _find_framework_entry(
                tf_matrix, pkg.package_version
            )
            if tf_info is None:
                continue
            required_cudnn_str = tf_info.get("cudnn")
            if required_cudnn_str is None:
                continue
            req_parts = required_cudnn_str.split(".")
            if len(req_parts) < 2:
                continue
            req_major = int(req_parts[0])
            req_minor = int(req_parts[1])
            if sys_cudnn[0] != req_major:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{pkg.package_name} needs cuDNN "
                            f"{required_cudnn_str}"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} was built for cuDNN "
                            f"{required_cudnn_str} but your system has "
                            f"cuDNN {_fmt_ver3(sys_cudnn)}.  The cuDNN "
                            f"major version must match; otherwise the "
                            f"framework will fail to load its GPU kernels."
                        ),
                        technical_detail=(
                            f"Required cuDNN: {required_cudnn_str}, "
                            f"System cuDNN: {_fmt_ver3(sys_cudnn)}"
                        ),
                        suggestion=(
                            f"Install the correct cuDNN version:\n"
                            f"  sudo apt install libcudnn{req_major}  "
                            f"# Debian/Ubuntu\n\n"
                            f"Or use conda:\n"
                            f"  conda install -c conda-forge cudnn="
                            f"{required_cudnn_str}"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
            elif sys_cudnn[1] < req_minor:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=(
                            f"{pkg.package_name} prefers cuDNN "
                            f"{required_cudnn_str}"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} was built for cuDNN "
                            f"{required_cudnn_str} but your system has "
                            f"cuDNN {_fmt_ver3(sys_cudnn)}.  While the "
                            f"major version matches, the minor version is "
                            f"older than expected and some features may "
                            f"not work correctly."
                        ),
                        technical_detail=(
                            f"Required cuDNN: {required_cudnn_str}, "
                            f"System cuDNN: {_fmt_ver3(sys_cudnn)}"
                        ),
                        suggestion=(
                            f"Upgrade cuDNN to {required_cudnn_str} or "
                            f"later for best compatibility."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings


class CUDANotFoundRule(Rule):
    """Warns when a GPU package is installed but no CUDA runtime is found.

    If a package was built with GPU support but the system has no CUDA
    runtime installed, GPU operations will fail silently (falling back to
    CPU) or raise an error.
    """

    rule_id = "CUDA_NOT_FOUND"
    description = (
        "Warn when a GPU-enabled package is installed but CUDA is not "
        "available."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.cuda_runtime_version is None

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            if not _is_cuda_package(pkg):
                continue
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.WARNING,
                    title=(
                        f"{pkg.package_name} is GPU-enabled but CUDA "
                        f"is not found"
                    ),
                    explanation=(
                        f"Package {pkg.package_name} "
                        f"{pkg.package_version} appears to be built with "
                        f"GPU support (it references CUDA libraries), but "
                        f"no CUDA runtime was detected on your system.  "
                        f"GPU acceleration will not work; the package may "
                        f"fall back to CPU mode or raise an error."
                    ),
                    technical_detail=(
                        f"Package version: {pkg.package_version}, "
                        f"CUDA detected: False"
                    ),
                    suggestion=(
                        f"If you have an NVIDIA GPU, install the CUDA "
                        f"toolkit:\n"
                        f"  sudo apt install nvidia-cuda-toolkit  "
                        f"# Debian/Ubuntu\n\n"
                        f"If you do not need GPU support, install the "
                        f"CPU-only version:\n"
                        f"  pip install {pkg.package_name}-cpu  "
                        f"# if available\n"
                        f"  pip install {pkg.package_name}  "
                        f"# without +cuXXX suffix"
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                )
            )
        return findings


class ComputeCapabilityLowRule(Rule):
    """Detects when the GPU's compute capability is below a framework's minimum.

    Newer versions of TensorFlow and PyTorch drop support for older GPU
    architectures.  For example, PyTorch 2.x requires compute capability
    3.7+.  Attempting to run on an older GPU produces silent wrong
    results or a ``CUDA error: no kernel image``.
    """

    rule_id = "COMPUTE_CAPABILITY_LOW"
    description = (
        "Check that the GPU compute capability meets the framework's "
        "minimum requirement."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return (
            profile.gpu_available
            and profile.gpu_compute_capability is not None
        )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        cc = profile.gpu_compute_capability
        if cc is None:
            return findings

        for pkg in packages:
            min_cc = _get_min_compute(pkg)
            if min_cc is None:
                continue
            if cc < min_cc:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"GPU compute capability too low for "
                            f"{pkg.package_name}"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} requires a GPU with "
                            f"compute capability >= {min_cc[0]}.{min_cc[1]}, "
                            f"but your GPU "
                            f"({profile.gpu_name or 'unknown'}) has "
                            f"compute capability {cc[0]}.{cc[1]}.  The "
                            f"package does not include compiled GPU kernels "
                            f"for your architecture, so GPU operations "
                            f"will fail."
                        ),
                        technical_detail=(
                            f"GPU: {profile.gpu_name or 'unknown'}, "
                            f"compute capability: {cc[0]}.{cc[1]}, "
                            f"minimum required: {min_cc[0]}.{min_cc[1]}"
                        ),
                        suggestion=(
                            f"Option 1 -- use an older version of "
                            f"{pkg.package_name} that still supports "
                            f"compute capability {cc[0]}.{cc[1]}.\n\n"
                            f"Option 2 -- upgrade to a newer NVIDIA GPU "
                            f"with compute capability >= "
                            f"{min_cc[0]}.{min_cc[1]}."
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings


class CUDALibMissingRule(Rule):
    """Detects missing CUDA shared libraries.

    If a package links against CUDA libraries (``libcudart``, ``libcublas``,
    etc.) that are not found on the system, the package will fail at
    import time.
    """

    rule_id = "CUDA_LIB_MISSING"
    description = (
        "Check that all required CUDA shared libraries are present."
    )

    _CUDA_LIB_PREFIXES = (
        "libcuda",
        "libcudart",
        "libcublas",
        "libcufft",
        "libcurand",
        "libcusolver",
        "libcusparse",
        "libnccl",
        "libnvrtc",
        "libnvjitlink",
        "libcudnn",
        "libnvToolsExt",
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            if not pkg.missing_libraries:
                continue
            cuda_missing = [
                lib
                for lib in sorted(pkg.missing_libraries)
                if any(lib.startswith(p) for p in self._CUDA_LIB_PREFIXES)
            ]
            if not cuda_missing:
                continue
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.CRITICAL,
                    title=(
                        f"{pkg.package_name} is missing CUDA libraries"
                    ),
                    explanation=(
                        f"Package {pkg.package_name} "
                        f"{pkg.package_version} requires the following "
                        f"CUDA shared libraries that were not found on "
                        f"your system: {', '.join(cuda_missing)}.  "
                        f"Without these libraries the package will fail "
                        f"to import."
                    ),
                    technical_detail=(
                        f"Missing CUDA libs: {', '.join(cuda_missing)}"
                    ),
                    suggestion=(
                        f"Install the CUDA toolkit and cuDNN:\n"
                        f"  sudo apt install nvidia-cuda-toolkit "
                        f"libcudnn8  # Debian/Ubuntu\n\n"
                        f"Or ensure LD_LIBRARY_PATH includes CUDA lib "
                        f"directories:\n"
                        f"  export LD_LIBRARY_PATH=/usr/local/cuda/lib64"
                        f":$LD_LIBRARY_PATH"
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                )
            )
        return findings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_load_json(filename: str) -> Optional[Dict[str, Any]]:
    """Load a JSON data file, returning ``None`` on any error."""
    try:
        return _load_json(filename)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _find_framework_entry(
    matrix: Dict[str, Any],
    version: str,
) -> Optional[Dict[str, Any]]:
    """Find the closest matching entry in a framework compatibility matrix.

    Tries an exact match first, then strips the patch version, then picks
    the highest version with the same major.minor.
    """
    # Strip any +cuXXX suffix for lookup.
    clean = _strip_cuda_suffix(version)

    # Exact match.
    if clean in matrix:
        return matrix[clean]

    # Try major.minor.0.
    parts = clean.split(".")
    if len(parts) >= 2:
        base = f"{parts[0]}.{parts[1]}.0"
        if base in matrix:
            return matrix[base]

    # Pick the highest version with the same major.minor.
    if len(parts) >= 2:
        prefix = f"{parts[0]}.{parts[1]}."
        candidates = [k for k in matrix if k.startswith(prefix)]
        if candidates:
            candidates.sort()
            return matrix[candidates[-1]]

    return None


def _strip_cuda_suffix(version: str) -> str:
    """Remove ``+cuXXX`` or similar local version suffix."""
    idx = version.find("+")
    if idx >= 0:
        return version[:idx]
    return version


def _get_min_compute(pkg: PackageBinaryInfo) -> Optional[Tuple[int, int]]:
    """Determine the minimum GPU compute capability for a package.

    Uses the PyTorch and TensorFlow compatibility matrices.
    """
    pkg_lower = pkg.package_name.lower()
    version = _strip_cuda_suffix(pkg.package_version)

    if pkg_lower in ("torch", "pytorch"):
        matrix = _try_load_json("pytorch_cuda_matrix.json")
        if matrix is not None:
            entry = _find_framework_entry(matrix, version)
            if entry is not None:
                mc = entry.get("min_compute")
                if mc is not None and len(mc) == 2:
                    return (int(mc[0]), int(mc[1]))

    if pkg_lower in ("tensorflow", "tensorflow-gpu"):
        matrix = _try_load_json("tensorflow_cuda_matrix.json")
        if matrix is not None:
            entry = _find_framework_entry(matrix, version)
            if entry is not None:
                mc = entry.get("min_compute")
                if mc is not None and len(mc) == 2:
                    return (int(mc[0]), int(mc[1]))

    return None
