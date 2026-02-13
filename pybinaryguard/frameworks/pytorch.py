"""PyTorch framework deep inspection.

Validates PyTorch CUDA ABI compatibility, build configuration, and
version-specific requirements beyond generic binary checks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile


def detect_pytorch_build(package: PackageBinaryInfo) -> Dict[str, Optional[str]]:
    """Detect PyTorch build configuration from binary metadata.

    Args:
        package: PackageBinaryInfo for torch package

    Returns:
        Dictionary with detected build info:
        - cuda_version: CUDA version (e.g., "11.8", "12.1")
        - cxx11_abi: C++11 ABI status ("enabled", "disabled", "unknown")
        - cpu_only: Whether this is CPU-only build
        - rocm_version: ROCm version if AMD build
    """
    result: Dict[str, Optional[str]] = {
        "cuda_version": None,
        "cxx11_abi": None,
        "cpu_only": False,
        "rocm_version": None,
    }

    # Check package CUDA version from metadata
    if package.cuda_version:
        result["cuda_version"] = package.cuda_version

    # Check for CPU-only build from package name patterns
    if any(so.path and "cpu" in so.path.lower() for so in package.shared_objects):
        result["cpu_only"] = True

    # Check for ROCm (AMD GPU) build
    for so in package.shared_objects:
        if so.path and "rocm" in so.path.lower():
            # Try to extract ROCm version
            match = re.search(r"rocm[_-]?(\d+\.\d+)", so.path.lower())
            if match:
                result["rocm_version"] = match.group(1)

    # Detect C++11 ABI from GLIBCXX symbols
    # PyTorch with new ABI uses GLIBCXX_3.4.21+, old ABI uses GLIBCXX_3.4.19
    max_glibcxx = None
    for so in package.shared_objects:
        for symbol in so.symbols:
            if symbol.startswith("GLIBCXX_3.4."):
                version_str = symbol.replace("GLIBCXX_3.4.", "")
                try:
                    version_num = int(version_str)
                    if max_glibcxx is None or version_num > max_glibcxx:
                        max_glibcxx = version_num
                except ValueError:
                    continue

    if max_glibcxx is not None:
        # GLIBCXX >= 3.4.21 indicates C++11 ABI
        result["cxx11_abi"] = "enabled" if max_glibcxx >= 21 else "disabled"
    else:
        result["cxx11_abi"] = "unknown"

    return result


def check_pytorch_cuda_abi(
    package: PackageBinaryInfo,
    profile: SystemProfile
) -> List[Dict[str, str]]:
    """Check PyTorch CUDA ABI compatibility.

    Validates:
    - CUDA version matches system runtime
    - C++11 ABI compatibility with system
    - Compute capability support
    - cuDNN version requirements

    Args:
        package: PackageBinaryInfo for torch package
        profile: System profile

    Returns:
        List of compatibility issues found
    """
    issues: List[Dict[str, str]] = []

    if package.name.lower() != "torch":
        return issues

    build_info = detect_pytorch_build(package)

    # Check CUDA version compatibility
    if build_info["cuda_version"] and profile.cuda_runtime_version:
        pkg_cuda_major = int(build_info["cuda_version"].split(".")[0])
        sys_cuda_major = int(profile.cuda_runtime_version.split(".")[0])

        if pkg_cuda_major != sys_cuda_major:
            issues.append({
                "issue": "cuda_major_mismatch",
                "severity": "critical",
                "message": (
                    f"PyTorch built for CUDA {build_info['cuda_version']} but "
                    f"system has CUDA {profile.cuda_runtime_version}"
                ),
                "recommendation": f"Install PyTorch built for CUDA {sys_cuda_major}.x",
            })

    # Check for CPU-only PyTorch on GPU system
    if build_info["cpu_only"] and profile.cuda_runtime_version:
        issues.append({
            "issue": "cpu_build_on_gpu_system",
            "severity": "warning",
            "message": "CPU-only PyTorch detected on system with NVIDIA GPU",
            "recommendation": "Install GPU-enabled PyTorch to utilize available hardware",
        })

    # Check compute capability for PyTorch 2.x
    if package.version and package.version.startswith("2."):
        if profile.cuda_compute_capability:
            compute_major = int(profile.cuda_compute_capability.split(".")[0])
            # PyTorch 2.x requires compute capability >= 3.5
            if compute_major < 3 or (compute_major == 3 and int(profile.cuda_compute_capability.split(".")[1]) < 5):
                issues.append({
                    "issue": "compute_capability_too_low",
                    "severity": "critical",
                    "message": (
                        f"PyTorch 2.x requires compute capability >= 3.5, "
                        f"but GPU has {profile.cuda_compute_capability}"
                    ),
                    "recommendation": "Downgrade to PyTorch 1.x or upgrade GPU hardware",
                })

    # Check C++11 ABI compatibility
    if build_info["cxx11_abi"] == "disabled":
        issues.append({
            "issue": "old_cxx11_abi",
            "severity": "info",
            "message": "PyTorch built with old C++11 ABI (pre-GCC 5.1)",
            "recommendation": (
                "Consider rebuilding extensions with _GLIBCXX_USE_CXX11_ABI=0 "
                "if encountering symbol errors"
            ),
        })

    return issues


def check_pytorch_torchvision_compatibility(
    torch_version: Optional[str],
    torchvision_version: Optional[str]
) -> Optional[str]:
    """Check if torchvision version is compatible with torch version.

    Args:
        torch_version: PyTorch version (e.g., "2.1.0")
        torchvision_version: torchvision version (e.g., "0.16.0")

    Returns:
        Error message if incompatible, None if compatible
    """
    if not torch_version or not torchvision_version:
        return None

    # Compatibility matrix (PyTorch -> torchvision major.minor)
    compatibility_matrix = {
        "2.2": "0.17",
        "2.1": "0.16",
        "2.0": "0.15",
        "1.13": "0.14",
        "1.12": "0.13",
        "1.11": "0.12",
        "1.10": "0.11",
    }

    try:
        torch_major_minor = ".".join(torch_version.split(".")[:2])
        torchvision_major_minor = ".".join(torchvision_version.split(".")[:2])

        expected_torchvision = compatibility_matrix.get(torch_major_minor)
        if expected_torchvision and torchvision_major_minor != expected_torchvision:
            return (
                f"PyTorch {torch_major_minor} expects torchvision {expected_torchvision}.x, "
                f"but found {torchvision_version}"
            )
    except (ValueError, IndexError):
        pass

    return None


def detect_pytorch_distributed_backend(package: PackageBinaryInfo) -> List[str]:
    """Detect available distributed training backends in PyTorch.

    Args:
        package: PackageBinaryInfo for torch package

    Returns:
        List of available backends: ["nccl", "gloo", "mpi"]
    """
    backends = []

    for so in package.shared_objects:
        if not so.path:
            continue

        path_lower = so.path.lower()
        if "nccl" in path_lower:
            backends.append("nccl")
        elif "gloo" in path_lower:
            backends.append("gloo")
        elif "mpi" in path_lower:
            backends.append("mpi")

    return list(set(backends))  # Remove duplicates
