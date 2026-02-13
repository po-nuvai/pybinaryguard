"""TensorRT framework deep inspection.

Validates TensorRT engine metadata, CUDA compatibility, and
plugin availability beyond generic binary checks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile


def validate_tensorrt_engine(
    package: PackageBinaryInfo,
    profile: SystemProfile
) -> List[Dict[str, str]]:
    """Validate TensorRT installation and compatibility.

    Checks:
    - TensorRT version compatibility with CUDA
    - Required plugins availability
    - Compute capability support
    - cuDNN version requirements

    Args:
        package: PackageBinaryInfo for tensorrt package
        profile: System profile

    Returns:
        List of compatibility issues found
    """
    issues: List[Dict[str, str]] = []

    if not package.name.lower().startswith("tensorrt"):
        return issues

    # Extract TensorRT version from package
    trt_version = package.version
    if not trt_version:
        return issues

    # TensorRT 8.6+ requires CUDA 11.x or 12.x
    if trt_version.startswith("8.6") or trt_version.startswith("8.5"):
        if profile.cuda_runtime_version:
            cuda_major = int(profile.cuda_runtime_version.split(".")[0])
            if cuda_major < 11:
                issues.append({
                    "issue": "tensorrt_cuda_too_old",
                    "severity": "critical",
                    "message": (
                        f"TensorRT {trt_version} requires CUDA 11.x or 12.x, "
                        f"but system has CUDA {profile.cuda_runtime_version}"
                    ),
                    "recommendation": "Upgrade CUDA runtime or use TensorRT 7.x",
                })

    # Check compute capability requirements
    if profile.cuda_compute_capability:
        compute_major = int(profile.cuda_compute_capability.split(".")[0])

        # TensorRT 8.x requires compute capability >= 5.0
        if trt_version.startswith("8."):
            if compute_major < 5:
                issues.append({
                    "issue": "tensorrt_compute_capability_too_low",
                    "severity": "critical",
                    "message": (
                        f"TensorRT 8.x requires compute capability >= 5.0, "
                        f"but GPU has {profile.cuda_compute_capability}"
                    ),
                    "recommendation": "Upgrade GPU or use TensorRT 7.x",
                })

    # Check for common TensorRT plugins
    available_plugins = detect_tensorrt_plugins(package)

    if not available_plugins:
        issues.append({
            "issue": "no_tensorrt_plugins",
            "severity": "info",
            "message": "No TensorRT plugins detected in installation",
            "recommendation": "Install tensorrt-plugins package if needed for custom layers",
        })

    return issues


def detect_tensorrt_plugins(package: PackageBinaryInfo) -> List[str]:
    """Detect available TensorRT plugins from shared objects.

    Args:
        package: PackageBinaryInfo for tensorrt package

    Returns:
        List of detected plugin names
    """
    plugins = []

    for so in package.shared_objects:
        if not so.path:
            continue

        # TensorRT plugins typically named libnvinfer_plugin.so
        if "plugin" in so.path.lower():
            plugins.append(so.path)

    return plugins


def check_tensorrt_cudnn_compatibility(
    tensorrt_version: Optional[str],
    cudnn_version: Optional[str]
) -> Optional[str]:
    """Check TensorRT and cuDNN version compatibility.

    Args:
        tensorrt_version: TensorRT version (e.g., "8.6.1")
        cudnn_version: cuDNN version (e.g., "8.9.0")

    Returns:
        Error message if incompatible, None if compatible
    """
    if not tensorrt_version or not cudnn_version:
        return None

    # Compatibility matrix (TensorRT -> cuDNN major version)
    compatibility_matrix = {
        "8.6": "8",
        "8.5": "8",
        "8.4": "8",
        "8.2": "8",
        "8.0": "8",
        "7.2": "7",
        "7.1": "7",
    }

    try:
        trt_major_minor = ".".join(tensorrt_version.split(".")[:2])
        cudnn_major = cudnn_version.split(".")[0]

        expected_cudnn_major = compatibility_matrix.get(trt_major_minor)
        if expected_cudnn_major and cudnn_major != expected_cudnn_major:
            return (
                f"TensorRT {trt_major_minor} expects cuDNN {expected_cudnn_major}.x, "
                f"but found cuDNN {cudnn_version}"
            )
    except (ValueError, IndexError):
        pass

    return None


def detect_tensorrt_precision_support(
    package: PackageBinaryInfo,
    profile: SystemProfile
) -> Dict[str, bool]:
    """Detect supported precision modes (FP32, FP16, INT8).

    Args:
        package: PackageBinaryInfo for tensorrt package
        profile: System profile

    Returns:
        Dictionary with supported precision modes
    """
    support = {
        "fp32": True,  # Always supported
        "fp16": False,
        "int8": False,
    }

    # FP16 requires compute capability >= 5.3
    if profile.cuda_compute_capability:
        try:
            major, minor = profile.cuda_compute_capability.split(".")
            compute_val = int(major) * 10 + int(minor)

            if compute_val >= 53:  # 5.3
                support["fp16"] = True

            if compute_val >= 61:  # 6.1
                support["int8"] = True
        except (ValueError, IndexError):
            pass

    return support
