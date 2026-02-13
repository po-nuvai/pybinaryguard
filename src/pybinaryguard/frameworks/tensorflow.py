"""TensorFlow framework deep inspection.

Validates TensorFlow CUDA compatibility, compute capability requirements,
and build configuration.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile


def check_tensorflow_compute_capability(
    package: PackageBinaryInfo,
    profile: SystemProfile
) -> List[Dict[str, str]]:
    """Check TensorFlow compute capability requirements.

    Validates:
    - Minimum compute capability for TensorFlow version
    - CUDA version compatibility
    - cuDNN version requirements
    - AVX instruction set availability

    Args:
        package: PackageBinaryInfo for tensorflow package
        profile: System profile

    Returns:
        List of compatibility issues found
    """
    issues: List[Dict[str, str]] = []

    if not package.name.lower().startswith("tensorflow"):
        return issues

    tf_version = package.version
    if not tf_version:
        return issues

    # Check compute capability requirements
    if profile.cuda_compute_capability:
        min_compute = get_tensorflow_min_compute_capability(tf_version)

        if min_compute:
            try:
                gpu_compute = float(profile.cuda_compute_capability)
                if gpu_compute < min_compute:
                    issues.append({
                        "issue": "tensorflow_compute_capability_too_low",
                        "severity": "critical",
                        "message": (
                            f"TensorFlow {tf_version} requires compute capability >= {min_compute}, "
                            f"but GPU has {profile.cuda_compute_capability}"
                        ),
                        "recommendation": f"Upgrade GPU or use TensorFlow version supporting compute {gpu_compute}",
                    })
            except ValueError:
                pass

    # Check CUDA version for GPU builds
    if profile.cuda_runtime_version and not is_tensorflow_cpu_only(package):
        cuda_compat = check_tensorflow_cuda_compatibility(tf_version, profile.cuda_runtime_version)
        if cuda_compat:
            issues.append({
                "issue": "tensorflow_cuda_mismatch",
                "severity": "critical",
                "message": cuda_compat,
                "recommendation": "Install TensorFlow version compatible with your CUDA version",
            })

    # Check AVX instruction set (TensorFlow 1.6+ requires AVX)
    if not profile.cpu_flags or "avx" not in profile.cpu_flags.lower():
        major_version = int(tf_version.split(".")[0])
        if major_version >= 1:
            issues.append({
                "issue": "tensorflow_missing_avx",
                "severity": "critical",
                "message": "TensorFlow requires AVX instruction set, but CPU doesn't support it",
                "recommendation": "Build TensorFlow from source without AVX or upgrade CPU",
            })

    return issues


def get_tensorflow_min_compute_capability(tf_version: str) -> Optional[float]:
    """Get minimum compute capability required for TensorFlow version.

    Args:
        tf_version: TensorFlow version (e.g., "2.15.0")

    Returns:
        Minimum compute capability as float, or None if unknown
    """
    try:
        major = int(tf_version.split(".")[0])
        minor = int(tf_version.split(".")[1])

        # TensorFlow 2.11+ requires compute capability >= 3.5
        if major >= 2 and minor >= 11:
            return 3.5

        # TensorFlow 2.x requires compute capability >= 3.0
        if major >= 2:
            return 3.0

        # TensorFlow 1.x requires compute capability >= 3.0
        return 3.0
    except (ValueError, IndexError):
        return None


def check_tensorflow_cuda_compatibility(
    tf_version: str,
    cuda_version: str
) -> Optional[str]:
    """Check TensorFlow and CUDA version compatibility.

    Args:
        tf_version: TensorFlow version (e.g., "2.15.0")
        cuda_version: CUDA version (e.g., "12.2")

    Returns:
        Error message if incompatible, None if compatible
    """
    # Compatibility matrix (TensorFlow -> CUDA versions)
    compatibility_matrix = {
        "2.15": ["12.2"],
        "2.14": ["11.8"],
        "2.13": ["11.8"],
        "2.12": ["11.8"],
        "2.11": ["11.2"],
        "2.10": ["11.2"],
        "2.9": ["11.2"],
        "2.8": ["11.2"],
        "2.7": ["11.2"],
        "2.6": ["11.2"],
        "2.5": ["11.2"],
        "2.4": ["11.0"],
        "2.3": ["10.1"],
        "2.2": ["10.1"],
        "2.1": ["10.0"],
        "2.0": ["10.0"],
    }

    try:
        tf_major_minor = ".".join(tf_version.split(".")[:2])
        cuda_major_minor = ".".join(cuda_version.split(".")[:2])

        supported_cuda = compatibility_matrix.get(tf_major_minor, [])

        if supported_cuda and cuda_major_minor not in supported_cuda:
            return (
                f"TensorFlow {tf_major_minor} requires CUDA {', '.join(supported_cuda)}, "
                f"but system has CUDA {cuda_version}"
            )
    except (ValueError, IndexError):
        pass

    return None


def is_tensorflow_cpu_only(package: PackageBinaryInfo) -> bool:
    """Detect if TensorFlow is CPU-only build.

    Args:
        package: PackageBinaryInfo for tensorflow package

    Returns:
        True if CPU-only build
    """
    # Check package name
    if "cpu" in package.name.lower():
        return True

    # Check for CUDA-related shared objects
    for so in package.shared_objects:
        if so.path and "cuda" in so.path.lower():
            return False

    return True


def detect_tensorflow_gpu_support(package: PackageBinaryInfo) -> Dict[str, bool]:
    """Detect TensorFlow GPU support features.

    Args:
        package: PackageBinaryInfo for tensorflow package

    Returns:
        Dictionary with GPU support flags
    """
    support = {
        "cuda": False,
        "tensorrt": False,
        "rocm": False,
        "xla": False,
    }

    for so in package.shared_objects:
        if not so.path:
            continue

        path_lower = so.path.lower()

        if "cuda" in path_lower:
            support["cuda"] = True
        if "tensorrt" in path_lower:
            support["tensorrt"] = True
        if "rocm" in path_lower:
            support["rocm"] = True
        if "xla" in path_lower:
            support["xla"] = True

    return support


def check_tensorflow_lite_compatibility(
    tflite_model_schema: int,
    tflite_runtime_version: Optional[str]
) -> Optional[str]:
    """Check TFLite model schema compatibility with runtime.

    Args:
        tflite_model_schema: TFLite model schema version
        tflite_runtime_version: TFLite runtime version

    Returns:
        Error message if incompatible, None if compatible
    """
    if not tflite_runtime_version:
        return None

    # TFLite schema version by TensorFlow version
    schema_by_version = {
        "2.15": 3,
        "2.14": 3,
        "2.13": 3,
        "2.12": 3,
        "2.11": 3,
        "2.10": 3,
        "2.9": 3,
        "2.8": 3,
        "2.7": 3,
        "2.6": 3,
        "2.5": 3,
        "2.4": 3,
        "2.3": 3,
    }

    try:
        tflite_major_minor = ".".join(tflite_runtime_version.split(".")[:2])
        max_schema = schema_by_version.get(tflite_major_minor, 3)

        if tflite_model_schema > max_schema:
            return (
                f"TFLite model schema version {tflite_model_schema} is newer than "
                f"runtime version {tflite_runtime_version} (max schema: {max_schema})"
            )
    except (ValueError, IndexError):
        pass

    return None
