"""ONNX Runtime framework deep inspection.

Validates ONNX Runtime execution providers, version compatibility,
and hardware acceleration support.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile


def check_onnx_runtime_providers(
    package: PackageBinaryInfo,
    profile: SystemProfile
) -> List[Dict[str, str]]:
    """Check ONNX Runtime execution providers and compatibility.

    Validates:
    - Available execution providers (CUDA, TensorRT, OpenVINO, etc.)
    - CUDA provider compatibility with system
    - Missing hardware-specific providers

    Args:
        package: PackageBinaryInfo for onnxruntime package
        profile: System profile

    Returns:
        List of compatibility issues and recommendations
    """
    issues: List[Dict[str, str]] = []

    if not package.name.lower().startswith("onnxruntime"):
        return issues

    # Detect available execution providers
    providers = detect_execution_providers(package)

    # Check if GPU package on GPU system
    has_gpu = profile.cuda_runtime_version is not None
    has_cuda_provider = "CUDAExecutionProvider" in providers

    if has_gpu and not has_cuda_provider:
        issues.append({
            "issue": "missing_cuda_provider",
            "severity": "warning",
            "message": "NVIDIA GPU detected but CUDA execution provider not available",
            "recommendation": "Install onnxruntime-gpu instead of onnxruntime for GPU acceleration",
        })

    if has_cuda_provider and not has_gpu:
        issues.append({
            "issue": "cuda_provider_no_gpu",
            "severity": "info",
            "message": "CUDA execution provider available but no NVIDIA GPU detected",
            "recommendation": "Install onnxruntime (CPU) to reduce package size",
        })

    # Check CUDA provider version compatibility
    if has_cuda_provider and profile.cuda_runtime_version:
        cuda_compat_issues = check_onnx_cuda_compatibility(
            package.version,
            profile.cuda_runtime_version
        )
        if cuda_compat_issues:
            issues.append({
                "issue": "onnx_cuda_version_mismatch",
                "severity": "warning",
                "message": cuda_compat_issues,
                "recommendation": "Check ONNX Runtime CUDA compatibility matrix",
            })

    # Check for TensorRT provider on Jetson
    if profile.board_name and "jetson" in profile.board_name.lower():
        if "TensorrtExecutionProvider" not in providers:
            issues.append({
                "issue": "missing_tensorrt_provider_jetson",
                "severity": "info",
                "message": "TensorRT execution provider not detected on Jetson device",
                "recommendation": "Build ONNX Runtime with TensorRT support for optimal performance",
            })

    return issues


def detect_execution_providers(package: PackageBinaryInfo) -> Set[str]:
    """Detect available ONNX Runtime execution providers.

    Args:
        package: PackageBinaryInfo for onnxruntime package

    Returns:
        Set of detected execution provider names
    """
    providers: Set[str] = set()

    # Always has CPU provider
    providers.add("CPUExecutionProvider")

    for so in package.shared_objects:
        if not so.path:
            continue

        path_lower = so.path.lower()

        # CUDA provider
        if "cuda" in path_lower:
            providers.add("CUDAExecutionProvider")

        # TensorRT provider
        if "tensorrt" in path_lower:
            providers.add("TensorrtExecutionProvider")

        # OpenVINO provider
        if "openvino" in path_lower:
            providers.add("OpenVINOExecutionProvider")

        # DirectML provider (Windows)
        if "directml" in path_lower:
            providers.add("DmlExecutionProvider")

        # CoreML provider (macOS)
        if "coreml" in path_lower:
            providers.add("CoreMLExecutionProvider")

    return providers


def check_onnx_cuda_compatibility(
    onnx_version: Optional[str],
    cuda_version: Optional[str]
) -> Optional[str]:
    """Check ONNX Runtime GPU version compatibility with CUDA.

    Args:
        onnx_version: ONNX Runtime version (e.g., "1.16.0")
        cuda_version: CUDA version (e.g., "12.1")

    Returns:
        Error message if incompatible, None if compatible
    """
    if not onnx_version or not cuda_version:
        return None

    # Compatibility matrix (ONNX Runtime -> CUDA versions)
    compatibility_matrix = {
        "1.16": ["11.8", "12.2"],
        "1.15": ["11.8", "12.1"],
        "1.14": ["11.6", "11.7", "11.8"],
        "1.13": ["11.6", "11.7"],
        "1.12": ["11.4", "11.6"],
        "1.11": ["11.4"],
    }

    try:
        onnx_major_minor = ".".join(onnx_version.split(".")[:2])
        cuda_major_minor = ".".join(cuda_version.split(".")[:2])

        supported_cuda = compatibility_matrix.get(onnx_major_minor, [])

        if supported_cuda and cuda_major_minor not in supported_cuda:
            return (
                f"ONNX Runtime {onnx_major_minor} supports CUDA {', '.join(supported_cuda)}, "
                f"but system has CUDA {cuda_version}"
            )
    except (ValueError, IndexError):
        pass

    return None


def check_onnx_opset_compatibility(
    model_opset: int,
    onnxruntime_version: Optional[str]
) -> Optional[str]:
    """Check if ONNX model opset is supported by ONNX Runtime version.

    Args:
        model_opset: ONNX model opset version
        onnxruntime_version: ONNX Runtime version

    Returns:
        Error message if incompatible, None if compatible
    """
    if not onnxruntime_version:
        return None

    # Maximum supported opset by ONNX Runtime version
    max_opset_by_version = {
        "1.16": 19,
        "1.15": 18,
        "1.14": 18,
        "1.13": 17,
        "1.12": 16,
        "1.11": 15,
        "1.10": 15,
    }

    try:
        ort_major_minor = ".".join(onnxruntime_version.split(".")[:2])
        max_opset = max_opset_by_version.get(ort_major_minor, 19)

        if model_opset > max_opset:
            return (
                f"Model requires ONNX opset {model_opset}, "
                f"but ONNX Runtime {ort_major_minor} only supports up to opset {max_opset}"
            )
    except (ValueError, IndexError):
        pass

    return None
