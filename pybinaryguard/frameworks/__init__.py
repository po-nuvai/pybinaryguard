"""AI framework deep inspection layer.

Framework-specific validators that understand internal structure beyond
generic binary inspection.
"""

from __future__ import annotations

from .pytorch import check_pytorch_cuda_abi, detect_pytorch_build
from .tensorrt import validate_tensorrt_engine
from .onnxruntime import check_onnx_runtime_providers
from .tensorflow import check_tensorflow_compute_capability

__all__ = [
    "check_pytorch_cuda_abi",
    "detect_pytorch_build",
    "validate_tensorrt_engine",
    "check_onnx_runtime_providers",
    "check_tensorflow_compute_capability",
]
