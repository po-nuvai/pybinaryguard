"""Built-in compatibility rules for PyBinaryGuard."""

from __future__ import annotations

from typing import List

from pybinaryguard.rules.base import Rule


def get_all_builtin_rules() -> List[Rule]:
    """Instantiate and return all built-in rule classes.

    Returns:
        One instance of each built-in rule, ready for registration with a
        :class:`~pybinaryguard.rules.engine.RuleEngine`.
    """
    from pybinaryguard.rules.builtin.arch_rules import ArchMismatchRule
    from pybinaryguard.rules.builtin.container_rules import (
        ContainerDriverMismatchRule,
        ContainerNoGPUMountRule,
    )
    from pybinaryguard.rules.builtin.cpu_rules import (
        AVX2RequiredRule,
        IllegalInstructionRiskRule,
    )
    from pybinaryguard.rules.builtin.cuda_rules import (
        ComputeCapabilityLowRule,
        CUDADriverTooOldRule,
        CUDALibMissingRule,
        CUDAMinorMismatchRule,
        CUDANotFoundRule,
        CUDARuntimeMismatchRule,
        CUDNNVersionMismatchRule,
    )
    from pybinaryguard.rules.builtin.glibc_rules import (
        GLIBCVersionMismatchRule,
        LibstdcxxVersionRule,
        ManylinuxTagViolationRule,
        MuslGlibcConflictRule,
    )
    from pybinaryguard.rules.builtin.numpy_rules import NumpyABIMismatchRule
    from pybinaryguard.rules.builtin.python_abi_rules import (
        DebugReleaseMixRule,
        PythonABIMismatchRule,
        PythonVersionMismatchRule,
    )
    from pybinaryguard.rules.builtin.board_profile_rules import (
        BOARD_CUDA_VERSION_MISMATCH,
        BOARD_GLIBC_MISMATCH,
        BOARD_INCOMPATIBLE_PACKAGE,
        BOARD_PYTHON_VERSION_UNSUPPORTED,
        KNOWN_BROKEN_WHEEL,
    )
    from pybinaryguard.rules.builtin.framework_rules import (
        ONNX_RUNTIME_PROVIDER_MISMATCH,
        PYTORCH_CUDA_ABI_MISMATCH,
        PYTORCH_TORCHVISION_INCOMPATIBLE,
        TENSORFLOW_COMPUTE_CAPABILITY_LOW,
        TENSORRT_INCOMPATIBLE,
    )
    from pybinaryguard.rules.builtin.predictive_rules import (
        PREDICTED_IMPORT_ERROR,
        UNRESOLVED_DEPENDENCY_CHAIN,
    )

    return [
        # GLIBC rules
        GLIBCVersionMismatchRule(),
        MuslGlibcConflictRule(),
        ManylinuxTagViolationRule(),
        LibstdcxxVersionRule(),
        # Python ABI rules
        PythonABIMismatchRule(),
        PythonVersionMismatchRule(),
        DebugReleaseMixRule(),
        # Architecture rules
        ArchMismatchRule(),
        # CPU instruction-set rules
        AVX2RequiredRule(),
        IllegalInstructionRiskRule(),
        # CUDA / GPU rules
        CUDADriverTooOldRule(),
        CUDARuntimeMismatchRule(),
        CUDAMinorMismatchRule(),
        CUDNNVersionMismatchRule(),
        CUDANotFoundRule(),
        ComputeCapabilityLowRule(),
        CUDALibMissingRule(),
        # NumPy ABI rules
        NumpyABIMismatchRule(),
        # Container rules
        ContainerNoGPUMountRule(),
        ContainerDriverMismatchRule(),
        # Board profile rules (embedded intelligence)
        KNOWN_BROKEN_WHEEL(),
        BOARD_INCOMPATIBLE_PACKAGE(),
        BOARD_CUDA_VERSION_MISMATCH(),
        BOARD_GLIBC_MISMATCH(),
        BOARD_PYTHON_VERSION_UNSUPPORTED(),
        # AI framework rules (deep inspection)
        PYTORCH_CUDA_ABI_MISMATCH(),
        PYTORCH_TORCHVISION_INCOMPATIBLE(),
        TENSORFLOW_COMPUTE_CAPABILITY_LOW(),
        TENSORRT_INCOMPATIBLE(),
        ONNX_RUNTIME_PROVIDER_MISMATCH(),
        # Predictive rules (runtime simulation)
        PREDICTED_IMPORT_ERROR(),
        UNRESOLVED_DEPENDENCY_CHAIN(),
    ]
