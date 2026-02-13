"""AI framework-specific compatibility rules.

Deep inspection rules for PyTorch, TensorFlow, TensorRT, and ONNX Runtime
that go beyond generic binary checks.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


class PYTORCH_CUDA_ABI_MISMATCH(Rule):
    """Detects PyTorch CUDA ABI mismatches using deep inspection."""

    rule_id = "PYTORCH_CUDA_ABI_MISMATCH"
    description = "Checks PyTorch CUDA ABI compatibility"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True  # Checked per-package inside evaluate

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.frameworks.pytorch import check_pytorch_cuda_abi

        findings: List[Finding] = []
        for package in packages:
            if package.package_name.lower() != "torch":
                continue

            issues = check_pytorch_cuda_abi(package, profile)
            for issue in issues:
                sev_map = {
                    "critical": Severity.CRITICAL,
                    "warning": Severity.WARNING,
                    "info": Severity.INFO,
                }
                severity = sev_map.get(
                    issue.get("severity", "warning"), Severity.WARNING
                )
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"PyTorch: {issue['issue'].replace('_', ' ').title()}",
                        explanation=issue["message"],
                        package=package.package_name,
                        suggestion=issue.get("recommendation", ""),
                        confidence=0.95,
                    )
                )
        return findings


class PYTORCH_TORCHVISION_INCOMPATIBLE(Rule):
    """Detects incompatible PyTorch and torchvision versions."""

    rule_id = "PYTORCH_TORCHVISION_INCOMPATIBLE"
    description = "Checks PyTorch and torchvision version compatibility"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.frameworks.pytorch import (
            check_pytorch_torchvision_compatibility,
        )

        findings: List[Finding] = []

        # Find torch and torchvision versions
        torch_version = None
        torchvision_pkg = None
        for package in packages:
            name_lower = package.package_name.lower()
            if name_lower == "torch" and package.package_version:
                torch_version = package.package_version
            elif name_lower == "torchvision":
                torchvision_pkg = package

        if torch_version is None or torchvision_pkg is None:
            return findings

        error = check_pytorch_torchvision_compatibility(
            torch_version, torchvision_pkg.package_version
        )
        if error:
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.WARNING,
                    title="Incompatible PyTorch and torchvision versions",
                    explanation=error,
                    package=torchvision_pkg.package_name,
                    suggestion=(
                        "Install matching torchvision version for your "
                        "PyTorch version"
                    ),
                    confidence=0.9,
                )
            )
        return findings


class TENSORFLOW_COMPUTE_CAPABILITY_LOW(Rule):
    """Detects TensorFlow compute capability requirements not met."""

    rule_id = "TENSORFLOW_COMPUTE_CAPABILITY_LOW"
    description = "Checks GPU compute capability for TensorFlow"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.frameworks.tensorflow import (
            check_tensorflow_compute_capability,
        )

        findings: List[Finding] = []
        for package in packages:
            if not package.package_name.lower().startswith("tensorflow"):
                continue

            issues = check_tensorflow_compute_capability(package, profile)
            for issue in issues:
                sev_map = {
                    "critical": Severity.CRITICAL,
                    "warning": Severity.WARNING,
                    "info": Severity.INFO,
                }
                severity = sev_map.get(
                    issue.get("severity", "warning"), Severity.WARNING
                )
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"TensorFlow: {issue['issue'].replace('_', ' ').title()}",
                        explanation=issue["message"],
                        package=package.package_name,
                        suggestion=issue.get("recommendation", ""),
                        confidence=0.95,
                    )
                )
        return findings


class TENSORRT_INCOMPATIBLE(Rule):
    """Detects TensorRT compatibility issues."""

    rule_id = "TENSORRT_INCOMPATIBLE"
    description = "Checks TensorRT compatibility with CUDA/cuDNN"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.frameworks.tensorrt import validate_tensorrt_engine

        findings: List[Finding] = []
        for package in packages:
            if not package.package_name.lower().startswith("tensorrt"):
                continue

            issues = validate_tensorrt_engine(package, profile)
            for issue in issues:
                sev_map = {
                    "critical": Severity.CRITICAL,
                    "warning": Severity.WARNING,
                    "info": Severity.INFO,
                }
                severity = sev_map.get(
                    issue.get("severity", "warning"), Severity.WARNING
                )
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"TensorRT: {issue['issue'].replace('_', ' ').title()}",
                        explanation=issue["message"],
                        package=package.package_name,
                        suggestion=issue.get("recommendation", ""),
                        confidence=0.9,
                    )
                )
        return findings


class ONNX_RUNTIME_PROVIDER_MISMATCH(Rule):
    """Detects ONNX Runtime execution provider mismatches."""

    rule_id = "ONNX_RUNTIME_PROVIDER_MISMATCH"
    description = "Checks ONNX Runtime execution provider compatibility"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.frameworks.onnxruntime import (
            check_onnx_runtime_providers,
        )

        findings: List[Finding] = []
        for package in packages:
            if not package.package_name.lower().startswith("onnxruntime"):
                continue

            issues = check_onnx_runtime_providers(package, profile)
            for issue in issues:
                sev_map = {
                    "critical": Severity.CRITICAL,
                    "warning": Severity.WARNING,
                    "info": Severity.INFO,
                }
                severity = sev_map.get(
                    issue.get("severity", "warning"), Severity.WARNING
                )
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=severity,
                        title=f"ONNX Runtime: {issue['issue'].replace('_', ' ').title()}",
                        explanation=issue["message"],
                        package=package.package_name,
                        suggestion=issue.get("recommendation", ""),
                        confidence=0.85,
                    )
                )
        return findings


__all__ = [
    "PYTORCH_CUDA_ABI_MISMATCH",
    "PYTORCH_TORCHVISION_INCOMPATIBLE",
    "TENSORFLOW_COMPUTE_CAPABILITY_LOW",
    "TENSORRT_INCOMPATIBLE",
    "ONNX_RUNTIME_PROVIDER_MISMATCH",
]
