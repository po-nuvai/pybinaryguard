"""Tests for all built-in compatibility rules."""

from __future__ import annotations

from typing import List

import pytest

from pybinaryguard.models.enums import Architecture, ContainerRuntime, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo, WheelTag
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.builtin.glibc_rules import (
    GLIBCVersionMismatchRule,
    ManylinuxTagViolationRule,
    MuslGlibcConflictRule,
)
from pybinaryguard.rules.builtin.python_abi_rules import (
    DebugReleaseMixRule,
    PythonABIMismatchRule,
    PythonVersionMismatchRule,
)
from pybinaryguard.rules.builtin.arch_rules import ArchMismatchRule
from pybinaryguard.rules.builtin.cuda_rules import (
    CUDANotFoundRule,
    CUDARuntimeMismatchRule,
)
from pybinaryguard.rules.builtin.numpy_rules import NumpyABIMismatchRule
from pybinaryguard.rules.builtin.container_rules import ContainerNoGPUMountRule
from pybinaryguard.rules.builtin.cpu_rules import (
    AVX2RequiredRule,
    IllegalInstructionRiskRule,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic profiles and packages
# ---------------------------------------------------------------------------


def _make_profile(**kwargs: object) -> SystemProfile:
    """Create a SystemProfile with defaults that can be overridden."""
    defaults = dict(
        python_version=(3, 12, 0),
        python_abi_tag="cpython-312-x86_64-linux-gnu",
        python_implementation="cpython",
        python_executable="/usr/bin/python3",
        stable_abi_supported=True,
        python_debug_build=False,
        os_name="Ubuntu",
        os_version="22.04",
        kernel_version="5.15.0-100-generic",
        architecture=Architecture.X86_64,
        glibc_version=(2, 35),
        musl_version=None,
        cpu_model="Intel(R) Core(TM) i7-12700K",
        cpu_flags=frozenset({"avx", "avx2", "sse4_2"}),
        has_avx=True,
        has_avx2=True,
        has_avx512=False,
        has_sse42=True,
        has_neon=False,
        cpu_cores=16,
        gpu_available=False,
        is_container=False,
        container_runtime=ContainerRuntime.NONE,
    )
    defaults.update(kwargs)
    return SystemProfile(**defaults)


def _make_package(**kwargs: object) -> PackageBinaryInfo:
    """Create a PackageBinaryInfo with defaults that can be overridden."""
    defaults = dict(
        package_name="testpkg",
        package_version="1.0.0",
        install_path="/tmp/testpkg",
        is_pure_python=False,
    )
    defaults.update(kwargs)
    return PackageBinaryInfo(**defaults)


# ---------------------------------------------------------------------------
# GLIBCVersionMismatchRule
# ---------------------------------------------------------------------------


class TestGLIBCVersionMismatchRule:
    def test_is_applicable_glibc_system(self) -> None:
        rule = GLIBCVersionMismatchRule()
        profile = _make_profile(glibc_version=(2, 35))
        assert rule.is_applicable(profile) is True

    def test_is_not_applicable_no_glibc(self) -> None:
        rule = GLIBCVersionMismatchRule()
        profile = _make_profile(glibc_version=None, musl_version=(1, 2))
        assert rule.is_applicable(profile) is False

    def test_no_finding_when_glibc_sufficient(self) -> None:
        rule = GLIBCVersionMismatchRule()
        profile = _make_profile(glibc_version=(2, 35))
        pkg = _make_package(required_glibc=(2, 28))
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_finding_when_glibc_too_old(self) -> None:
        rule = GLIBCVersionMismatchRule()
        profile = _make_profile(glibc_version=(2, 17))
        pkg = _make_package(
            package_name="numpy",
            required_glibc=(2, 28),
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].rule_id == "GLIBC_VERSION_MISMATCH"
        assert "numpy" in findings[0].title

    def test_skips_packages_without_glibc_requirement(self) -> None:
        rule = GLIBCVersionMismatchRule()
        profile = _make_profile(glibc_version=(2, 35))
        pkg = _make_package(required_glibc=None)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# MuslGlibcConflictRule
# ---------------------------------------------------------------------------


class TestMuslGlibcConflictRule:
    def test_is_applicable_musl_system(self) -> None:
        rule = MuslGlibcConflictRule()
        profile = _make_profile(glibc_version=None, musl_version=(1, 2))
        assert rule.is_applicable(profile) is True

    def test_is_not_applicable_glibc_system(self) -> None:
        rule = MuslGlibcConflictRule()
        profile = _make_profile(glibc_version=(2, 35), musl_version=None)
        assert rule.is_applicable(profile) is False

    def test_finding_for_manylinux_package_on_musl(self) -> None:
        rule = MuslGlibcConflictRule()
        profile = _make_profile(glibc_version=None, musl_version=(1, 2))
        pkg = _make_package(
            package_name="pandas",
            wheel_tags=[WheelTag("cp312", "cp312", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert "musl" in findings[0].title.lower() or "manylinux" in findings[0].title.lower()

    def test_no_finding_for_non_manylinux_package(self) -> None:
        rule = MuslGlibcConflictRule()
        profile = _make_profile(glibc_version=None, musl_version=(1, 2))
        pkg = _make_package(
            wheel_tags=[WheelTag("py3", "none", "any")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# ManylinuxTagViolationRule
# ---------------------------------------------------------------------------


class TestManylinuxTagViolationRule:
    def test_finding_when_actual_exceeds_claimed(self) -> None:
        rule = ManylinuxTagViolationRule()
        profile = _make_profile()
        pkg = _make_package(
            package_name="badpkg",
            wheel_tags=[WheelTag("cp312", "cp312", "manylinux_2_17_x86_64")],
            required_glibc=(2, 28),  # Actual > claimed (2, 17)
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert findings[0].rule_id == "MANYLINUX_TAG_VIOLATION"

    def test_no_finding_when_actual_within_claimed(self) -> None:
        rule = ManylinuxTagViolationRule()
        profile = _make_profile()
        pkg = _make_package(
            wheel_tags=[WheelTag("cp312", "cp312", "manylinux_2_17_x86_64")],
            required_glibc=(2, 17),
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_no_finding_when_no_manylinux_tag(self) -> None:
        rule = ManylinuxTagViolationRule()
        profile = _make_profile()
        pkg = _make_package(
            wheel_tags=[WheelTag("py3", "none", "any")],
            required_glibc=(2, 28),
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# PythonABIMismatchRule
# ---------------------------------------------------------------------------


class TestPythonABIMismatchRule:
    def test_finding_when_abi_mismatch(self) -> None:
        rule = PythonABIMismatchRule()
        profile = _make_profile(python_abi_tag="cpython-312-x86_64-linux-gnu")
        pkg = _make_package(
            package_name="scipy",
            wheel_tags=[WheelTag("cp310", "cp310", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].rule_id == "PYTHON_ABI_MISMATCH"

    def test_no_finding_when_abi_matches(self) -> None:
        rule = PythonABIMismatchRule()
        profile = _make_profile(python_abi_tag="cpython-312-x86_64-linux-gnu")
        pkg = _make_package(
            wheel_tags=[WheelTag("cp312", "cpython-312-x86_64-linux-gnu", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_abi3_tags(self) -> None:
        rule = PythonABIMismatchRule()
        profile = _make_profile(python_abi_tag="cpython-312-x86_64-linux-gnu")
        pkg = _make_package(
            wheel_tags=[WheelTag("cp312", "abi3", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_none_abi(self) -> None:
        rule = PythonABIMismatchRule()
        profile = _make_profile(python_abi_tag="cpython-312-x86_64-linux-gnu")
        pkg = _make_package(
            wheel_tags=[WheelTag("py3", "none", "any")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_pure_python(self) -> None:
        rule = PythonABIMismatchRule()
        profile = _make_profile(python_abi_tag="cpython-312-x86_64-linux-gnu")
        pkg = _make_package(is_pure_python=True)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# PythonVersionMismatchRule
# ---------------------------------------------------------------------------


class TestPythonVersionMismatchRule:
    def test_finding_when_version_differs(self) -> None:
        rule = PythonVersionMismatchRule()
        profile = _make_profile(python_version=(3, 12, 0))
        pkg = _make_package(
            package_name="sklearn",
            wheel_tags=[WheelTag("cp310", "cp310", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert "3.10" in findings[0].title

    def test_no_finding_when_version_matches(self) -> None:
        rule = PythonVersionMismatchRule()
        profile = _make_profile(python_version=(3, 12, 0))
        pkg = _make_package(
            wheel_tags=[WheelTag("cp312", "cp312", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_py3_universal_tag(self) -> None:
        rule = PythonVersionMismatchRule()
        profile = _make_profile(python_version=(3, 12, 0))
        pkg = _make_package(
            wheel_tags=[WheelTag("py3", "none", "any")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# ArchMismatchRule
# ---------------------------------------------------------------------------


class TestArchMismatchRule:
    def test_finding_when_arch_differs(self) -> None:
        rule = ArchMismatchRule()
        profile = _make_profile(architecture=Architecture.X86_64)
        pkg = _make_package(
            package_name="mypkg",
            target_architecture=Architecture.AARCH64,
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].rule_id == "ARCH_MISMATCH"

    def test_no_finding_when_arch_matches(self) -> None:
        rule = ArchMismatchRule()
        profile = _make_profile(architecture=Architecture.X86_64)
        pkg = _make_package(target_architecture=Architecture.X86_64)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_unknown_system_arch(self) -> None:
        rule = ArchMismatchRule()
        profile = _make_profile(architecture=Architecture.UNKNOWN)
        pkg = _make_package(target_architecture=Architecture.X86_64)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_unknown_package_arch(self) -> None:
        rule = ArchMismatchRule()
        profile = _make_profile(architecture=Architecture.X86_64)
        pkg = _make_package(target_architecture=Architecture.UNKNOWN)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_no_target_architecture(self) -> None:
        rule = ArchMismatchRule()
        profile = _make_profile(architecture=Architecture.X86_64)
        pkg = _make_package(target_architecture=None)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# CUDARuntimeMismatchRule
# ---------------------------------------------------------------------------


class TestCUDARuntimeMismatchRule:
    def test_is_applicable_with_cuda(self) -> None:
        rule = CUDARuntimeMismatchRule()
        profile = _make_profile(
            gpu_available=True,
            cuda_runtime_version=(12, 2),
        )
        assert rule.is_applicable(profile) is True

    def test_is_not_applicable_without_cuda(self) -> None:
        rule = CUDARuntimeMismatchRule()
        profile = _make_profile(gpu_available=False, cuda_runtime_version=None)
        assert rule.is_applicable(profile) is False

    def test_finding_when_cuda_major_differs(self) -> None:
        rule = CUDARuntimeMismatchRule()
        profile = _make_profile(
            gpu_available=True,
            cuda_runtime_version=(12, 2),
        )
        pkg = _make_package(
            package_name="torch",
            package_version="2.1.0+cu118",
            cuda_build_version=(11, 8),
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].rule_id == "CUDA_RUNTIME_MISMATCH"

    def test_no_finding_when_cuda_major_matches(self) -> None:
        rule = CUDARuntimeMismatchRule()
        profile = _make_profile(
            gpu_available=True,
            cuda_runtime_version=(12, 2),
        )
        pkg = _make_package(
            package_name="torch",
            package_version="2.4.0+cu124",
            cuda_build_version=(12, 4),
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_extracts_cuda_from_version_string(self) -> None:
        rule = CUDARuntimeMismatchRule()
        profile = _make_profile(
            gpu_available=True,
            cuda_runtime_version=(12, 2),
        )
        pkg = _make_package(
            package_name="torch",
            package_version="2.1.0+cu118",
            cuda_build_version=None,  # Not set, must extract from version
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1  # 11 != 12 -> mismatch


# ---------------------------------------------------------------------------
# CUDANotFoundRule
# ---------------------------------------------------------------------------


class TestCUDANotFoundRule:
    def test_is_applicable_when_no_cuda(self) -> None:
        rule = CUDANotFoundRule()
        profile = _make_profile(cuda_runtime_version=None)
        assert rule.is_applicable(profile) is True

    def test_is_not_applicable_when_cuda_present(self) -> None:
        rule = CUDANotFoundRule()
        profile = _make_profile(cuda_runtime_version=(12, 2))
        assert rule.is_applicable(profile) is False

    def test_finding_for_cuda_package_without_runtime(self) -> None:
        rule = CUDANotFoundRule()
        profile = _make_profile(cuda_runtime_version=None)
        pkg = _make_package(
            package_name="torch",
            package_version="2.4.0+cu124",
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert findings[0].rule_id == "CUDA_NOT_FOUND"

    def test_no_finding_for_non_cuda_package(self) -> None:
        rule = CUDANotFoundRule()
        profile = _make_profile(cuda_runtime_version=None)
        pkg = _make_package(
            package_name="requests",
            package_version="2.31.0",
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# NumpyABIMismatchRule
# ---------------------------------------------------------------------------


class TestNumpyABIMismatchRule:
    def test_finding_when_api_version_differs(self) -> None:
        rule = NumpyABIMismatchRule()
        profile = _make_profile()
        numpy_pkg = _make_package(
            package_name="numpy",
            package_version="1.26.4",
            numpy_api_version=0x10,
        )
        scipy_pkg = _make_package(
            package_name="scipy",
            package_version="1.12.0",
            numpy_api_version=0x0F,  # Different from numpy
        )
        findings = rule.evaluate(profile, [numpy_pkg, scipy_pkg])
        assert len(findings) == 1
        assert findings[0].rule_id == "NUMPY_ABI_MISMATCH"
        assert "scipy" in findings[0].title.lower()

    def test_no_finding_when_api_versions_match(self) -> None:
        rule = NumpyABIMismatchRule()
        profile = _make_profile()
        numpy_pkg = _make_package(
            package_name="numpy",
            package_version="1.26.4",
            numpy_api_version=0x10,
        )
        scipy_pkg = _make_package(
            package_name="scipy",
            package_version="1.12.0",
            numpy_api_version=0x10,
        )
        findings = rule.evaluate(profile, [numpy_pkg, scipy_pkg])
        assert len(findings) == 0

    def test_no_finding_when_numpy_not_installed(self) -> None:
        rule = NumpyABIMismatchRule()
        profile = _make_profile()
        pkg = _make_package(
            package_name="pandas",
            numpy_api_version=0x10,
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skips_numpy_itself(self) -> None:
        rule = NumpyABIMismatchRule()
        profile = _make_profile()
        numpy_pkg = _make_package(
            package_name="numpy",
            package_version="1.26.4",
            numpy_api_version=0x10,
        )
        findings = rule.evaluate(profile, [numpy_pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# ContainerNoGPUMountRule
# ---------------------------------------------------------------------------


class TestContainerNoGPUMountRule:
    def test_is_applicable_container_no_gpu(self) -> None:
        rule = ContainerNoGPUMountRule()
        profile = _make_profile(
            is_container=True,
            container_runtime=ContainerRuntime.DOCKER,
            gpu_available=False,
        )
        assert rule.is_applicable(profile) is True

    def test_is_not_applicable_not_container(self) -> None:
        rule = ContainerNoGPUMountRule()
        profile = _make_profile(is_container=False, gpu_available=False)
        assert rule.is_applicable(profile) is False

    def test_is_not_applicable_container_with_gpu(self) -> None:
        rule = ContainerNoGPUMountRule()
        profile = _make_profile(
            is_container=True,
            gpu_available=True,
            container_runtime=ContainerRuntime.DOCKER,
        )
        assert rule.is_applicable(profile) is False

    def test_finding_for_cuda_package_in_container(self) -> None:
        rule = ContainerNoGPUMountRule()
        profile = _make_profile(
            is_container=True,
            container_runtime=ContainerRuntime.DOCKER,
            gpu_available=False,
        )
        pkg = _make_package(
            package_name="torch",
            package_version="2.4.0+cu124",
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert findings[0].rule_id == "CONTAINER_NO_GPU_MOUNT"


# ---------------------------------------------------------------------------
# AVX2RequiredRule
# ---------------------------------------------------------------------------


class TestAVX2RequiredRule:
    def test_is_applicable_no_avx2(self) -> None:
        rule = AVX2RequiredRule()
        profile = _make_profile(has_avx2=False)
        assert rule.is_applicable(profile) is True

    def test_is_not_applicable_with_avx2(self) -> None:
        rule = AVX2RequiredRule()
        profile = _make_profile(has_avx2=True)
        assert rule.is_applicable(profile) is False

    def test_finding_for_faiss_without_avx2(self) -> None:
        rule = AVX2RequiredRule()
        profile = _make_profile(has_avx2=False)
        pkg = _make_package(
            package_name="faiss-cpu",
            package_version="1.7.4",
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL

    def test_no_finding_for_unknown_package(self) -> None:
        rule = AVX2RequiredRule()
        profile = _make_profile(has_avx2=False)
        pkg = _make_package(package_name="myunknownpkg")
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# IllegalInstructionRiskRule
# ---------------------------------------------------------------------------


class TestIllegalInstructionRiskRule:
    def test_finding_for_tensorflow_without_avx(self) -> None:
        rule = IllegalInstructionRiskRule()
        profile = _make_profile(
            has_avx=False,
            has_avx2=False,
            cpu_flags=frozenset({"sse4_2"}),
        )
        pkg = _make_package(
            package_name="tensorflow",
            package_version="2.15.0",
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].rule_id == "ILLEGAL_INSTRUCTION_RISK"

    def test_no_finding_when_instructions_met(self) -> None:
        rule = IllegalInstructionRiskRule()
        profile = _make_profile(has_avx=True, has_avx2=True)
        pkg = _make_package(
            package_name="tensorflow",
            package_version="2.15.0",
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# DebugReleaseMixRule
# ---------------------------------------------------------------------------


class TestDebugReleaseMixRule:
    def test_no_finding_for_release_build(self) -> None:
        rule = DebugReleaseMixRule()
        profile = _make_profile(python_debug_build=False)
        pkg = _make_package(
            wheel_tags=[WheelTag("cp312", "cp312", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_finding_for_debug_build_with_release_extension(self) -> None:
        rule = DebugReleaseMixRule()
        profile = _make_profile(python_debug_build=True)
        pkg = _make_package(
            package_name="scipy",
            wheel_tags=[WheelTag("cp312", "cp312", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert findings[0].rule_id == "DEBUG_RELEASE_MIX"

    def test_no_finding_for_debug_wheel_on_debug_interpreter(self) -> None:
        rule = DebugReleaseMixRule()
        profile = _make_profile(python_debug_build=True)
        pkg = _make_package(
            wheel_tags=[WheelTag("cp312", "cp312d", "manylinux_2_17_x86_64")],
        )
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0
