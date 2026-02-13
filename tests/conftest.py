"""Shared test fixtures for PyBinaryGuard test suite."""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict

import pytest

from pybinaryguard.models.enums import Architecture, ContainerRuntime, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo, WheelTag
from pybinaryguard.models.system import SystemProfile


@pytest.fixture
def sample_profile() -> SystemProfile:
    """A SystemProfile with common values (Ubuntu 22.04, x86_64, Python 3.12, GLIBC 2.35, CUDA 12.2)."""
    return SystemProfile(
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
        cpu_flags=frozenset({"avx", "avx2", "sse4_2", "avx512f"}),
        has_avx=True,
        has_avx2=True,
        has_avx512=True,
        has_sse42=True,
        has_neon=False,
        cpu_cores=20,
        gpu_available=True,
        gpu_driver_version="535.129.03",
        cuda_runtime_version=(12, 2),
        cuda_toolkit_version=(12, 2),
        gpu_compute_capability=(8, 9),
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_memory_mb=24576,
        cudnn_version=(8, 9, 7),
        is_container=False,
        container_runtime=ContainerRuntime.NONE,
        is_virtual_machine=False,
        is_embedded_board=False,
        board_name=None,
        jetpack_version=None,
        ld_library_path=("/usr/local/cuda/lib64",),
        site_packages_paths=("/usr/lib/python3/dist-packages",),
    )


@pytest.fixture
def jetson_profile() -> SystemProfile:
    """A SystemProfile for Jetson Orin (aarch64, JetPack 6.0, GLIBC 2.35, CUDA 12.2)."""
    return SystemProfile(
        python_version=(3, 10, 12),
        python_abi_tag="cpython-310-aarch64-linux-gnu",
        python_implementation="cpython",
        python_executable="/usr/bin/python3",
        stable_abi_supported=True,
        python_debug_build=False,
        os_name="Ubuntu",
        os_version="22.04",
        kernel_version="5.15.136-tegra",
        architecture=Architecture.AARCH64,
        glibc_version=(2, 35),
        musl_version=None,
        cpu_model="ARMv8 Processor rev 1 (v8l)",
        cpu_flags=frozenset({"neon", "fp", "asimd"}),
        has_avx=False,
        has_avx2=False,
        has_avx512=False,
        has_sse42=False,
        has_neon=True,
        cpu_cores=12,
        gpu_available=True,
        gpu_driver_version="535.104.05",
        cuda_runtime_version=(12, 2),
        cuda_toolkit_version=(12, 2),
        gpu_compute_capability=(8, 7),
        gpu_name="NVIDIA Tegra Orin (iGPU)",
        gpu_memory_mb=32768,
        cudnn_version=(8, 9, 4),
        is_container=False,
        container_runtime=ContainerRuntime.NONE,
        is_virtual_machine=False,
        is_embedded_board=True,
        board_name="NVIDIA Jetson AGX Orin",
        jetpack_version="6.0",
    )


@pytest.fixture
def alpine_profile() -> SystemProfile:
    """A SystemProfile for Alpine Linux (musl, no glibc)."""
    return SystemProfile(
        python_version=(3, 11, 6),
        python_abi_tag="cpython-311-x86_64-linux-musl",
        python_implementation="cpython",
        python_executable="/usr/bin/python3",
        stable_abi_supported=True,
        python_debug_build=False,
        os_name="Alpine Linux",
        os_version="3.19",
        kernel_version="6.1.0-0-virt",
        architecture=Architecture.X86_64,
        glibc_version=None,
        musl_version=(1, 2),
        cpu_model="AMD EPYC 7R13",
        cpu_flags=frozenset({"avx", "avx2", "sse4_2"}),
        has_avx=True,
        has_avx2=True,
        has_avx512=False,
        has_sse42=True,
        has_neon=False,
        cpu_cores=4,
        gpu_available=False,
        is_container=True,
        container_runtime=ContainerRuntime.DOCKER,
    )


@pytest.fixture
def no_gpu_profile() -> SystemProfile:
    """A SystemProfile with no GPU."""
    return SystemProfile(
        python_version=(3, 12, 0),
        python_abi_tag="cpython-312-x86_64-linux-gnu",
        python_implementation="cpython",
        python_executable="/usr/bin/python3",
        stable_abi_supported=True,
        os_name="Ubuntu",
        os_version="22.04",
        kernel_version="5.15.0-100-generic",
        architecture=Architecture.X86_64,
        glibc_version=(2, 35),
        cpu_model="Intel(R) Core(TM) i5-10400",
        has_avx=True,
        has_avx2=True,
        has_sse42=True,
        cpu_cores=12,
        gpu_available=False,
    )


@pytest.fixture
def sample_package() -> PackageBinaryInfo:
    """A PackageBinaryInfo for a torch-like package."""
    so1 = SharedObjectInfo(
        path="/usr/lib/python3/dist-packages/torch/lib/libtorch_cuda.so",
        filename="libtorch_cuda.so",
        architecture=Architecture.X86_64,
        elf_class=64,
        endianness="little",
        dt_needed=["libcudart.so.12", "libc.so.6", "libstdc++.so.6"],
        required_glibc=(2, 28),
        gnu_version_requirements=["libc.so.6(GLIBC_2.28)"],
        file_size=1024000,
    )
    so2 = SharedObjectInfo(
        path="/usr/lib/python3/dist-packages/torch/lib/libtorch_cpu.so",
        filename="libtorch_cpu.so",
        architecture=Architecture.X86_64,
        elf_class=64,
        endianness="little",
        dt_needed=["libc.so.6", "libstdc++.so.6"],
        required_glibc=(2, 17),
        gnu_version_requirements=["libc.so.6(GLIBC_2.17)"],
        file_size=512000,
    )
    pkg = PackageBinaryInfo(
        package_name="torch",
        package_version="2.4.0+cu124",
        install_path="/usr/lib/python3/dist-packages/torch",
        wheel_tags=[
            WheelTag(
                interpreter="cp312",
                abi="cp312",
                platform="manylinux_2_17_x86_64",
            )
        ],
        is_pure_python=False,
        shared_objects=[so1, so2],
        required_glibc=(2, 28),
        target_architecture=Architecture.X86_64,
        required_libraries={"libcudart.so.12", "libc.so.6", "libstdc++.so.6"},
        cuda_build_version=(12, 4),
    )
    return pkg


@pytest.fixture
def pure_python_package() -> PackageBinaryInfo:
    """A PackageBinaryInfo for a pure Python package."""
    return PackageBinaryInfo(
        package_name="requests",
        package_version="2.31.0",
        install_path="/usr/lib/python3/dist-packages/requests",
        wheel_tags=[
            WheelTag(interpreter="py3", abi="none", platform="any"),
        ],
        is_pure_python=True,
        shared_objects=[],
    )


@pytest.fixture
def tmp_site_packages(tmp_path: Any) -> str:
    """Creates a temporary site-packages dir with fake dist-info."""
    sp = tmp_path / "site-packages"
    sp.mkdir()

    # Create a fake package dist-info
    dist_info = sp / "mypackage-1.0.0.dist-info"
    dist_info.mkdir()

    wheel_content = textwrap.dedent("""\
        Wheel-Version: 1.0
        Generator: setuptools
        Root-Is-Purelib: false
        Tag: cp312-cp312-manylinux_2_17_x86_64
    """)
    (dist_info / "WHEEL").write_text(wheel_content)

    metadata_content = textwrap.dedent("""\
        Metadata-Version: 2.1
        Name: mypackage
        Version: 1.0.0
        Summary: A test package
    """)
    (dist_info / "METADATA").write_text(metadata_content)

    record_content = textwrap.dedent("""\
        mypackage/__init__.py,sha256=abc123,100
        mypackage/core.cpython-312-x86_64-linux-gnu.so,sha256=def456,50000
    """)
    (dist_info / "RECORD").write_text(record_content)

    # Create the package directory itself
    pkg_dir = sp / "mypackage"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("# mypackage\n")

    return str(sp)
