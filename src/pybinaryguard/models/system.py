"""System profile data model — the fingerprint of your machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, FrozenSet, Dict

from .enums import Architecture, ContainerRuntime


@dataclass(frozen=True)
class SystemProfile:
    """Complete profile of the current system's binary compatibility surface."""

    # Python
    python_version: Tuple[int, int, int] = (0, 0, 0)
    python_abi_tag: str = ""
    python_implementation: str = "cpython"
    python_executable: str = ""
    stable_abi_supported: bool = False
    python_debug_build: bool = False

    # System
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""
    architecture: Architecture = Architecture.UNKNOWN
    glibc_version: Optional[Tuple[int, int]] = None
    musl_version: Optional[Tuple[int, int]] = None

    # CPU
    cpu_model: str = ""
    cpu_flags: FrozenSet[str] = field(default_factory=frozenset)
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_sse42: bool = False
    has_neon: bool = False
    cpu_cores: int = 0

    # GPU
    gpu_available: bool = False
    gpu_driver_version: Optional[str] = None
    cuda_runtime_version: Optional[Tuple[int, int]] = None
    cuda_toolkit_version: Optional[Tuple[int, int]] = None
    gpu_compute_capability: Optional[Tuple[int, int]] = None
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    cudnn_version: Optional[Tuple[int, int, int]] = None

    # Environment
    is_container: bool = False
    container_runtime: ContainerRuntime = ContainerRuntime.NONE
    is_virtual_machine: bool = False
    is_embedded_board: bool = False
    board_name: Optional[str] = None
    jetpack_version: Optional[str] = None

    # Build toolchain
    toolchain_versions: Dict[str, str] = field(default_factory=dict)
    default_cc: str = ""
    default_cxx: str = ""
    has_build_tools: bool = False
    has_python_dev_headers: bool = False

    # Virtual environment
    venv_type: str = "system"
    is_system_python: bool = True
    is_virtual_env: bool = False
    base_prefix: str = ""
    prefix: str = ""
    conda_env_name: str = ""
    conda_prefix: str = ""
    pip_user_site_enabled: bool = False
    mixed_env_risk: bool = False

    # Library paths
    ld_library_path: Tuple[str, ...] = ()
    ldconfig_cache: Dict[str, str] = field(default_factory=dict)
    site_packages_paths: Tuple[str, ...] = ()

    def summary(self) -> Dict[str, str]:
        """Return a human-readable summary dict for display."""
        info: Dict[str, str] = {}
        if self.python_version != (0, 0, 0):
            py_ver = ".".join(str(v) for v in self.python_version)
            info["Python"] = f"{py_ver} ({self.python_abi_tag}) @ {self.python_executable}"
        if self.os_name:
            info["OS"] = f"{self.os_name} {self.os_version} (kernel {self.kernel_version})"
        if self.architecture != Architecture.UNKNOWN:
            info["Architecture"] = self.architecture.value
        if self.glibc_version:
            info["GLIBC"] = f"{self.glibc_version[0]}.{self.glibc_version[1]}"
        elif self.musl_version:
            info["musl"] = f"{self.musl_version[0]}.{self.musl_version[1]}"
        if self.cuda_runtime_version:
            info["CUDA Runtime"] = f"{self.cuda_runtime_version[0]}.{self.cuda_runtime_version[1]}"
        if self.gpu_driver_version:
            info["GPU Driver"] = self.gpu_driver_version
        if self.gpu_name:
            cc = ""
            if self.gpu_compute_capability:
                cc = f" (compute {self.gpu_compute_capability[0]}.{self.gpu_compute_capability[1]})"
            info["GPU"] = f"{self.gpu_name}{cc}"
        if self.is_embedded_board and self.board_name:
            info["Board"] = self.board_name
            if self.jetpack_version:
                info["JetPack"] = self.jetpack_version
        if self.is_container:
            info["Container"] = self.container_runtime.value
        if self.venv_type != "system":
            info["Environment"] = self.venv_type
        if self.toolchain_versions:
            cc = self.toolchain_versions.get("gcc") or self.toolchain_versions.get("clang")
            if cc:
                compiler = "gcc" if "gcc" in self.toolchain_versions else "clang"
                info["Compiler"] = f"{compiler} {cc}"
        return info
