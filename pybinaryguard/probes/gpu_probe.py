"""Probe for GPU, CUDA, and cuDNN information."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

from .base import ProbeBase


class GpuProbe(ProbeBase):
    """Layered GPU detection probe.

    Detection layers (tried in order, each one additive):

    1. Device nodes: ``/dev/nvidia0``, ``/dev/nvidiactl``
    2. Driver version: ``/proc/driver/nvidia/version``
    3. NVML via ctypes: ``libnvidia-ml.so.1`` -- GPU name, memory,
       compute capability, driver version
    4. CUDA runtime via ctypes: ``libcudart.so`` -- runtime version
    5. CUDA environment: ``CUDA_HOME`` / ``CUDA_PATH``, ``nvcc``
    6. Jetson-specific: ``/etc/nv_tegra_release`` for L4T / JetPack
    7. cuDNN via ctypes: ``libcudnn.so`` -- version

    All layers are optional.  If no GPU libraries are available the
    probe returns ``gpu_available=False`` and nothing else.
    """

    name = "gpu"

    def collect(self) -> Dict[str, Any]:
        """Return GPU, CUDA, and cuDNN information."""
        data: Dict[str, Any] = {}

        # Layer 1: device nodes
        has_device = self._check_device_nodes()

        # Layer 2: driver version from /proc
        driver_version = self._parse_proc_driver_version()
        if driver_version:
            data["gpu_driver_version"] = driver_version

        # Layer 3: NVML
        nvml_info = self._query_nvml()
        if nvml_info:
            data.update(nvml_info)

        # Layer 4: CUDA runtime
        cuda_runtime = self._query_cuda_runtime()
        if cuda_runtime:
            data["cuda_runtime_version"] = cuda_runtime

        # Layer 5: CUDA toolkit from environment
        cuda_toolkit = self._detect_cuda_toolkit()
        if cuda_toolkit:
            data["cuda_toolkit_version"] = cuda_toolkit

        # Layer 6: Jetson-specific L4T info
        jetson_info = self._detect_jetson_gpu()
        if jetson_info:
            # Jetson has an integrated GPU -- mark available even if
            # /dev/nvidia0 is absent (Jetson uses /dev/nvhost-*)
            has_device = True
            # Only set fields that weren't already populated
            for key, value in jetson_info.items():
                if key not in data:
                    data[key] = value

        # Layer 7: cuDNN
        cudnn_ver = self._detect_cudnn()
        if cudnn_ver:
            data["cudnn_version"] = cudnn_ver

        # Final: determine gpu_available
        data["gpu_available"] = has_device or bool(data.get("gpu_driver_version"))

        return data

    # ------------------------------------------------------------------
    # Layer 1: Device nodes
    # ------------------------------------------------------------------

    @staticmethod
    def _check_device_nodes() -> bool:
        """Check for NVIDIA device nodes in ``/dev``."""
        return os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl")

    # ------------------------------------------------------------------
    # Layer 2: /proc driver version
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_proc_driver_version() -> Optional[str]:
        """Parse ``/proc/driver/nvidia/version`` for the driver version.

        Example content::

            NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.129.03 ...
        """
        try:
            with open("/proc/driver/nvidia/version", "r") as fh:
                content = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        match = re.search(r"Kernel Module\s+([\d.]+)", content)
        if match:
            return match.group(1)

        # Fallback: look for any version-like pattern on the NVRM line
        for line in content.splitlines():
            if "NVRM" in line:
                ver_match = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
                if ver_match:
                    return ver_match.group(1)
        return None

    # ------------------------------------------------------------------
    # Layer 3: NVML
    # ------------------------------------------------------------------

    @staticmethod
    def _query_nvml() -> Optional[Dict[str, Any]]:
        """Query NVIDIA Management Library (NVML) via ctypes.

        Collects GPU name, memory, compute capability, and driver version
        for the first GPU (device index 0).
        """
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            return None

        result: Dict[str, Any] = {}

        try:
            # nvmlInit_v2
            ret = nvml.nvmlInit_v2()
            if ret != 0:
                return None

            # Driver version
            buf = ctypes.create_string_buffer(256)
            ret = nvml.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(256))
            if ret == 0:
                result["gpu_driver_version"] = buf.value.decode("utf-8", errors="replace")

            # Get device handle for GPU 0
            handle = ctypes.c_void_p()
            ret = nvml.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(handle))
            if ret != 0:
                # Try legacy API
                ret = nvml.nvmlDeviceGetHandleByIndex(ctypes.c_uint(0), ctypes.byref(handle))
            if ret != 0:
                nvml.nvmlShutdown()
                return result if result else None

            # GPU name
            name_buf = ctypes.create_string_buffer(256)
            ret = nvml.nvmlDeviceGetName(handle, name_buf, ctypes.c_uint(256))
            if ret == 0:
                result["gpu_name"] = name_buf.value.decode("utf-8", errors="replace")

            # Memory info
            class NvmlMemory(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            mem = NvmlMemory()
            ret = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
            if ret == 0:
                result["gpu_memory_mb"] = int(mem.total // (1024 * 1024))

            # Compute capability
            major = ctypes.c_int()
            minor = ctypes.c_int()
            ret = nvml.nvmlDeviceGetCudaComputeCapability(
                handle, ctypes.byref(major), ctypes.byref(minor)
            )
            if ret == 0:
                result["gpu_compute_capability"] = (major.value, minor.value)

            nvml.nvmlShutdown()
        except Exception:
            # Catch any ctypes-level errors
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
            return result if result else None

        return result if result else None

    # ------------------------------------------------------------------
    # Layer 4: CUDA runtime
    # ------------------------------------------------------------------

    @staticmethod
    def _query_cuda_runtime() -> Optional[Tuple[int, int]]:
        """Query the CUDA runtime library for its version.

        Tries common soname patterns for ``libcudart``.
        """
        lib_names = [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "libcudart.so.11",
            "libcudart.so.10.2",
            "libcudart.so.10.1",
        ]

        for lib_name in lib_names:
            try:
                cudart = ctypes.CDLL(lib_name)
                version = ctypes.c_int()
                ret = cudart.cudaRuntimeGetVersion(ctypes.byref(version))
                if ret == 0 and version.value > 0:
                    # CUDA encodes version as major*1000 + minor*10
                    major = version.value // 1000
                    minor = (version.value % 1000) // 10
                    return (major, minor)
            except (OSError, AttributeError):
                continue

        return None

    # ------------------------------------------------------------------
    # Layer 5: CUDA toolkit from environment
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_cuda_toolkit() -> Optional[Tuple[int, int]]:
        """Detect the CUDA toolkit version from environment variables and nvcc.

        Checks ``CUDA_HOME``, ``CUDA_PATH``, and common install
        directories for ``nvcc --version`` output.
        """
        cuda_dirs = []

        # Environment variables
        for env_var in ("CUDA_HOME", "CUDA_PATH"):
            val = os.environ.get(env_var, "")
            if val and os.path.isdir(val):
                cuda_dirs.append(val)

        # Common install paths
        common_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12",
            "/usr/local/cuda-11",
            "/opt/cuda",
        ]
        for path in common_paths:
            if os.path.isdir(path) and path not in cuda_dirs:
                cuda_dirs.append(path)

        # Try nvcc in each candidate directory
        for cuda_dir in cuda_dirs:
            nvcc = os.path.join(cuda_dir, "bin", "nvcc")
            ver = GpuProbe._nvcc_version(nvcc)
            if ver is not None:
                return ver

        # Try nvcc on PATH
        ver = GpuProbe._nvcc_version("nvcc")
        if ver is not None:
            return ver

        return None

    @staticmethod
    def _nvcc_version(nvcc_path: str) -> Optional[Tuple[int, int]]:
        """Run ``nvcc --version`` and parse the CUDA version.

        Example output line::

            Cuda compilation tools, release 12.2, V12.2.140
        """
        try:
            result = subprocess.run(
                [nvcc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            match = re.search(r"release\s+(\d+)\.(\d+)", output)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            pass
        return None

    # ------------------------------------------------------------------
    # Layer 6: Jetson / L4T
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_jetson_gpu() -> Optional[Dict[str, Any]]:
        """Detect Jetson integrated GPU via ``/etc/nv_tegra_release``.

        Jetson devices have an integrated NVIDIA GPU that does not
        expose ``/dev/nvidia0`` -- they use ``/dev/nvhost-*`` instead.
        """
        try:
            with open("/etc/nv_tegra_release", "r") as fh:
                content = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        if not content.strip():
            return None

        info: Dict[str, Any] = {}
        info["gpu_name"] = "NVIDIA Tegra (Jetson)"

        # Try to determine CUDA version from the toolkit symlink
        cuda_ver = GpuProbe._detect_cuda_toolkit()
        if cuda_ver:
            info["cuda_toolkit_version"] = cuda_ver

        return info

    # ------------------------------------------------------------------
    # Layer 7: cuDNN
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_cudnn() -> Optional[Tuple[int, int, int]]:
        """Detect cuDNN version via ctypes or header file.

        Tries:
        1. Loading ``libcudnn.so`` and calling ``cudnnGetVersion()``.
        2. Parsing ``cudnn_version.h`` from CUDA include directories.
        """
        # Method 1: ctypes
        ver = GpuProbe._cudnn_via_ctypes()
        if ver is not None:
            return ver

        # Method 2: header file
        ver = GpuProbe._cudnn_via_header()
        if ver is not None:
            return ver

        return None

    @staticmethod
    def _cudnn_via_ctypes() -> Optional[Tuple[int, int, int]]:
        """Load ``libcudnn.so`` and query the version."""
        lib_names = [
            "libcudnn.so",
            "libcudnn.so.9",
            "libcudnn.so.8",
            "libcudnn.so.7",
        ]
        for lib_name in lib_names:
            try:
                cudnn = ctypes.CDLL(lib_name)
                cudnn.cudnnGetVersion.restype = ctypes.c_size_t
                version = cudnn.cudnnGetVersion()
                if version > 0:
                    # cuDNN encodes version as major*1000 + minor*100 + patch
                    major = version // 1000
                    minor = (version % 1000) // 100
                    patch = version % 100
                    return (major, minor, patch)
            except (OSError, AttributeError):
                continue
        return None

    @staticmethod
    def _cudnn_via_header() -> Optional[Tuple[int, int, int]]:
        """Parse cuDNN version from ``cudnn_version.h``.

        The header defines::

            #define CUDNN_MAJOR 8
            #define CUDNN_MINOR 9
            #define CUDNN_PATCHLEVEL 7
        """
        search_dirs = []

        # CUDA_HOME / CUDA_PATH
        for env_var in ("CUDA_HOME", "CUDA_PATH"):
            val = os.environ.get(env_var, "")
            if val:
                search_dirs.append(os.path.join(val, "include"))

        # Common locations
        search_dirs.extend([
            "/usr/local/cuda/include",
            "/usr/include",
            "/usr/include/x86_64-linux-gnu",
            "/usr/include/aarch64-linux-gnu",
        ])

        for include_dir in search_dirs:
            # Try cudnn_version.h first (cuDNN >= 8), then cudnn.h
            for header_name in ("cudnn_version.h", "cudnn.h"):
                header_path = os.path.join(include_dir, header_name)
                ver = GpuProbe._parse_cudnn_header(header_path)
                if ver is not None:
                    return ver

        return None

    @staticmethod
    def _parse_cudnn_header(path: str) -> Optional[Tuple[int, int, int]]:
        """Extract CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL from a header."""
        try:
            with open(path, "r") as fh:
                content = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        major_m = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", content)
        minor_m = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", content)
        patch_m = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", content)

        if major_m and minor_m and patch_m:
            return (int(major_m.group(1)), int(minor_m.group(1)), int(patch_m.group(1)))
        return None
