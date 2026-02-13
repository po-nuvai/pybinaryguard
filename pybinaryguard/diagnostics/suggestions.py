"""Context-aware fix suggestion generator.

Given a :class:`Finding` and a :class:`SystemProfile`, this module produces
actionable, copy-pasteable remediation commands tailored to the user's
specific system (OS, architecture, CUDA version, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pybinaryguard.models.finding import Finding
from pybinaryguard.models.system import SystemProfile


# ---------------------------------------------------------------------------
# Well-known package index URLs for CUDA-specific wheels
# ---------------------------------------------------------------------------

PYTORCH_INDEX_URLS: Dict[Tuple[int, int], str] = {
    (11, 8): "https://download.pytorch.org/whl/cu118",
    (12, 1): "https://download.pytorch.org/whl/cu121",
    (12, 4): "https://download.pytorch.org/whl/cu124",
}

PYTORCH_CPU_INDEX_URL: str = "https://download.pytorch.org/whl/cpu"


# ---------------------------------------------------------------------------
# OS-family detection helper
# ---------------------------------------------------------------------------

_DEBIAN_LIKE = frozenset({"ubuntu", "debian", "linuxmint", "pop", "elementary", "kali"})
_RHEL_LIKE = frozenset({"centos", "rhel", "fedora", "rocky", "almalinux", "amazon"})
_SUSE_LIKE = frozenset({"suse", "opensuse", "sles"})
_ARCH_LIKE = frozenset({"arch", "manjaro", "endeavouros"})


def _detect_pkg_manager(profile: SystemProfile) -> str:
    """Return the likely package-manager command for the system.

    Falls back to ``"apt-get"`` when the OS cannot be identified.
    """
    os_lower = profile.os_name.lower()
    for name in _DEBIAN_LIKE:
        if name in os_lower:
            return "apt-get"
    for name in _RHEL_LIKE:
        if name in os_lower:
            return "yum"
    for name in _SUSE_LIKE:
        if name in os_lower:
            return "zypper"
    for name in _ARCH_LIKE:
        if name in os_lower:
            return "pacman -S"
    return "apt-get"


def _format_glibc_version(ver: Optional[Tuple[int, int]]) -> str:
    """Format a GLIBC version tuple as ``"X.Y"``."""
    if ver is None:
        return "unknown"
    return f"{ver[0]}.{ver[1]}"


def _format_cuda_version(ver: Optional[Tuple[int, int]]) -> str:
    """Format a CUDA version tuple as ``"X.Y"``."""
    if ver is None:
        return "unknown"
    return f"{ver[0]}.{ver[1]}"


# ---------------------------------------------------------------------------
# Per-rule suggestion generators
# ---------------------------------------------------------------------------


def _suggest_glibc_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for GLIBC version mismatches."""
    system_glibc = _format_glibc_version(profile.glibc_version)
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"Your system has GLIBC {system_glibc}, which is too old for {pkg_name}.",
        "",
        "Option 1 -- Install an older version of the package that supports your GLIBC:",
    ]

    if finding.package:
        lines.append(f"  pip install '{finding.package}<OLDER_VERSION'")
    else:
        lines.append("  pip install 'PACKAGE<OLDER_VERSION'")

    lines.extend([
        "",
        "Option 2 -- Upgrade your operating system to get a newer GLIBC:",
    ])
    pkg_mgr = _detect_pkg_manager(profile)
    if "apt" in pkg_mgr:
        lines.append("  sudo apt-get update && sudo apt-get dist-upgrade")
    elif "yum" in pkg_mgr:
        lines.append("  sudo yum update")
    elif "zypper" in pkg_mgr:
        lines.append("  sudo zypper update")
    else:
        lines.append(f"  sudo {pkg_mgr} --sysupgrade")

    lines.extend([
        "",
        "Option 3 -- Use a container with a newer base image:",
        "  docker run --rm -it python:3.12-bookworm pip install " + (finding.package or "PACKAGE"),
    ])

    return "\n".join(lines)


def _suggest_cuda_runtime_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for CUDA runtime mismatches."""
    system_cuda = _format_cuda_version(profile.cuda_runtime_version)
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"Your system has CUDA {system_cuda}, but {pkg_name} needs a different version.",
        "",
        "Option 1 -- Reinstall the package for your CUDA version:",
    ]

    if profile.cuda_runtime_version and profile.cuda_runtime_version in PYTORCH_INDEX_URLS:
        index_url = PYTORCH_INDEX_URLS[profile.cuda_runtime_version]
        if finding.package:
            lines.append(f"  pip install {finding.package} --index-url {index_url}")
        else:
            lines.append(f"  pip install PACKAGE --index-url {index_url}")
    elif profile.cuda_runtime_version is None:
        lines.append("  # No CUDA detected. Install the CPU-only version instead:")
        if finding.package:
            lines.append(
                f"  pip install {finding.package} --index-url {PYTORCH_CPU_INDEX_URL}"
            )
        else:
            lines.append(f"  pip install PACKAGE --index-url {PYTORCH_CPU_INDEX_URL}")
    else:
        cuda_ver = _format_cuda_version(profile.cuda_runtime_version)
        lines.append(
            f"  # Look for a wheel built for CUDA {cuda_ver} on the package's "
            f"documentation or PyPI page."
        )

    lines.extend([
        "",
        "Option 2 -- Install the matching CUDA toolkit:",
        "  # Visit https://developer.nvidia.com/cuda-toolkit-archive",
        "  # and install the version required by the package.",
    ])

    return "\n".join(lines)


def _suggest_cuda_lib_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for missing CUDA libraries."""
    pkg_mgr = _detect_pkg_manager(profile)
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"{pkg_name} requires CUDA math libraries that are not installed.",
        "",
        "Option 1 -- Install the CUDA toolkit:",
    ]

    if "apt" in pkg_mgr:
        lines.append("  sudo apt-get install nvidia-cuda-toolkit")
    elif "yum" in pkg_mgr:
        lines.append("  sudo yum install cuda")
    else:
        lines.append(f"  sudo {pkg_mgr} install cuda")

    lines.extend([
        "",
        "Option 2 -- Reinstall the package for your CUDA version:",
    ])

    if profile.cuda_runtime_version and profile.cuda_runtime_version in PYTORCH_INDEX_URLS:
        index_url = PYTORCH_INDEX_URLS[profile.cuda_runtime_version]
        if finding.package:
            lines.append(f"  pip install --force-reinstall {finding.package} --index-url {index_url}")
        else:
            lines.append(f"  pip install --force-reinstall PACKAGE --index-url {index_url}")
    else:
        lines.append("  pip install --force-reinstall PACKAGE  # use the correct --index-url for your CUDA version")

    return "\n".join(lines)


def _suggest_cudnn_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for missing cuDNN libraries."""
    pkg_mgr = _detect_pkg_manager(profile)
    cuda_ver = _format_cuda_version(profile.cuda_runtime_version)

    lines: List[str] = [
        "The cuDNN library is required for GPU-accelerated deep learning but is not installed.",
        "",
        "Option 1 -- Install cuDNN via your package manager:",
    ]

    if "apt" in pkg_mgr:
        lines.append("  sudo apt-get install libcudnn8 libcudnn8-dev")
    elif "yum" in pkg_mgr:
        lines.append("  sudo yum install libcudnn8 libcudnn8-devel")
    else:
        lines.append(f"  sudo {pkg_mgr} install libcudnn8")

    lines.extend([
        "",
        "Option 2 -- Install cuDNN from NVIDIA:",
        f"  # Visit https://developer.nvidia.com/cudnn and download the version for CUDA {cuda_ver}.",
    ])

    return "\n".join(lines)


def _suggest_cuda_driver_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for outdated GPU drivers."""
    pkg_mgr = _detect_pkg_manager(profile)
    driver_ver = profile.gpu_driver_version or "unknown"

    lines: List[str] = [
        f"Your GPU driver ({driver_ver}) is too old for the installed CUDA version.",
        "",
        "Option 1 -- Update your NVIDIA driver:",
    ]

    if "apt" in pkg_mgr:
        lines.extend([
            "  sudo apt-get update",
            "  sudo apt-get install --upgrade nvidia-driver-550",
        ])
    elif "yum" in pkg_mgr:
        lines.append("  sudo yum install nvidia-driver-latest-dkms")
    else:
        lines.append(f"  sudo {pkg_mgr} install nvidia-driver")

    lines.extend([
        "",
        "Option 2 -- Download the latest driver from NVIDIA:",
        "  # Visit https://www.nvidia.com/Download/index.aspx",
    ])

    return "\n".join(lines)


def _suggest_container_driver_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for container/host driver mismatches."""
    lines: List[str] = [
        "The NVIDIA driver inside your container does not match the host driver.",
        "",
        "Option 1 -- Use nvidia-container-toolkit (recommended):",
        "  # On the host, install the toolkit:",
        "  #   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html",
        "  # Then run your container with GPU access:",
        "  docker run --gpus all YOUR_IMAGE",
        "",
        "Option 2 -- Match CUDA versions:",
        "  # Use a container base image whose CUDA version is supported by your host driver.",
    ]

    if profile.gpu_driver_version:
        lines.append(f"  # Your host driver is {profile.gpu_driver_version}.")
    lines.append("  # Check compatibility at https://docs.nvidia.com/deploy/cuda-compatibility/")

    return "\n".join(lines)


def _suggest_python_abi_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for Python ABI mismatches."""
    py_ver = ".".join(str(v) for v in profile.python_version) if profile.python_version != (0, 0, 0) else "unknown"
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"{pkg_name} was compiled for a different Python version (you are running {py_ver}).",
        "",
        "Fix -- Reinstall the package for your current Python:",
    ]

    if finding.package:
        lines.append(f"  pip install --force-reinstall --no-cache-dir {finding.package}")
    else:
        lines.append("  pip install --force-reinstall --no-cache-dir PACKAGE")

    lines.extend([
        "",
        "If that fails, the package may not support your Python version yet.",
        "Check the package's PyPI page for compatible Python versions.",
    ])

    return "\n".join(lines)


def _suggest_numpy_abi_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for NumPy C API version mismatches."""
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"{pkg_name} was built against a different NumPy C API version.",
        "",
        "Option 1 -- Reinstall the package (will rebuild against current NumPy):",
    ]

    if finding.package:
        lines.append(f"  pip install --force-reinstall --no-cache-dir {finding.package}")
    else:
        lines.append("  pip install --force-reinstall --no-cache-dir PACKAGE")

    lines.extend([
        "",
        "Option 2 -- Install a compatible NumPy version:",
        "  pip install 'numpy<2'  # if the package was built for NumPy 1.x",
        "  pip install 'numpy>=2' # if the package was built for NumPy 2.x",
    ])

    return "\n".join(lines)


def _suggest_arch_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for platform/architecture mismatches."""
    arch = profile.architecture.value if profile.architecture else "unknown"
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"{pkg_name} was built for a different platform (your architecture: {arch}).",
        "",
        "Option 1 -- Install the correct wheel for your platform:",
    ]

    if finding.package:
        lines.append(
            f"  pip install --force-reinstall {finding.package} "
            f"--only-binary :all: --platform manylinux_2_17_{arch}"
        )
    else:
        lines.append(
            f"  pip install --force-reinstall PACKAGE "
            f"--only-binary :all: --platform manylinux_2_17_{arch}"
        )

    lines.extend([
        "",
        "Option 2 -- Build from source:",
    ])

    if finding.package:
        lines.append(f"  pip install --no-binary :all: {finding.package}")
    else:
        lines.append("  pip install --no-binary :all: PACKAGE")

    return "\n".join(lines)


def _suggest_missing_lib_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest installing missing shared libraries."""
    pkg_mgr = _detect_pkg_manager(profile)

    lines: List[str] = [
        "One or more shared libraries required by the package are not installed.",
        "",
        "Install the missing libraries with your system package manager:",
    ]

    if "apt" in pkg_mgr:
        lines.extend([
            "  sudo apt-get update",
            "  sudo apt-get install LIBRARY_PACKAGE_NAME",
            "",
            "  # To find which package provides a specific .so file:",
            "  apt-file search LIBRARY_NAME.so",
        ])
    elif "yum" in pkg_mgr:
        lines.extend([
            "  sudo yum install LIBRARY_PACKAGE_NAME",
            "",
            "  # To find which package provides a specific .so file:",
            "  yum provides '*/LIBRARY_NAME.so'",
        ])
    elif "zypper" in pkg_mgr:
        lines.extend([
            "  sudo zypper install LIBRARY_PACKAGE_NAME",
            "",
            "  # To find which package provides a specific .so file:",
            "  zypper search --provides LIBRARY_NAME.so",
        ])
    elif "pacman" in pkg_mgr:
        lines.extend([
            "  sudo pacman -S LIBRARY_PACKAGE_NAME",
            "",
            "  # To find which package provides a specific .so file:",
            "  pacman -F LIBRARY_NAME.so",
        ])
    else:
        lines.append(f"  sudo {pkg_mgr} install LIBRARY_PACKAGE_NAME")

    return "\n".join(lines)


def _suggest_illegal_instruction_fix(finding: Finding, profile: SystemProfile) -> str:
    """Suggest remediation for CPU instruction set mismatches."""
    cpu_model = profile.cpu_model or "your CPU"
    pkg_name = finding.package or "the package"

    lines: List[str] = [
        f"{pkg_name} uses CPU instructions not supported by {cpu_model}.",
    ]

    if not profile.has_avx2:
        lines.append("  Your CPU does NOT support AVX2 instructions.")
    if not profile.has_avx512:
        lines.append("  Your CPU does NOT support AVX-512 instructions.")

    lines.extend([
        "",
        "Option 1 -- Install a version built without advanced CPU instructions:",
    ])

    if finding.package:
        lines.append(
            f"  pip install --force-reinstall --no-cache-dir {finding.package}"
        )
        lines.append(
            "  # Look for a build without SIMD optimizations on the package's releases page."
        )
    else:
        lines.append("  pip install --force-reinstall --no-cache-dir PACKAGE")

    lines.extend([
        "",
        "Option 2 -- Run on a machine with a newer CPU that supports AVX2/AVX-512.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule-ID to handler mapping
# ---------------------------------------------------------------------------

_SUGGESTION_HANDLERS: Dict[str, object] = {
    "GLIBC_VERSION_MISMATCH": _suggest_glibc_fix,
    "CUDA_RUNTIME_MISMATCH": _suggest_cuda_runtime_fix,
    "CUDA_LIB_MISSING": _suggest_cuda_lib_fix,
    "CUDNN_VERSION_MISMATCH": _suggest_cudnn_fix,
    "CUDA_DRIVER_TOO_OLD": _suggest_cuda_driver_fix,
    "CONTAINER_DRIVER_MISMATCH": _suggest_container_driver_fix,
    "PYTHON_ABI_MISMATCH": _suggest_python_abi_fix,
    "NUMPY_ABI_MISMATCH": _suggest_numpy_abi_fix,
    "ARCH_MISMATCH": _suggest_arch_fix,
    "ILLEGAL_INSTRUCTION_RISK": _suggest_illegal_instruction_fix,
    "MISSING_SHARED_LIB": _suggest_missing_lib_fix,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggest_fix(finding: Finding, profile: SystemProfile) -> str:
    """Generate a context-aware fix suggestion for a finding.

    The suggestion is tailored to the user's system (OS, CUDA version,
    architecture, etc.) and includes copy-pasteable commands wherever
    possible.

    Parameters
    ----------
    finding:
        The diagnostic finding to remediate.
    profile:
        The system profile describing the target machine.

    Returns
    -------
    str
        A multi-line string with numbered options and shell commands.
        If no specific handler is registered for the finding's
        ``rule_id``, a generic suggestion based on the finding's own
        ``suggestion`` field is returned.
    """
    if not isinstance(finding, Finding):
        raise TypeError(
            f"finding must be a Finding instance, got {type(finding).__name__}"
        )
    if not isinstance(profile, SystemProfile):
        raise TypeError(
            f"profile must be a SystemProfile instance, got {type(profile).__name__}"
        )

    handler = _SUGGESTION_HANDLERS.get(finding.rule_id)
    if handler is not None:
        # All handlers have the same (Finding, SystemProfile) -> str signature.
        return handler(finding, profile)  # type: ignore[operator]

    # Fallback: use the finding's own suggestion or a generic message.
    if finding.suggestion:
        return finding.suggestion

    return (
        f"No specific fix suggestion is available for rule '{finding.rule_id}'. "
        f"Check the package documentation or open an issue with the maintainers."
    )
