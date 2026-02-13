"""Pre-install compatibility simulator.

Predicts whether a package will work on the current system BEFORE
``pip install``.  Parses wheel filenames / version specifiers and
checks against the live SystemProfile.

No network calls — works fully offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pybinaryguard.agent.tool_interface import AgentSimulateResult


# ---------------------------------------------------------------------------
# Wheel filename parser
# ---------------------------------------------------------------------------

# Pattern: name-version(-build)?-python-abi-platform.whl
_WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9]([A-Za-z0-9._]*[A-Za-z0-9])?)"
    r"-(?P<version>[^-]+)"
    r"(-(?P<build>\d[^-]*))?"
    r"-(?P<python>[^-]+)"
    r"-(?P<abi>[^-]+)"
    r"-(?P<platform>[^-]+)"
    r"\.whl$"
)

# CUDA variant in version: +cu118, +cu121, +cu124
_CUDA_VARIANT_RE = re.compile(r"\+cu(\d{2,3})")

# manylinux tag: manylinux_2_17_x86_64 or manylinux2014_x86_64
_MANYLINUX_RE = re.compile(r"manylinux_(\d+)_(\d+)_(\w+)")
_MANYLINUX_LEGACY = {
    "manylinux1": ((2, 5), None),
    "manylinux2010": ((2, 12), None),
    "manylinux2014": ((2, 17), None),
}

# Architecture mapping from platform tag
_PLATFORM_ARCH = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "armv7l": "armv7l",
    "i686": "i686",
    "ppc64le": "ppc64le",
    "s390x": "s390x",
}


@dataclass
class ParsedWheel:
    """Parsed wheel filename components."""

    name: str
    version: str
    python_tag: str  # e.g. "cp312"
    abi_tag: str     # e.g. "cp312"
    platform_tag: str  # e.g. "manylinux_2_17_x86_64"
    cuda_variant: Optional[Tuple[int, int]] = None  # (12, 4) from +cu124
    required_glibc: Optional[Tuple[int, int]] = None
    target_arch: Optional[str] = None


def _parse_wheel_filename(filename: str) -> Optional[ParsedWheel]:
    """Parse a wheel filename into structured components."""
    m = _WHEEL_RE.match(filename)
    if not m:
        return None

    name = m.group("name")
    version = m.group("version")
    python_tag = m.group("python")
    abi_tag = m.group("abi")
    platform_tag = m.group("platform")

    # Extract CUDA variant from version
    cuda_variant = None
    cuda_m = _CUDA_VARIANT_RE.search(version)
    if cuda_m:
        cu_str = cuda_m.group(1)
        if len(cu_str) == 2:
            cuda_variant = (int(cu_str[0]), int(cu_str[1]))
        elif len(cu_str) == 3:
            cuda_variant = (int(cu_str[:2]), int(cu_str[2]))

    # Parse manylinux GLIBC requirement
    required_glibc = None
    target_arch = None
    ml_m = _MANYLINUX_RE.search(platform_tag)
    if ml_m:
        required_glibc = (int(ml_m.group(1)), int(ml_m.group(2)))
        target_arch = _PLATFORM_ARCH.get(ml_m.group(3))
    else:
        # Check legacy tags
        for prefix, (glibc, _) in _MANYLINUX_LEGACY.items():
            if platform_tag.startswith(prefix):
                required_glibc = glibc
                # Try to extract arch from rest of tag
                rest = platform_tag[len(prefix):]
                if rest.startswith("_"):
                    rest = rest[1:]
                target_arch = _PLATFORM_ARCH.get(rest)
                break

    # If no manylinux, check for direct arch in platform
    if target_arch is None:
        for arch_str, arch_val in _PLATFORM_ARCH.items():
            if arch_str in platform_tag.lower():
                target_arch = arch_val
                break

    return ParsedWheel(
        name=name,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        cuda_variant=cuda_variant,
        required_glibc=required_glibc,
        target_arch=target_arch,
    )


def _parse_version_spec(spec: str) -> Tuple[str, str, Optional[Tuple[int, int]]]:
    """Parse 'torch==2.4.0+cu124' into (name, version, cuda_variant)."""
    # Split on version operators
    for op in ("==", ">=", "<=", "!=", "~=", ">", "<"):
        if op in spec:
            name, version = spec.split(op, 1)
            cuda_variant = None
            cuda_m = _CUDA_VARIANT_RE.search(version)
            if cuda_m:
                cu_str = cuda_m.group(1)
                if len(cu_str) == 2:
                    cuda_variant = (int(cu_str[0]), int(cu_str[1]))
                elif len(cu_str) == 3:
                    cuda_variant = (int(cu_str[:2]), int(cu_str[2]))
            return name.strip(), version.strip(), cuda_variant

    return spec.strip(), "", None


# ---------------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------------

def _check_python_compat(
    python_tag: str,
    system_python: Tuple[int, int, int],
) -> Optional[Dict[str, object]]:
    """Check Python version compatibility from wheel tag."""
    if python_tag in ("py3", "py2.py3", "none"):
        return None  # Universal wheel

    # Parse cp312 -> (3, 12)
    m = re.match(r"cp(\d)(\d+)", python_tag)
    if not m:
        return None

    wheel_major = int(m.group(1))
    wheel_minor = int(m.group(2))

    if (wheel_major, wheel_minor) != (system_python[0], system_python[1]):
        return {
            "type": "python_version_mismatch",
            "severity": "critical",
            "message": (
                f"Wheel requires Python {wheel_major}.{wheel_minor} "
                f"but system has {system_python[0]}.{system_python[1]}"
            ),
            "wheel_python": f"{wheel_major}.{wheel_minor}",
            "system_python": f"{system_python[0]}.{system_python[1]}",
        }
    return None


def _check_arch_compat(
    target_arch: Optional[str],
    system_arch: str,
) -> Optional[Dict[str, object]]:
    """Check architecture compatibility."""
    if target_arch is None:
        return None
    if target_arch == system_arch:
        return None

    return {
        "type": "architecture_mismatch",
        "severity": "critical",
        "message": (
            f"Wheel built for {target_arch} but system is {system_arch}"
        ),
        "wheel_arch": target_arch,
        "system_arch": system_arch,
    }


def _check_glibc_compat(
    required_glibc: Optional[Tuple[int, int]],
    system_glibc: Optional[Tuple[int, int]],
) -> Optional[Dict[str, object]]:
    """Check GLIBC version compatibility."""
    if required_glibc is None or system_glibc is None:
        return None

    if required_glibc > system_glibc:
        return {
            "type": "glibc_too_old",
            "severity": "critical",
            "message": (
                f"Wheel requires GLIBC {required_glibc[0]}.{required_glibc[1]} "
                f"but system has {system_glibc[0]}.{system_glibc[1]}"
            ),
            "required": f"{required_glibc[0]}.{required_glibc[1]}",
            "system": f"{system_glibc[0]}.{system_glibc[1]}",
        }
    return None


def _check_cuda_compat(
    cuda_variant: Optional[Tuple[int, int]],
    system_cuda: Optional[Tuple[int, int]],
    gpu_available: bool,
) -> Optional[Dict[str, object]]:
    """Check CUDA version compatibility."""
    if cuda_variant is None:
        return None  # Not a CUDA wheel

    if not gpu_available:
        return {
            "type": "no_gpu",
            "severity": "warning",
            "message": (
                f"Wheel is CUDA-enabled (cu{cuda_variant[0]}{cuda_variant[1]}) "
                f"but no GPU detected on this system"
            ),
            "wheel_cuda": f"{cuda_variant[0]}.{cuda_variant[1]}",
        }

    if system_cuda is None:
        return {
            "type": "no_cuda_runtime",
            "severity": "critical",
            "message": (
                f"Wheel requires CUDA {cuda_variant[0]}.{cuda_variant[1]} "
                f"but no CUDA runtime detected"
            ),
            "wheel_cuda": f"{cuda_variant[0]}.{cuda_variant[1]}",
        }

    # CUDA major version must match, minor can be >= wheel's
    if cuda_variant[0] != system_cuda[0]:
        return {
            "type": "cuda_major_mismatch",
            "severity": "critical",
            "message": (
                f"Wheel built for CUDA {cuda_variant[0]}.{cuda_variant[1]} "
                f"but system has CUDA {system_cuda[0]}.{system_cuda[1]}"
            ),
            "wheel_cuda": f"{cuda_variant[0]}.{cuda_variant[1]}",
            "system_cuda": f"{system_cuda[0]}.{system_cuda[1]}",
        }

    if cuda_variant[1] > system_cuda[1]:
        return {
            "type": "cuda_minor_mismatch",
            "severity": "warning",
            "message": (
                f"Wheel built for CUDA {cuda_variant[0]}.{cuda_variant[1]} "
                f"but system has {system_cuda[0]}.{system_cuda[1]} — "
                f"minor version mismatch may cause issues"
            ),
            "wheel_cuda": f"{cuda_variant[0]}.{cuda_variant[1]}",
            "system_cuda": f"{system_cuda[0]}.{system_cuda[1]}",
        }

    return None


def _check_musl_compat(
    platform_tag: str,
    system_glibc: Optional[Tuple[int, int]],
    system_musl: Optional[Tuple[int, int]],
) -> Optional[Dict[str, object]]:
    """Check musl vs glibc compatibility."""
    is_musllinux = "musllinux" in platform_tag
    is_manylinux = "manylinux" in platform_tag

    if is_manylinux and system_musl is not None and system_glibc is None:
        return {
            "type": "glibc_on_musl",
            "severity": "critical",
            "message": (
                "Wheel is glibc-linked (manylinux) but system uses musl "
                "(Alpine Linux). Build from source instead."
            ),
            "fix_hint": "pip install --no-binary :all: <package>",
        }

    if is_musllinux and system_musl is None:
        return {
            "type": "musl_on_glibc",
            "severity": "critical",
            "message": (
                "Wheel is musl-linked (musllinux) but system uses glibc."
            ),
        }

    return None


# ---------------------------------------------------------------------------
# Main simulate function
# ---------------------------------------------------------------------------

def simulate(package_spec: str) -> AgentSimulateResult:
    """Run pre-install compatibility simulation.

    Parameters
    ----------
    package_spec:
        Package name, version pin, or wheel filename.

    Returns
    -------
    AgentSimulateResult
        Structured prediction result.
    """
    from pybinaryguard.scanner import Scanner

    # Collect system profile
    scanner = Scanner(timeout=10.0)
    profile = scanner.get_profile()

    blockers: List[Dict[str, object]] = []
    warnings: List[Dict[str, object]] = []
    parsed_tags: Optional[Dict[str, str]] = None
    confidence = 0.5  # Base confidence for name-only

    # Try parsing as wheel filename
    wheel = _parse_wheel_filename(package_spec)

    if wheel is not None:
        confidence = 0.95  # Wheel filenames are highly informative
        parsed_tags = {
            "name": wheel.name,
            "version": wheel.version,
            "python": wheel.python_tag,
            "abi": wheel.abi_tag,
            "platform": wheel.platform_tag,
        }
        if wheel.cuda_variant:
            parsed_tags["cuda"] = f"{wheel.cuda_variant[0]}.{wheel.cuda_variant[1]}"

        # Run all checks
        checks = [
            _check_python_compat(wheel.python_tag, profile.python_version),
            _check_arch_compat(wheel.target_arch, profile.architecture.value),
            _check_glibc_compat(wheel.required_glibc, profile.glibc_version),
            _check_cuda_compat(
                wheel.cuda_variant,
                profile.cuda_runtime_version,
                profile.gpu_available,
            ),
            _check_musl_compat(
                wheel.platform_tag,
                profile.glibc_version,
                profile.musl_version,
            ),
        ]

        for check_result in checks:
            if check_result is None:
                continue
            if check_result["severity"] == "critical":
                blockers.append(check_result)
            else:
                warnings.append(check_result)

    else:
        # Parse as name or name==version
        name, version, cuda_variant = _parse_version_spec(package_spec)
        parsed_tags = {"name": name}
        if version:
            parsed_tags["version"] = version
            confidence = 0.6

        if cuda_variant:
            parsed_tags["cuda"] = f"{cuda_variant[0]}.{cuda_variant[1]}"
            confidence = 0.7

            cuda_check = _check_cuda_compat(
                cuda_variant,
                profile.cuda_runtime_version,
                profile.gpu_available,
            )
            if cuda_check:
                if cuda_check["severity"] == "critical":
                    blockers.append(cuda_check)
                else:
                    warnings.append(cuda_check)

    # Determine overall prediction
    predicted_compatible = len(blockers) == 0

    # Risk level
    if blockers:
        risk_level = "critical" if len(blockers) >= 2 else "high"
    elif warnings:
        risk_level = "medium" if len(warnings) >= 2 else "low"
    else:
        risk_level = "none"

    return AgentSimulateResult(
        package_spec=package_spec,
        predicted_compatible=predicted_compatible,
        confidence=confidence,
        risk_level=risk_level,
        warnings=warnings,
        blockers=blockers,
        parsed_tags=parsed_tags,
    )
