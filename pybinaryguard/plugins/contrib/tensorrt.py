"""TensorRT plugin for PyBinaryGuard.

Provides rules that check TensorRT version compatibility with installed
deep-learning frameworks and detect serialised engine files built for a
different GPU architecture.

Provides
--------
- **TensorRTVersionRule** -- Checks that the installed TensorRT version
  is compatible with the installed versions of TensorFlow, PyTorch, and
  ONNX Runtime.
- **TensorRTEngineMismatchRule** -- Scans for ``.engine`` / ``.trt``
  files that were serialised for a different GPU compute capability than
  the current device.
"""

from __future__ import annotations

import logging
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule

if TYPE_CHECKING:
    from pybinaryguard.plugins.hooks import HookRegistry

logger = logging.getLogger(__name__)

# -- Version compatibility tables -------------------------------------------

# Minimum TensorRT version required by framework major.minor releases.
# Format: {("framework", (major, minor)): (trt_min_major, trt_min_minor)}
_FRAMEWORK_TRT_COMPAT: Dict[Tuple[str, Tuple[int, int]], Tuple[int, int]] = {
    # PyTorch (torch.tensorrt / torch2trt)
    ("pytorch", (2, 0)): (8, 5),
    ("pytorch", (2, 1)): (8, 6),
    ("pytorch", (2, 2)): (8, 6),
    ("pytorch", (2, 3)): (10, 0),
    ("pytorch", (2, 4)): (10, 0),
    ("pytorch", (2, 5)): (10, 3),
    # TensorFlow
    ("tensorflow", (2, 12)): (8, 5),
    ("tensorflow", (2, 13)): (8, 6),
    ("tensorflow", (2, 14)): (8, 6),
    ("tensorflow", (2, 15)): (8, 6),
    ("tensorflow", (2, 16)): (8, 6),
    # ONNX Runtime
    ("onnxruntime", (1, 15)): (8, 6),
    ("onnxruntime", (1, 16)): (8, 6),
    ("onnxruntime", (1, 17)): (8, 6),
    ("onnxruntime", (1, 18)): (10, 0),
    ("onnxruntime", (1, 19)): (10, 0),
}

# Package names that correspond to each framework key above.
_FRAMEWORK_PACKAGE_NAMES: Dict[str, List[str]] = {
    "pytorch": ["torch"],
    "tensorflow": ["tensorflow", "tensorflow-gpu", "tf-nightly"],
    "onnxruntime": ["onnxruntime", "onnxruntime-gpu"],
}


# -- Helpers ----------------------------------------------------------------


def _get_tensorrt_version() -> Optional[Tuple[int, int, int]]:
    """Attempt to determine the installed TensorRT version.

    Tries the ``tensorrt`` Python package first, then falls back to
    parsing ``libnvinfer.so`` via ``ldconfig -p``.

    Returns:
        A ``(major, minor, patch)`` tuple, or ``None`` if TensorRT is
        not detected.
    """
    # Method 1: Python package
    try:
        import tensorrt  # type: ignore[import-untyped]
        ver = getattr(tensorrt, "__version__", None)
        if ver:
            parts = ver.split(".")
            if len(parts) >= 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]), 0)
    except ImportError:
        pass
    except Exception:
        logger.debug("Failed to read tensorrt.__version__", exc_info=True)

    # Method 2: Parse libnvinfer soname from ldconfig
    try:
        import subprocess
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import re
            match = re.search(r"libnvinfer\.so\.(\d+)\.(\d+)\.(\d+)", result.stdout)
            if match:
                return (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            # Fallback: just major version from soname
            match = re.search(r"libnvinfer\.so\.(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), 0, 0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def _parse_package_version(version_str: str) -> Optional[Tuple[int, int]]:
    """Parse a package version string into (major, minor).

    Handles common formats like ``2.1.0``, ``2.1.0+cu118``,
    ``2.1.0.post1``, etc.

    Args:
        version_str: The raw version string.

    Returns:
        A ``(major, minor)`` tuple, or ``None`` on parse failure.
    """
    import re
    match = re.match(r"(\d+)\.(\d+)", version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# -- Rules ------------------------------------------------------------------


class TensorRTVersionRule(Rule):
    """Check TensorRT version compatibility with installed frameworks.

    Verifies that the installed TensorRT version meets the minimum
    requirements of installed deep-learning frameworks (PyTorch,
    TensorFlow, ONNX Runtime).
    """

    rule_id = "TENSORRT_VERSION_COMPAT"
    description = (
        "Verify that the installed TensorRT version is compatible with "
        "installed deep-learning frameworks."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applies when a GPU is available."""
        return profile.gpu_available

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Check TensorRT version against framework requirements."""
        findings: List[Finding] = []

        trt_version = _get_tensorrt_version()
        if trt_version is None:
            # TensorRT not installed -- nothing to check.
            return findings

        trt_major, trt_minor = trt_version[0], trt_version[1]

        # Build a lookup of installed packages for quick access.
        installed: Dict[str, str] = {}
        for pkg in packages:
            installed[pkg.package_name.lower()] = pkg.package_version

        for framework_key, package_names in _FRAMEWORK_PACKAGE_NAMES.items():
            for pkg_name in package_names:
                version_str = installed.get(pkg_name.lower())
                if version_str is None:
                    continue

                fw_version = _parse_package_version(version_str)
                if fw_version is None:
                    continue

                required = _FRAMEWORK_TRT_COMPAT.get((framework_key, fw_version))
                if required is None:
                    # No compatibility data for this version -- skip.
                    continue

                req_major, req_minor = required
                if trt_major < req_major or (trt_major == req_major and trt_minor < req_minor):
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=(
                            f"TensorRT {trt_major}.{trt_minor} may be "
                            f"incompatible with {pkg_name} {version_str}"
                        ),
                        explanation=(
                            f"{pkg_name}=={version_str} requires TensorRT >= "
                            f"{req_major}.{req_minor}, but TensorRT "
                            f"{trt_major}.{trt_minor}.{trt_version[2]} is installed."
                        ),
                        technical_detail=(
                            f"Framework: {pkg_name}=={version_str}, "
                            f"TensorRT: {trt_major}.{trt_minor}.{trt_version[2]}, "
                            f"Required: >= {req_major}.{req_minor}"
                        ),
                        suggestion=(
                            f"Upgrade TensorRT to version {req_major}.{req_minor} "
                            f"or later, or downgrade {pkg_name} to a version "
                            f"compatible with TensorRT {trt_major}.{trt_minor}."
                        ),
                        package=pkg_name,
                        package_version=version_str,
                    ))
                else:
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        severity=Severity.PASSED,
                        title=(
                            f"TensorRT {trt_major}.{trt_minor} is compatible "
                            f"with {pkg_name} {version_str}"
                        ),
                        explanation=(
                            f"TensorRT {trt_major}.{trt_minor}.{trt_version[2]} "
                            f"meets the minimum requirement "
                            f"(>= {req_major}.{req_minor}) for "
                            f"{pkg_name}=={version_str}."
                        ),
                        package=pkg_name,
                        package_version=version_str,
                    ))
                # Only check the first matching package name per framework.
                break

        return findings


class TensorRTEngineMismatchRule(Rule):
    """Detect serialised TensorRT engine files built for a different GPU.

    TensorRT engine files (``.engine``, ``.trt``, ``.plan``) are
    serialised for a specific GPU architecture and are not portable.
    Loading an engine built for a different compute capability will fail
    at runtime.

    This rule scans site-packages directories for engine files and
    performs a best-effort check of their target compute capability by
    reading the header bytes.
    """

    rule_id = "TENSORRT_ENGINE_MISMATCH"
    description = (
        "Detect TensorRT engine files serialised for a different GPU "
        "compute capability."
    )

    # TensorRT serialised engines have a small header.  The exact layout
    # varies by TensorRT version, but a common pattern is that the magic
    # bytes start with "ptrt" (0x70747274) followed by version info.
    # This is a best-effort heuristic.
    _ENGINE_EXTENSIONS = (".engine", ".trt", ".plan")
    _MAX_SCAN_BYTES = 256

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applies when GPU compute capability is known."""
        return (
            profile.gpu_available
            and profile.gpu_compute_capability is not None
        )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Scan for engine files in package install paths."""
        findings: List[Finding] = []

        if profile.gpu_compute_capability is None:
            return findings

        current_cc = profile.gpu_compute_capability

        for pkg in packages:
            if not pkg.install_path or not os.path.isdir(pkg.install_path):
                continue

            engine_files = self._find_engine_files(pkg.install_path)
            for engine_path in engine_files:
                engine_cc = self._read_engine_compute_capability(engine_path)
                if engine_cc is not None and engine_cc != current_cc:
                    findings.append(Finding(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        title=(
                            f"TensorRT engine built for different GPU: "
                            f"{os.path.basename(engine_path)}"
                        ),
                        explanation=(
                            f"Engine file {engine_path} appears to be "
                            f"serialised for compute capability "
                            f"{engine_cc[0]}.{engine_cc[1]}, but the "
                            f"current GPU has compute capability "
                            f"{current_cc[0]}.{current_cc[1]}. "
                            f"Loading this engine will fail at runtime."
                        ),
                        technical_detail=(
                            f"Engine CC: {engine_cc[0]}.{engine_cc[1]}, "
                            f"GPU CC: {current_cc[0]}.{current_cc[1]}, "
                            f"File: {engine_path}"
                        ),
                        suggestion=(
                            "Re-serialise the TensorRT engine on this GPU, "
                            "or use the ONNX model to rebuild the engine: "
                            "trtexec --onnx=model.onnx --saveEngine=model.engine"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    ))

        return findings

    def _find_engine_files(self, directory: str) -> List[str]:
        """Recursively find TensorRT engine files in a directory.

        Limits the search to 3 directory levels deep to avoid traversing
        overly large trees.
        """
        engine_files: List[str] = []
        try:
            for root, dirs, files in os.walk(directory):
                # Limit depth to avoid excessive traversal.
                depth = root[len(directory):].count(os.sep)
                if depth >= 3:
                    dirs.clear()
                    continue
                for fname in files:
                    if any(fname.endswith(ext) for ext in self._ENGINE_EXTENSIONS):
                        engine_files.append(os.path.join(root, fname))
        except (PermissionError, OSError):
            pass
        return engine_files

    @staticmethod
    def _read_engine_compute_capability(
        path: str,
    ) -> Optional[Tuple[int, int]]:
        """Attempt to read the compute capability from a TensorRT engine file.

        This is a best-effort heuristic.  TensorRT engine files do not
        have a publicly documented header format, but engine files
        serialised by common TensorRT versions embed the compute
        capability as two bytes in the header region.

        We look for the ``ptrt`` magic sequence and attempt to extract
        CC from known offsets.  Returns ``None`` when the format is
        unrecognised.
        """
        try:
            with open(path, "rb") as fh:
                header = fh.read(256)
        except (FileNotFoundError, PermissionError, OSError):
            return None

        if len(header) < 32:
            return None

        # Look for the "ptrt" magic (0x70747274)
        magic_offset = header.find(b"ptrt")
        if magic_offset < 0:
            # Try little-endian uint32 magic
            magic_offset = header.find(b"trtp")
            if magic_offset < 0:
                return None

        # In several TensorRT versions, the compute capability is stored
        # as two uint8 values at offset magic+20 and magic+21 (major, minor).
        cc_offset = magic_offset + 20
        if cc_offset + 2 > len(header):
            return None

        cc_major = header[cc_offset]
        cc_minor = header[cc_offset + 1]

        # Sanity check: compute capability major should be 2-12
        if 2 <= cc_major <= 12 and 0 <= cc_minor <= 9:
            return (cc_major, cc_minor)

        return None


# -- Plugin entry point -----------------------------------------------------


def register(registry: HookRegistry) -> None:
    """Register TensorRT rules if TensorRT is available.

    This function is called by the plugin loader.  It activates when
    either the ``tensorrt`` Python package is importable or
    ``libnvinfer`` is found in the system library cache.
    """
    trt_version = _get_tensorrt_version()
    if trt_version is None:
        logger.debug("TensorRT plugin: TensorRT not detected; not activating")
        return

    registry.add_rule(TensorRTVersionRule())
    registry.add_rule(TensorRTEngineMismatchRule())
    logger.info(
        "TensorRT plugin activated (version %d.%d.%d)",
        trt_version[0],
        trt_version[1],
        trt_version[2],
    )
