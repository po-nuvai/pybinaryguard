"""Build toolchain probe — detects compilers and build tools on the system."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

from .base import ProbeBase


class ToolchainProbe(ProbeBase):
    """Detect available build toolchain (gcc, g++, cmake, make, rustc, etc.).

    This is critical for detecting packages built from source (sdist)
    and validating that the compiler used is compatible with system binaries.
    """

    name = "toolchain"

    def collect(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        gcc = self._detect_tool("gcc", r"(\d+\.\d+\.\d+)")
        gpp = self._detect_tool("g++", r"(\d+\.\d+\.\d+)")
        clang = self._detect_tool("clang", r"(\d+\.\d+\.\d+)")
        cmake = self._detect_tool("cmake", r"(\d+\.\d+\.\d+)")
        make = self._detect_tool("make", r"(\d+\.\d+)")
        rustc = self._detect_tool("rustc", r"(\d+\.\d+\.\d+)")

        toolchain: Dict[str, Optional[str]] = {
            "gcc": gcc,
            "gpp": gpp,
            "clang": clang,
            "cmake": cmake,
            "make": make,
            "rustc": rustc,
        }

        data["toolchain_versions"] = {
            k: v for k, v in toolchain.items() if v is not None
        }

        # Detect the default C compiler
        cc = os.environ.get("CC", "")
        cxx = os.environ.get("CXX", "")
        data["default_cc"] = cc if cc else ("gcc" if gcc else "clang" if clang else "")
        data["default_cxx"] = cxx if cxx else ("g++" if gpp else "clang++" if clang else "")

        # Check if build-essential / dev headers are available
        data["has_build_tools"] = gcc is not None or clang is not None
        data["has_python_dev_headers"] = self._has_python_headers()

        return data

    @staticmethod
    def _detect_tool(tool: str, version_pattern: str) -> Optional[str]:
        """Run ``tool --version`` and extract version string."""
        path = shutil.which(tool)
        if not path:
            return None
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            match = re.search(version_pattern, output)
            if match:
                return match.group(1)
            return "unknown"
        except (subprocess.TimeoutExpired, OSError):
            return None

    @staticmethod
    def _has_python_headers() -> bool:
        """Check if Python development headers are installed."""
        try:
            import sysconfig
            include_dir = sysconfig.get_path("include")
            if include_dir and os.path.isfile(os.path.join(include_dir, "Python.h")):
                return True
        except Exception:
            pass
        return False
