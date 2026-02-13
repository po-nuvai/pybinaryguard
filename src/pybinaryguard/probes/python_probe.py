"""Probe for Python interpreter information."""

from __future__ import annotations

import sys
import sysconfig
from typing import Any, Dict, Tuple

from .base import ProbeBase


class PythonProbe(ProbeBase):
    """Collects information about the running Python interpreter.

    Gathered fields:
    - ``python_version``: 3-tuple of (major, minor, micro)
    - ``python_abi_tag``: e.g. ``"cp312-cp312-linux_x86_64"``
    - ``python_implementation``: e.g. ``"cpython"``, ``"pypy"``
    - ``python_executable``: absolute path to the interpreter
    - ``stable_abi_supported``: whether the stable ABI (abi3) is supported
    - ``python_debug_build``: whether this is a debug build
    """

    name = "python"

    def collect(self) -> Dict[str, Any]:
        """Collect Python interpreter details using only stdlib modules."""
        data: Dict[str, Any] = {}

        data["python_version"] = self._get_version()
        data["python_abi_tag"] = self._get_abi_tag()
        data["python_implementation"] = self._get_implementation()
        data["python_executable"] = self._get_executable()
        data["stable_abi_supported"] = self._check_stable_abi()
        data["python_debug_build"] = self._check_debug_build()

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_version() -> Tuple[int, int, int]:
        """Return the Python version as a (major, minor, micro) tuple."""
        try:
            return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
        except Exception:
            return (0, 0, 0)

    @staticmethod
    def _get_abi_tag() -> str:
        """Build the ABI tag from sysconfig variables.

        Falls back to a constructed tag from ``sys.implementation`` when
        sysconfig does not provide ``SOABI``.
        """
        try:
            soabi = sysconfig.get_config_var("SOABI")
            if soabi:
                return str(soabi)

            # Fallback: build from implementation info
            impl = sys.implementation.name[0:2]  # "cp" for cpython
            ver = f"{sys.version_info.major}{sys.version_info.minor}"
            tag = f"{impl}{ver}"

            # Append debug/unicode suffixes for older Python builds
            abiflags = sysconfig.get_config_var("abiflags") or ""
            if abiflags:
                tag += abiflags

            return tag
        except Exception:
            return ""

    @staticmethod
    def _get_implementation() -> str:
        """Return a normalised implementation name (lowercase)."""
        try:
            return sys.implementation.name.lower()
        except Exception:
            return "cpython"

    @staticmethod
    def _get_executable() -> str:
        """Return the absolute path to the Python executable."""
        try:
            return sys.executable or ""
        except Exception:
            return ""

    @staticmethod
    def _check_stable_abi() -> bool:
        """Determine whether the stable ABI (abi3) is supported.

        The stable ABI was introduced in CPython 3.2.  For other
        implementations it is generally not available.
        """
        try:
            if sys.implementation.name.lower() != "cpython":
                return False
            return sys.version_info >= (3, 2)
        except Exception:
            return False

    @staticmethod
    def _check_debug_build() -> bool:
        """Detect whether the interpreter was compiled in debug mode.

        Checks multiple indicators:
        1. ``sys.flags.debug``
        2. ``Py_DEBUG`` sysconfig variable
        3. ``abiflags`` containing ``"d"``
        4. Pointer size heuristic (debug builds often have larger objects)
           -- this is a weak signal and is used only as a last resort.
        """
        try:
            # Primary: interpreter flag
            if sys.flags.debug:
                return True

            # Secondary: sysconfig variable
            py_debug = sysconfig.get_config_var("Py_DEBUG")
            if py_debug and int(py_debug):
                return True

            # Tertiary: abiflags
            abiflags = sysconfig.get_config_var("abiflags") or ""
            if "d" in abiflags:
                return True

            return False
        except Exception:
            return False
