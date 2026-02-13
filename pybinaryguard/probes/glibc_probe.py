"""Probe for GLIBC / musl libc information."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

from .base import ProbeBase


class GlibcProbe(ProbeBase):
    """Collects C library version information (GLIBC or musl).

    Detection strategy for GLIBC:
    1. ``os.confstr("CS_GNU_LIBC_VERSION")`` -- fast and authoritative.
    2. ctypes fallback: load ``libc.so.6`` and call ``gnu_get_libc_version()``.

    Detection strategy for musl:
    1. Scan ``/proc/self/maps`` for paths containing ``musl``.
    2. Fall back to running ``ldd --version`` and inspecting stderr for
       the ``musl`` banner.
    """

    name = "glibc"

    def collect(self) -> Dict[str, Any]:
        """Return ``glibc_version`` and/or ``musl_version``."""
        data: Dict[str, Any] = {}

        # Try musl first -- if the system is musl-based there is no glibc
        musl_ver = self._detect_musl()
        if musl_ver is not None:
            data["musl_version"] = musl_ver
            return data

        # Try glibc
        glibc_ver = self._detect_glibc()
        if glibc_ver is not None:
            data["glibc_version"] = glibc_ver

        return data

    # ------------------------------------------------------------------
    # GLIBC detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_glibc() -> Optional[Tuple[int, int]]:
        """Detect the GLIBC version.

        Returns a ``(major, minor)`` tuple or ``None`` if GLIBC cannot be
        found.
        """
        # Method 1: os.confstr
        ver = GlibcProbe._glibc_via_confstr()
        if ver is not None:
            return ver

        # Method 2: ctypes
        ver = GlibcProbe._glibc_via_ctypes()
        if ver is not None:
            return ver

        return None

    @staticmethod
    def _glibc_via_confstr() -> Optional[Tuple[int, int]]:
        """Parse ``os.confstr('CS_GNU_LIBC_VERSION')``."""
        try:
            libc_string = os.confstr("CS_GNU_LIBC_VERSION")  # e.g. "glibc 2.35"
            if not libc_string:
                return None
            return GlibcProbe._parse_glibc_version_string(libc_string)
        except (ValueError, OSError, AttributeError):
            return None

    @staticmethod
    def _glibc_via_ctypes() -> Optional[Tuple[int, int]]:
        """Load ``libc.so.6`` via ctypes and call ``gnu_get_libc_version``."""
        try:
            libc = ctypes.CDLL("libc.so.6")
            gnu_get_libc_version = libc.gnu_get_libc_version
            gnu_get_libc_version.restype = ctypes.c_char_p
            version_bytes: bytes = gnu_get_libc_version()
            version_str = version_bytes.decode("ascii", errors="replace")
            return GlibcProbe._parse_dotted_version(version_str)
        except (OSError, AttributeError, TypeError):
            return None

    @staticmethod
    def _parse_glibc_version_string(s: str) -> Optional[Tuple[int, int]]:
        """Parse a string like ``'glibc 2.35'`` into ``(2, 35)``."""
        match = re.search(r"(\d+)\.(\d+)", s)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    @staticmethod
    def _parse_dotted_version(s: str) -> Optional[Tuple[int, int]]:
        """Parse ``'2.35'`` into ``(2, 35)``."""
        match = re.match(r"(\d+)\.(\d+)", s.strip())
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

    # ------------------------------------------------------------------
    # musl detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_musl() -> Optional[Tuple[int, int]]:
        """Detect musl libc and its version.

        Returns a ``(major, minor)`` tuple or ``None``.
        """
        # Method 1: /proc/self/maps
        ver = GlibcProbe._musl_via_proc_maps()
        if ver is not None:
            return ver

        # Method 2: ldd --version (musl ldd writes to stderr)
        ver = GlibcProbe._musl_via_ldd()
        if ver is not None:
            return ver

        return None

    @staticmethod
    def _musl_via_proc_maps() -> Optional[Tuple[int, int]]:
        """Scan ``/proc/self/maps`` for musl shared objects.

        A musl-based system maps something like::

            7f...  /lib/ld-musl-x86_64.so.1

        We then try to extract the version by reading the library's
        banner (the first few bytes often contain the version string)
        or by invoking the linker with ``--version``.
        """
        try:
            with open("/proc/self/maps", "r") as fh:
                for line in fh:
                    if "musl" in line:
                        # Extract the library path
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            lib_path = parts[-1]
                            ver = GlibcProbe._musl_version_from_binary(lib_path)
                            if ver is not None:
                                return ver
                        # Even if we cannot determine the version, signal
                        # that musl is present with a fallback.
                        return (1, 0)
        except (FileNotFoundError, PermissionError, OSError):
            pass
        return None

    @staticmethod
    def _musl_version_from_binary(path: str) -> Optional[Tuple[int, int]]:
        """Execute the musl linker binary to extract version information.

        ``ld-musl-*.so.1 --version`` prints something like::

            musl libc (x86_64)
            Version 1.2.3
        """
        try:
            result = subprocess.run(
                [path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # musl's ld.so exits non-zero when invoked directly but
            # prints version info to stderr.
            output = result.stdout + result.stderr
            return GlibcProbe._parse_musl_version_output(output)
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _musl_via_ldd() -> Optional[Tuple[int, int]]:
        """Run ``ldd --version`` and parse musl's banner."""
        try:
            result = subprocess.run(
                ["ldd", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            if "musl" not in output.lower():
                return None
            return GlibcProbe._parse_musl_version_output(output)
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _parse_musl_version_output(output: str) -> Optional[Tuple[int, int]]:
        """Extract musl version from its banner text.

        Looks for patterns like ``Version 1.2.3`` or ``musl libc ... 1.2.3``.
        """
        # "Version 1.2.3"
        match = re.search(r"[Vv]ersion\s+(\d+)\.(\d+)", output)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        # Fallback: any x.y after "musl"
        match = re.search(r"musl.*?(\d+)\.(\d+)", output, re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
