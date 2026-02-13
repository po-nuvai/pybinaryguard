"""Probe for CPU architecture and feature flags."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, FrozenSet

from pybinaryguard.models.enums import Architecture

from .base import ProbeBase


class CpuProbe(ProbeBase):
    """Collects CPU architecture, model, core count, and feature flags.

    On x86 the flags are read from the ``flags`` line in
    ``/proc/cpuinfo``.  On ARM/AArch64 the equivalent is the
    ``Features`` line.  This probe works without external dependencies.
    """

    name = "cpu"

    def collect(self) -> Dict[str, Any]:
        """Return architecture, CPU model, core count, and feature flags."""
        data: Dict[str, Any] = {}

        machine = self._get_machine()
        data["architecture"] = Architecture.from_machine(machine)

        cpuinfo = self._read_cpuinfo()

        data["cpu_model"] = self._extract_cpu_model(cpuinfo, machine)
        data["cpu_cores"] = self._get_core_count()

        flags = self._extract_flags(cpuinfo, machine)
        data["cpu_flags"] = flags

        # Feature detection based on flags
        data["has_sse42"] = self._has_flag(flags, {"sse4_2", "sse4.2"})
        data["has_avx"] = self._has_flag(flags, {"avx"})
        data["has_avx2"] = self._has_flag(flags, {"avx2"})
        data["has_avx512"] = self._has_any_avx512(flags)
        data["has_neon"] = self._detect_neon(flags, machine)

        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_machine() -> str:
        """Return the raw machine string from the platform module."""
        try:
            return platform.machine()
        except Exception:
            return ""

    @staticmethod
    def _read_cpuinfo() -> str:
        """Read ``/proc/cpuinfo`` and return its content as a string."""
        try:
            with open("/proc/cpuinfo", "r") as fh:
                return fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    @staticmethod
    def _extract_cpu_model(cpuinfo: str, machine: str) -> str:
        """Extract the CPU model name from cpuinfo text.

        On x86: ``model name  : Intel(R) Core(TM) ...``
        On ARM: ``Hardware    : BCM2835`` or ``model name`` if present.
        """
        # Try "model name" first (works on x86 and some ARM)
        for line in cpuinfo.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("model name"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()

        # ARM fallback: "Hardware" line
        if machine.startswith(("aarch64", "arm")):
            for line in cpuinfo.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("hardware"):
                    parts = stripped.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip()

        # Last resort: "CPU implementer" + "CPU part" (common on AArch64)
        if machine.startswith(("aarch64", "arm")):
            implementer = ""
            part = ""
            for line in cpuinfo.splitlines():
                stripped = line.strip().lower()
                if stripped.startswith("cpu implementer"):
                    p = stripped.split(":", 1)
                    if len(p) == 2:
                        implementer = p[1].strip()
                elif stripped.startswith("cpu part"):
                    p = stripped.split(":", 1)
                    if len(p) == 2:
                        part = p[1].strip()
            if implementer or part:
                return f"implementer={implementer} part={part}"

        return ""

    @staticmethod
    def _get_core_count() -> int:
        """Return the number of logical CPU cores."""
        try:
            count = os.cpu_count()
            return count if count is not None else 0
        except Exception:
            return 0

    @staticmethod
    def _extract_flags(cpuinfo: str, machine: str) -> FrozenSet[str]:
        """Extract CPU feature flags from ``/proc/cpuinfo``.

        On x86, flags appear on a ``flags`` line.
        On ARM/AArch64, they appear on a ``Features`` line.
        """
        # Determine which key to look for
        if machine in ("x86_64", "AMD64", "i686", "i386"):
            key_candidates = ["flags"]
        elif machine.startswith(("aarch64", "arm")):
            key_candidates = ["features", "flags"]
        else:
            key_candidates = ["flags", "features"]

        for line in cpuinfo.splitlines():
            stripped = line.strip().lower()
            for key in key_candidates:
                if stripped.startswith(key):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        flag_list = parts[1].strip().split()
                        return frozenset(f.lower() for f in flag_list)

        return frozenset()

    @staticmethod
    def _has_flag(flags: FrozenSet[str], names: set) -> bool:  # type: ignore[type-arg]
        """Return ``True`` if *any* of the given names are present in *flags*."""
        return bool(flags & names)

    @staticmethod
    def _has_any_avx512(flags: FrozenSet[str]) -> bool:
        """Return ``True`` if any AVX-512 feature flag is present.

        AVX-512 is reported as multiple sub-features (``avx512f``,
        ``avx512bw``, ``avx512vl``, etc.).  The foundation flag
        ``avx512f`` must be present for any AVX-512 to be usable, but
        we also accept any flag starting with ``avx512``.
        """
        return any(f.startswith("avx512") for f in flags)

    @staticmethod
    def _detect_neon(flags: FrozenSet[str], machine: str) -> bool:
        """Detect ARM NEON support.

        On AArch64 NEON is mandatory, so we return ``True`` even when
        the ``neon`` flag is not explicitly listed in ``/proc/cpuinfo``.
        On 32-bit ARM we look for the ``neon`` flag.
        """
        if machine in ("aarch64", "arm64"):
            # NEON is mandatory in the ARMv8-A architecture
            return True
        if "neon" in flags:
            return True
        return False
