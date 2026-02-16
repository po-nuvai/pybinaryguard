"""System probes for PyBinaryGuard.

This package contains all probes that collect information about the
current system's environment.  Each probe is a subclass of
:class:`ProbeBase` and implements a ``collect()`` method that returns
a dictionary whose keys correspond to :class:`SystemProfile` fields.

Usage::

    from pybinaryguard.probes import get_all_probes

    for probe in get_all_probes():
        if probe.is_applicable():
            data = probe.collect()
"""

from __future__ import annotations

from typing import List

from .base import ProbeBase
from .board_probe import BoardProbe
from .cpu_probe import CpuProbe
from .glibc_probe import GlibcProbe
from .gpu_probe import GpuProbe
from .library_probe import LibraryProbe
from .os_probe import OsProbe
from .python_probe import PythonProbe
from .toolchain_probe import ToolchainProbe
from .venv_probe import VenvProbe

__all__ = [
    "ProbeBase",
    "BoardProbe",
    "CpuProbe",
    "GlibcProbe",
    "GpuProbe",
    "LibraryProbe",
    "OsProbe",
    "PythonProbe",
    "ToolchainProbe",
    "VenvProbe",
    "get_all_probes",
]


def get_all_probes() -> List[ProbeBase]:
    """Return an instance of every available probe.

    The probes are returned in a deterministic order chosen so that
    cheaper / more fundamental probes run first:

    1. **PythonProbe** -- interpreter info (fast, no I/O beyond stdlib)
    2. **VenvProbe** -- virtual environment detection (fast)
    3. **CpuProbe** -- CPU architecture and flags (reads /proc/cpuinfo)
    4. **OsProbe** -- OS and container detection
    5. **GlibcProbe** -- C library version
    6. **ToolchainProbe** -- build toolchain (gcc, cmake, etc.)
    7. **LibraryProbe** -- shared library paths and ldconfig cache
    8. **BoardProbe** -- embedded board detection
    9. **GpuProbe** -- GPU / CUDA detection (most expensive)
    """
    return [
        PythonProbe(),
        VenvProbe(),
        CpuProbe(),
        OsProbe(),
        GlibcProbe(),
        ToolchainProbe(),
        LibraryProbe(),
        BoardProbe(),
        GpuProbe(),
    ]
