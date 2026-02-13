"""Core enumerations for PyBinaryGuard."""

from __future__ import annotations

from enum import Enum


class Severity(Enum):
    """Severity level for diagnostic findings."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    PASSED = "passed"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2, Severity.PASSED: 3}
        return order[self] < order[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self == other or self < other


class Architecture(Enum):
    """CPU architecture identifiers."""

    X86_64 = "x86_64"
    AARCH64 = "aarch64"
    ARMV7L = "armv7l"
    I686 = "i686"
    PPC64LE = "ppc64le"
    S390X = "s390x"
    UNKNOWN = "unknown"

    @classmethod
    def from_machine(cls, machine: str) -> Architecture:
        """Convert platform.machine() output to Architecture enum."""
        mapping = {
            "x86_64": cls.X86_64,
            "AMD64": cls.X86_64,
            "aarch64": cls.AARCH64,
            "arm64": cls.AARCH64,
            "armv7l": cls.ARMV7L,
            "armv6l": cls.ARMV7L,
            "i686": cls.I686,
            "i386": cls.I686,
            "ppc64le": cls.PPC64LE,
            "s390x": cls.S390X,
        }
        return mapping.get(machine, cls.UNKNOWN)

    @property
    def elf_machine(self) -> int:
        """Return the ELF e_machine constant for this architecture."""
        mapping = {
            Architecture.X86_64: 62,     # EM_X86_64
            Architecture.AARCH64: 183,   # EM_AARCH64
            Architecture.ARMV7L: 40,     # EM_ARM
            Architecture.I686: 3,        # EM_386
            Architecture.PPC64LE: 21,    # EM_PPC64
            Architecture.S390X: 22,      # EM_S390
        }
        return mapping.get(self, 0)


class ContainerRuntime(Enum):
    """Container runtime types."""

    DOCKER = "docker"
    PODMAN = "podman"
    LXC = "lxc"
    CONTAINERD = "containerd"
    NONE = "none"


class ScanMode(Enum):
    """Scan depth mode.

    FAST:
        Metadata-only scan. Reads WHEEL/METADATA files and checks tags
        against the system profile. Skips ELF binary parsing entirely.
        Target: < 1 second for 100 packages.

    STANDARD:
        Default mode. Parses ELF headers for architecture, GLIBC version
        requirements, and DT_NEEDED libraries. Runs all built-in rules.
        Target: < 3 seconds for 100 packages (x86).

    DEEP:
        Full analysis. Everything in STANDARD plus: SHA256 hash verification
        of all shared objects, recursive DT_NEEDED dependency chain resolution,
        symbol-level version checking, and predictive import failure simulation.
        Target: < 10 seconds for 100 packages.
    """

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"
