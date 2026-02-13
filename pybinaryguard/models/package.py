"""Package binary information data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Set

from .enums import Architecture


@dataclass
class SharedObjectInfo:
    """Information extracted from a single .so file."""

    path: str
    filename: str
    architecture: Architecture = Architecture.UNKNOWN
    elf_class: int = 0  # 32 or 64
    endianness: str = "little"
    dt_needed: List[str] = field(default_factory=list)
    dt_soname: Optional[str] = None
    dt_rpath: Optional[str] = None
    dt_runpath: Optional[str] = None
    required_glibc: Optional[Tuple[int, int]] = None
    required_glibcxx: Optional[str] = None
    gnu_version_requirements: List[str] = field(default_factory=list)
    has_python_symbols: bool = False
    build_id: Optional[str] = None
    file_size: int = 0
    sha256: Optional[str] = None  # Populated in DEEP scan mode


@dataclass
class WheelTag:
    """A parsed wheel compatibility tag."""

    interpreter: str  # e.g., "cp312"
    abi: str          # e.g., "cp312"
    platform: str     # e.g., "manylinux_2_17_x86_64"


@dataclass
class PackageBinaryInfo:
    """Complete binary analysis of an installed Python package."""

    package_name: str
    package_version: str
    install_path: str

    # Wheel metadata
    wheel_tags: List[WheelTag] = field(default_factory=list)
    is_pure_python: bool = True

    # Binary analysis
    shared_objects: List[SharedObjectInfo] = field(default_factory=list)
    required_glibc: Optional[Tuple[int, int]] = None
    required_glibcxx: Optional[str] = None
    target_architecture: Optional[Architecture] = None
    required_libraries: Set[str] = field(default_factory=set)
    missing_libraries: Set[str] = field(default_factory=set)

    # Framework-specific
    cuda_build_version: Optional[Tuple[int, int]] = None
    numpy_api_version: Optional[int] = None

    @property
    def name(self) -> str:
        """Alias for package_name (used by snapshot module)."""
        return self.package_name

    @property
    def version(self) -> Optional[str]:
        """Alias for package_version (used by snapshot module)."""
        return self.package_version

    @property
    def cuda_version(self) -> Optional[str]:
        """CUDA version string extracted from cuda_build_version tuple."""
        if self.cuda_build_version:
            return f"{self.cuda_build_version[0]}.{self.cuda_build_version[1]}"
        return None

    @property
    def has_binaries(self) -> bool:
        """Whether this package contains any compiled extensions."""
        return len(self.shared_objects) > 0

    @property
    def so_count(self) -> int:
        """Number of shared object files in this package."""
        return len(self.shared_objects)

    @property
    def manylinux_tag(self) -> Optional[str]:
        """Extract the manylinux tag from wheel tags, if any."""
        for tag in self.wheel_tags:
            if "manylinux" in tag.platform:
                return tag.platform
        return None

    @property
    def manylinux_glibc(self) -> Optional[Tuple[int, int]]:
        """Extract the GLIBC version claimed by the manylinux tag."""
        tag = self.manylinux_tag
        if not tag:
            return None
        # Parse manylinux_2_17_x86_64 or manylinux2014_x86_64
        import re
        match = re.search(r"manylinux_(\d+)_(\d+)", tag)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        # Legacy tags
        legacy_map = {
            "manylinux1": (2, 5),
            "manylinux2010": (2, 12),
            "manylinux2014": (2, 17),
        }
        for prefix, version in legacy_map.items():
            if tag.startswith(prefix):
                return version
        return None
