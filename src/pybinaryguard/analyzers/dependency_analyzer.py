"""Dependency analyzer -- resolves DT_NEEDED chains statically.

For each shared library required by a package's ``.so`` files, this
analyzer checks whether the library can be found at runtime without
executing any binaries.  Libraries that cannot be resolved are added to
``PackageBinaryInfo.missing_libraries``.
"""

from __future__ import annotations

import logging
import os
import platform
from typing import Dict, List, Optional, Set, Tuple

from pybinaryguard.analyzers.base import AnalyzerBase
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile

logger = logging.getLogger(__name__)

# Standard library search paths per architecture
_STANDARD_LIB_PATHS: Tuple[str, ...] = (
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
)

# Architecture-specific multilib directories
_ARCH_LIB_DIRS: Dict[str, Tuple[str, ...]] = {
    "x86_64": (
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
    ),
    "aarch64": (
        "/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ),
    "armv7l": (
        "/lib/arm-linux-gnueabihf",
        "/usr/lib/arm-linux-gnueabihf",
    ),
    "i686": (
        "/lib/i386-linux-gnu",
        "/usr/lib/i386-linux-gnu",
        "/lib32",
        "/usr/lib32",
    ),
    "ppc64le": (
        "/lib/powerpc64le-linux-gnu",
        "/usr/lib/powerpc64le-linux-gnu",
    ),
    "s390x": (
        "/lib/s390x-linux-gnu",
        "/usr/lib/s390x-linux-gnu",
    ),
}

# Libraries that are always expected to be provided by the C runtime and
# should not be flagged as missing.
_ALWAYS_PRESENT: frozenset = frozenset({
    "linux-vdso.so.1",
    "linux-gate.so.1",
    "linux-vdso64.so.1",
    "ld-linux-x86-64.so.2",
    "ld-linux.so.2",
    "ld-linux-aarch64.so.1",
    "ld-linux-armhf.so.3",
    "ld-linux-riscv64-lp64d.so.1",
    "ld64.so.1",
    "ld64.so.2",
})


class DependencyAnalyzer(AnalyzerBase):
    """Resolves DT_NEEDED library dependencies through static path searching.

    For each library referenced by a package's shared objects, the analyzer
    searches (in order):

    1. ``DT_RPATH`` / ``DT_RUNPATH`` from the binary itself
    2. ``LD_LIBRARY_PATH`` from the system profile
    3. The ldconfig cache (``SystemProfile.ldconfig_cache``)
    4. Standard library directories (``/lib``, ``/usr/lib``, arch-specific)

    Libraries that cannot be found are added to
    ``PackageBinaryInfo.missing_libraries``.

    This analyzer never executes any binary (no ``ldd``, no ``ldconfig``).
    """

    name: str = "dependency"

    def __init__(self, system_profile: Optional[SystemProfile] = None) -> None:
        self._profile = system_profile

    def analyze(self, package_info: PackageBinaryInfo) -> PackageBinaryInfo:
        """Resolve library dependencies and flag missing ones.

        Parameters
        ----------
        package_info:
            The package descriptor whose ``shared_objects`` list has
            already been populated by the ELF analyzer.

        Returns
        -------
        PackageBinaryInfo
            The same (mutated) instance.
        """
        if not package_info.shared_objects:
            return package_info

        # Build the set of all DT_NEEDED libraries across the package
        all_needed: Set[str] = set()
        for so in package_info.shared_objects:
            for lib in so.dt_needed:
                all_needed.add(lib)

        if not all_needed:
            return package_info

        # Collect search paths from system profile
        ld_library_paths = self._get_ld_library_path()
        ldconfig_cache = self._get_ldconfig_cache()
        arch_dirs = self._get_arch_lib_dirs()

        # Collect per-binary RPATH/RUNPATH
        rpath_dirs: List[str] = []
        runpath_dirs: List[str] = []
        for so in package_info.shared_objects:
            if so.dt_rpath:
                for p in so.dt_rpath.split(":"):
                    resolved = self._resolve_rpath_token(p, so.path)
                    if resolved and resolved not in rpath_dirs:
                        rpath_dirs.append(resolved)
            if so.dt_runpath:
                for p in so.dt_runpath.split(":"):
                    resolved = self._resolve_rpath_token(p, so.path)
                    if resolved and resolved not in runpath_dirs:
                        runpath_dirs.append(resolved)

        # Also include the package's own install path as a search location
        # (many packages bundle their own .so files alongside each other)
        package_lib_dirs: List[str] = []
        if os.path.isdir(package_info.install_path):
            package_lib_dirs.append(package_info.install_path)
            # Also check common subdirectories
            for subdir in ("lib", "libs", ".libs"):
                candidate = os.path.join(package_info.install_path, subdir)
                if os.path.isdir(candidate):
                    package_lib_dirs.append(candidate)

        # Build the names of libraries that the package itself provides
        # (so intra-package deps don't get flagged as missing)
        provided_by_package: Set[str] = set()
        for so in package_info.shared_objects:
            provided_by_package.add(so.filename)
            if so.dt_soname:
                provided_by_package.add(so.dt_soname)

        # Resolve each library
        missing: Set[str] = set()
        for lib_name in sorted(all_needed):
            if lib_name in _ALWAYS_PRESENT:
                continue
            if lib_name in provided_by_package:
                continue
            if not self._find_library(
                lib_name,
                rpath_dirs=rpath_dirs,
                runpath_dirs=runpath_dirs,
                ld_library_paths=ld_library_paths,
                ldconfig_cache=ldconfig_cache,
                package_dirs=package_lib_dirs,
                arch_dirs=arch_dirs,
            ):
                missing.add(lib_name)

        package_info.missing_libraries = missing
        return package_info

    # ------------------------------------------------------------------
    # Library resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _find_library(
        name: str,
        *,
        rpath_dirs: List[str],
        runpath_dirs: List[str],
        ld_library_paths: List[str],
        ldconfig_cache: Dict[str, str],
        package_dirs: List[str],
        arch_dirs: List[str],
    ) -> bool:
        """Search for a library by name using the ELF resolution order.

        The search order mirrors the dynamic linker (ld.so) behavior:

        1. DT_RPATH (deprecated but still used)
        2. LD_LIBRARY_PATH
        3. DT_RUNPATH
        4. ldconfig cache
        5. Package-internal directories
        6. Standard and architecture-specific paths

        Parameters
        ----------
        name:
            The library file name (e.g. ``libz.so.1``).

        Returns
        -------
        bool
            ``True`` if the library was found.
        """
        # 1. RPATH
        for d in rpath_dirs:
            if _file_exists(os.path.join(d, name)):
                return True

        # 2. LD_LIBRARY_PATH
        for d in ld_library_paths:
            if _file_exists(os.path.join(d, name)):
                return True

        # 3. RUNPATH
        for d in runpath_dirs:
            if _file_exists(os.path.join(d, name)):
                return True

        # 4. ldconfig cache
        if name in ldconfig_cache:
            path = ldconfig_cache[name]
            if _file_exists(path):
                return True

        # 5. Package-internal directories
        for d in package_dirs:
            if _file_exists(os.path.join(d, name)):
                return True

        # 6. Standard + arch-specific paths
        for d in list(_STANDARD_LIB_PATHS) + arch_dirs:
            if _file_exists(os.path.join(d, name)):
                return True

        return False

    # ------------------------------------------------------------------
    # RPATH token resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_rpath_token(path: str, binary_path: str) -> Optional[str]:
        """Resolve ``$ORIGIN`` and ``${ORIGIN}`` tokens in RPATH/RUNPATH.

        Parameters
        ----------
        path:
            A single path component from RPATH/RUNPATH.
        binary_path:
            The absolute path to the binary that contains the RPATH.

        Returns
        -------
        str or None
            The resolved absolute path, or ``None`` if invalid.
        """
        if not path:
            return None

        origin = os.path.dirname(os.path.abspath(binary_path))
        resolved = path.replace("$ORIGIN", origin).replace("${ORIGIN}", origin)
        resolved = os.path.normpath(resolved)

        if os.path.isdir(resolved):
            return resolved

        # The directory might not exist yet on the system being analyzed,
        # but we still return it so the caller can check
        return resolved

    # ------------------------------------------------------------------
    # Profile helpers
    # ------------------------------------------------------------------

    def _get_ld_library_path(self) -> List[str]:
        """Return LD_LIBRARY_PATH entries from the system profile or env."""
        if self._profile is not None and self._profile.ld_library_path:
            return list(self._profile.ld_library_path)

        # Fallback: read from the current environment
        env_val = os.environ.get("LD_LIBRARY_PATH", "")
        if env_val:
            return [p for p in env_val.split(":") if p]
        return []

    def _get_ldconfig_cache(self) -> Dict[str, str]:
        """Return the ldconfig cache mapping from the system profile."""
        if self._profile is not None:
            return dict(self._profile.ldconfig_cache)
        return {}

    def _get_arch_lib_dirs(self) -> List[str]:
        """Return architecture-specific library directories."""
        arch = ""
        if self._profile is not None:
            arch = self._profile.architecture.value
        else:
            arch = platform.machine()

        dirs = list(_ARCH_LIB_DIRS.get(arch, ()))

        # Also check the current machine if different
        if not dirs:
            machine = platform.machine()
            dirs = list(_ARCH_LIB_DIRS.get(machine, ()))

        return dirs


def _file_exists(path: str) -> bool:
    """Check if *path* exists and is a file (or symlink to a file).

    Silently returns ``False`` on permission errors.
    """
    try:
        return os.path.isfile(path)
    except OSError:
        return False
