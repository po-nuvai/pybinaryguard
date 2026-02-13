"""Binary analyzers for PyBinaryGuard.

This module provides the analysis pipeline that inspects installed Python
packages for binary compatibility issues.  Each analyzer is responsible
for a specific aspect of the analysis:

- **ELFAnalyzer** -- discovers and parses ELF shared objects.
- **WheelAnalyzer** -- reads wheel / dist-info metadata.
- **SymbolAnalyzer** -- extracts GLIBC version requirements and CPython ABI.
- **DependencyAnalyzer** -- resolves DT_NEEDED chains statically.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

from pybinaryguard.analyzers.base import AnalyzerBase
from pybinaryguard.analyzers.dependency_analyzer import DependencyAnalyzer
from pybinaryguard.analyzers.elf_analyzer import ELFAnalyzer, ELFParseError, MinimalELFParser
from pybinaryguard.analyzers.symbol_analyzer import (
    SymbolAnalyzer,
    compute_max_glibc,
    parse_glibc_version,
)
from pybinaryguard.analyzers.wheel_analyzer import (
    WheelAnalyzer,
    find_dist_info_dirs,
)
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyzerBase",
    "DependencyAnalyzer",
    "ELFAnalyzer",
    "ELFParseError",
    "MinimalELFParser",
    "SymbolAnalyzer",
    "WheelAnalyzer",
    "analyze_packages",
    "compute_max_glibc",
    "find_dist_info_dirs",
    "get_all_analyzers",
    "parse_glibc_version",
]


def get_all_analyzers(
    system_profile: Optional[SystemProfile] = None,
) -> List[AnalyzerBase]:
    """Return an ordered list of all built-in analyzers.

    The order matters: the ELF analyzer must run first (it discovers
    ``.so`` files), followed by the wheel analyzer (metadata), then the
    symbol analyzer (refines GLIBC data), and finally the dependency
    analyzer (resolves library paths).

    Parameters
    ----------
    system_profile:
        Optional system profile.  Passed to analyzers that need system
        information (e.g. ``DependencyAnalyzer``).

    Returns
    -------
    list of AnalyzerBase
    """
    return [
        ELFAnalyzer(),
        WheelAnalyzer(),
        SymbolAnalyzer(),
        DependencyAnalyzer(system_profile=system_profile),
    ]


def analyze_packages(
    site_packages_paths: List[str],
    system_profile: Optional[SystemProfile] = None,
    analyzers: Optional[Sequence[AnalyzerBase]] = None,
) -> List[PackageBinaryInfo]:
    """Discover and analyze all installed packages with binary components.

    Walks the given ``site-packages`` directories, creates a
    ``PackageBinaryInfo`` for each discovered distribution, runs the
    full analyzer pipeline, and returns only packages that contain
    compiled extensions.

    Parameters
    ----------
    site_packages_paths:
        Paths to ``site-packages`` directories to scan.
    system_profile:
        Optional system profile for dependency resolution.
    analyzers:
        Optional list of analyzers to run.  If ``None``, the default
        set from ``get_all_analyzers()`` is used.

    Returns
    -------
    list of PackageBinaryInfo
        Only packages that have at least one ``.so`` file.
    """
    if analyzers is None:
        analyzers = get_all_analyzers(system_profile=system_profile)

    results: List[PackageBinaryInfo] = []

    for site_path in site_packages_paths:
        if not os.path.isdir(site_path):
            logger.debug("Skipping non-existent site-packages: %s", site_path)
            continue

        dist_infos = find_dist_info_dirs(site_path)
        logger.debug("Found %d distributions in %s", len(dist_infos), site_path)

        for dist_info_path, pkg_name, pkg_version in dist_infos:
            # Determine the package's install directory.  This is
            # typically a directory under site-packages with the same
            # name as the package (lowercased), but we also check
            # top_level.txt for accuracy.
            install_path = _guess_install_path(site_path, dist_info_path, pkg_name)
            if install_path is None:
                continue

            pkg_info = PackageBinaryInfo(
                package_name=pkg_name,
                package_version=pkg_version,
                install_path=install_path,
            )

            for analyzer in analyzers:
                try:
                    analyzer.analyze(pkg_info)
                except Exception:
                    logger.debug(
                        "Analyzer %s failed on %s",
                        getattr(analyzer, "name", type(analyzer).__name__),
                        pkg_name,
                        exc_info=True,
                    )

            if pkg_info.has_binaries:
                results.append(pkg_info)

    return results


def _guess_install_path(
    site_path: str, dist_info_path: str, pkg_name: str
) -> Optional[str]:
    """Determine the package's install directory under *site_path*.

    Checks (in order):
    1. ``top_level.txt`` in the dist-info directory
    2. A directory matching the package name (lowercased, with hyphens
       replaced by underscores)
    3. The site-packages directory itself (for namespace packages or
       single-file installs)

    Returns
    -------
    str or None
        The install path, or ``None`` if the package directory cannot
        be determined.
    """
    # 1. Check top_level.txt
    top_level_path = os.path.join(dist_info_path, "top_level.txt")
    try:
        if os.path.isfile(top_level_path):
            with open(top_level_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    top_pkg = line.strip()
                    if not top_pkg:
                        continue
                    candidate = os.path.join(site_path, top_pkg)
                    if os.path.isdir(candidate):
                        return candidate
    except OSError:
        pass

    # 2. Guess from package name
    normalized = pkg_name.lower().replace("-", "_").replace(".", "_")
    candidate = os.path.join(site_path, normalized)
    if os.path.isdir(candidate):
        return candidate

    # 3. Try the RECORD file to find actual installed files
    record_path = os.path.join(dist_info_path, "RECORD")
    try:
        if os.path.isfile(record_path):
            with open(record_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    file_path = line.split(",", 1)[0].strip()
                    if file_path.endswith((".so", ".pyd")):
                        # The first path component is the package directory
                        parts = file_path.split("/", 1)
                        if len(parts) >= 1:
                            pkg_dir = os.path.join(site_path, parts[0])
                            if os.path.isdir(pkg_dir):
                                return pkg_dir
    except OSError:
        pass

    # 4. Fall back to site-packages itself (namespace packages, etc.)
    return site_path
