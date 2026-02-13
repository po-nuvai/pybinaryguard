"""Wheel / dist-info metadata analyzer.

Reads packaging metadata from ``.dist-info`` directories to extract wheel
tags, package identity, file lists, and framework-specific build markers.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from pybinaryguard.analyzers.base import AnalyzerBase
from pybinaryguard.models.package import PackageBinaryInfo, WheelTag

logger = logging.getLogger(__name__)

# Pre-compiled patterns
_TAG_RE = re.compile(r"^Tag:\s*(.+)$", re.MULTILINE)
_NAME_RE = re.compile(r"^Name:\s*(.+)$", re.MULTILINE)
_VERSION_RE = re.compile(r"^Version:\s*(.+)$", re.MULTILINE)
_CUDA_VERSION_RE = re.compile(r"\+cu(\d{2,3})")


class WheelAnalyzer(AnalyzerBase):
    """Analyzes wheel packaging metadata from ``.dist-info`` directories.

    Extracts:
    - Wheel compatibility tags (``Tag:`` lines in ``WHEEL``)
    - Package name and version (from ``METADATA``)
    - File manifest (from ``RECORD``)
    - Pure-Python detection
    - CUDA build version from the version string
    """

    name: str = "wheel"

    def analyze(self, package_info: PackageBinaryInfo) -> PackageBinaryInfo:
        """Read dist-info metadata and enrich *package_info*.

        Parameters
        ----------
        package_info:
            The package descriptor to populate.

        Returns
        -------
        PackageBinaryInfo
            The same (mutated) instance.
        """
        dist_info = self._find_dist_info(package_info.install_path, package_info.package_name)
        if dist_info is None:
            return package_info

        self._parse_wheel_file(dist_info, package_info)
        self._parse_metadata_file(dist_info, package_info)
        self._detect_pure_python(dist_info, package_info)
        self._extract_cuda_version(package_info)

        return package_info

    # ------------------------------------------------------------------
    # dist-info discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_dist_info(install_path: str, package_name: str) -> Optional[str]:
        """Locate the ``.dist-info`` directory for *package_name*.

        Searches the parent directory of *install_path* (typically
        ``site-packages``) for a matching ``.dist-info`` folder.

        Parameters
        ----------
        install_path:
            The package's own install directory (e.g.
            ``site-packages/numpy``).
        package_name:
            The distribution name (e.g. ``numpy``).

        Returns
        -------
        str or None
            Absolute path to the ``.dist-info`` directory, or ``None``.
        """
        site_dir = os.path.dirname(install_path)
        if not os.path.isdir(site_dir):
            return None

        # Normalize: PEP 503 says compare names case-insensitively after
        # replacing [-_.] with dashes.
        normalized = _normalize_name(package_name)

        try:
            entries = os.listdir(site_dir)
        except OSError as exc:
            logger.debug("Cannot list %s: %s", site_dir, exc)
            return None

        for entry in sorted(entries):
            if not entry.endswith(".dist-info"):
                continue
            # Entry format: <name>-<version>.dist-info
            entry_name_part = entry.rsplit("-", 1)[0] if "-" in entry else entry
            if _normalize_name(entry_name_part) == normalized:
                full = os.path.join(site_dir, entry)
                if os.path.isdir(full):
                    return full

        # Broader search: sometimes the package directory name and the
        # distribution name diverge (e.g. ``Pillow`` installs as ``PIL``).
        # Try matching any dist-info whose top_level.txt contains the
        # package directory basename.
        pkg_basename = os.path.basename(install_path).lower()
        for entry in sorted(entries):
            if not entry.endswith(".dist-info"):
                continue
            full = os.path.join(site_dir, entry)
            top_level_path = os.path.join(full, "top_level.txt")
            try:
                if os.path.isfile(top_level_path):
                    with open(top_level_path, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            if line.strip().lower() == pkg_basename:
                                return full
            except OSError:
                continue

        return None

    # ------------------------------------------------------------------
    # WHEEL file
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_wheel_file(dist_info: str, package_info: PackageBinaryInfo) -> None:
        """Parse ``WHEEL`` for ``Tag:`` entries."""
        wheel_path = os.path.join(dist_info, "WHEEL")
        try:
            with open(wheel_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as exc:
            logger.debug("Cannot read %s: %s", wheel_path, exc)
            return

        for match in _TAG_RE.finditer(content):
            raw_tag = match.group(1).strip()
            parts = raw_tag.split("-")
            if len(parts) == 3:
                package_info.wheel_tags.append(
                    WheelTag(interpreter=parts[0], abi=parts[1], platform=parts[2])
                )
            elif len(parts) > 3:
                # Platform tag itself may contain hyphens (rare, but possible
                # with compound tags like manylinux_2_17_x86_64.manylinux2014_x86_64)
                package_info.wheel_tags.append(
                    WheelTag(
                        interpreter=parts[0],
                        abi=parts[1],
                        platform="-".join(parts[2:]),
                    )
                )

    # ------------------------------------------------------------------
    # METADATA file
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_metadata_file(dist_info: str, package_info: PackageBinaryInfo) -> None:
        """Parse ``METADATA`` for ``Name:`` and ``Version:``."""
        meta_path = os.path.join(dist_info, "METADATA")
        try:
            with open(meta_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as exc:
            logger.debug("Cannot read %s: %s", meta_path, exc)
            return

        name_match = _NAME_RE.search(content)
        if name_match:
            package_info.package_name = name_match.group(1).strip()

        ver_match = _VERSION_RE.search(content)
        if ver_match:
            package_info.package_version = ver_match.group(1).strip()

    # ------------------------------------------------------------------
    # Pure-Python detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_pure_python(dist_info: str, package_info: PackageBinaryInfo) -> None:
        """Determine if the package is pure Python.

        A package is pure Python if:
        - All wheel tags have platform ``any``, OR
        - The RECORD file lists no ``.so`` / ``.pyd`` / ``.dll`` files
        """
        # Check tags first (faster)
        if package_info.wheel_tags:
            all_any = all(tag.platform == "any" for tag in package_info.wheel_tags)
            if all_any:
                package_info.is_pure_python = True
                return
            # Has platform-specific tags -> likely not pure
            package_info.is_pure_python = False
            return

        # Fallback: scan RECORD
        record_path = os.path.join(dist_info, "RECORD")
        try:
            with open(record_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    path_part = line.split(",", 1)[0].strip()
                    if path_part.endswith((".so", ".pyd", ".dll")) or ".so." in path_part:
                        package_info.is_pure_python = False
                        return
        except OSError:
            pass

        # No evidence of binaries
        package_info.is_pure_python = True

    # ------------------------------------------------------------------
    # CUDA version extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cuda_version(package_info: PackageBinaryInfo) -> None:
        """Extract CUDA build version from the package version string.

        Looks for the ``+cuXYZ`` local version suffix.
        Examples:
        - ``2.4.0+cu124`` -> ``(12, 4)``
        - ``2.1.0+cu118`` -> ``(11, 8)``
        - ``1.0.0+cu121`` -> ``(12, 1)``
        """
        match = _CUDA_VERSION_RE.search(package_info.package_version)
        if match:
            digits = match.group(1)
            if len(digits) == 3:
                major = int(digits[:2])
                minor = int(digits[2:])
            elif len(digits) == 2:
                major = int(digits[0])
                minor = int(digits[1])
            else:
                return
            package_info.cuda_build_version = (major, minor)


def _normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503.

    Lowercases and replaces all runs of ``[-_.]`` with a single dash.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def find_dist_info_dirs(site_packages: str) -> List[Tuple[str, str, str]]:
    """Enumerate all ``.dist-info`` directories in a site-packages folder.

    Returns
    -------
    list of (dist_info_path, package_name, package_version)
        One tuple per discovered distribution.
    """
    results: List[Tuple[str, str, str]] = []
    try:
        entries = os.listdir(site_packages)
    except OSError as exc:
        logger.debug("Cannot list %s: %s", site_packages, exc)
        return results

    for entry in sorted(entries):
        if not entry.endswith(".dist-info"):
            continue
        full = os.path.join(site_packages, entry)
        if not os.path.isdir(full):
            continue

        # Parse name-version from the directory name
        stem = entry[: -len(".dist-info")]
        parts = stem.rsplit("-", 1)
        if len(parts) == 2:
            name, version = parts
        else:
            name = stem
            version = "unknown"

        # Override with METADATA if available
        meta_path = os.path.join(full, "METADATA")
        try:
            with open(meta_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(4096)  # Only need the header section
            name_match = _NAME_RE.search(content)
            if name_match:
                name = name_match.group(1).strip()
            ver_match = _VERSION_RE.search(content)
            if ver_match:
                version = ver_match.group(1).strip()
        except OSError:
            pass

        results.append((full, name, version))

    return results
