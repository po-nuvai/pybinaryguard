"""Probe for shared library paths and linker cache."""

from __future__ import annotations

import os
import re
import site
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from .base import ProbeBase


class LibraryProbe(ProbeBase):
    """Collects shared library path and linker cache information.

    Gathered fields:
    - ``ld_library_path``: directories from ``$LD_LIBRARY_PATH``
    - ``ldconfig_cache``: mapping of library names to resolved paths
      as reported by ``ldconfig -p``
    - ``site_packages_paths``: Python site-packages directories
    """

    name = "library"

    def collect(self) -> Dict[str, Any]:
        """Return library path, ldconfig cache, and site-packages paths."""
        data: Dict[str, Any] = {}

        data["ld_library_path"] = self._get_ld_library_path()
        data["ldconfig_cache"] = self._get_ldconfig_cache()
        data["site_packages_paths"] = self._get_site_packages()

        return data

    # ------------------------------------------------------------------
    # LD_LIBRARY_PATH
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ld_library_path() -> Tuple[str, ...]:
        """Parse ``$LD_LIBRARY_PATH`` into a tuple of directory strings.

        Empty and duplicate entries are preserved because the dynamic
        linker processes them in order and their semantics may matter.
        """
        raw = os.environ.get("LD_LIBRARY_PATH", "")
        if not raw:
            return ()
        return tuple(raw.split(os.pathsep))

    # ------------------------------------------------------------------
    # ldconfig cache
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ldconfig_cache() -> Dict[str, str]:
        """Run ``ldconfig -p`` and parse the output into a dict.

        Returns a mapping of library soname to its absolute file path,
        e.g. ``{"libz.so.1": "/lib/x86_64-linux-gnu/libz.so.1"}``.

        Only the *first* occurrence of each soname is kept (ldconfig
        prints in search order).
        """
        try:
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {}
            return LibraryProbe._parse_ldconfig_output(result.stdout)
        except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
            return {}

    @staticmethod
    def _parse_ldconfig_output(output: str) -> Dict[str, str]:
        """Parse the human-readable output of ``ldconfig -p``.

        Each line after the header looks like::

            \tlibz.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libz.so.1
        """
        cache: Dict[str, str] = {}
        # Pattern: leading whitespace, soname, parenthesised flags, =>, path
        pattern = re.compile(r"^\s+(\S+)\s+\(.*?\)\s+=>\s+(\S+)")
        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                soname = match.group(1)
                path = match.group(2)
                # Keep only the first occurrence
                if soname not in cache:
                    cache[soname] = path
        return cache

    # ------------------------------------------------------------------
    # site-packages
    # ------------------------------------------------------------------

    @staticmethod
    def _get_site_packages() -> Tuple[str, ...]:
        """Return all site-packages directories known to the interpreter.

        Combines ``site.getsitepackages()`` (system-level) with
        ``site.getusersitepackages()`` (user-level).
        """
        paths: List[str] = []

        try:
            system_paths = site.getsitepackages()
            paths.extend(system_paths)
        except (AttributeError, Exception):
            # site.getsitepackages may not exist in virtualenvs
            pass

        try:
            user_path = site.getusersitepackages()
            if isinstance(user_path, str):
                paths.append(user_path)
        except (AttributeError, Exception):
            pass

        # Fallback: derive from sys.path if the above yielded nothing
        if not paths:
            paths = [p for p in sys.path if "site-packages" in p or "dist-packages" in p]

        return tuple(paths)
