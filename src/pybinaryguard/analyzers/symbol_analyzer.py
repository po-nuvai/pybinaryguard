"""Symbol analyzer for GLIBC version requirements and CPython ABI detection.

Operates on the ``SharedObjectInfo`` instances already populated by the
``ELFAnalyzer``.  This analyzer extracts the maximum GLIBC version
required across all ``.so`` files and detects CPython-specific symbols.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from pybinaryguard.analyzers.base import AnalyzerBase
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo

logger = logging.getLogger(__name__)

# Regex for GLIBC version strings like "GLIBC_2.17"
_GLIBC_VERSION_RE = re.compile(r"GLIBC_(\d+)\.(\d+)")

# Regex for GLIBCXX version strings like "GLIBCXX_3.4.29"
_GLIBCXX_VERSION_RE = re.compile(r"GLIBCXX_([\d.]+)")

# CPython ABI symbols begin with these prefixes
_CPYTHON_SYMBOL_PREFIXES = (
    "_Py",
    "Py",
    "PyInit_",
    "_PyArg_",
    "PyModule_",
    "PyErr_",
    "PyObject_",
    "PyType_",
    "PyLong_",
    "PyFloat_",
    "PyUnicode_",
    "PyList_",
    "PyDict_",
    "PyTuple_",
    "PyBytes_",
    "PyMem_",
    "PyGILState_",
)


class SymbolAnalyzer(AnalyzerBase):
    """Analyzes GNU version requirements and CPython symbols.

    This analyzer runs *after* the ``ELFAnalyzer`` and refines the
    package-level ``required_glibc`` and ``required_glibcxx`` fields by
    inspecting the ``gnu_version_requirements`` already extracted from
    each ``SharedObjectInfo``.

    It also sets ``has_python_symbols`` on individual shared objects
    based on whether their ``DT_NEEDED`` list references ``libpython``
    or their version requirements reference CPython-specific version
    tags.
    """

    name: str = "symbol"

    def analyze(self, package_info: PackageBinaryInfo) -> PackageBinaryInfo:
        """Compute aggregate GLIBC/GLIBCXX requirements and detect CPython ABI.

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

        max_glibc: Optional[Tuple[int, int]] = None
        max_glibcxx: Optional[str] = None

        for so in package_info.shared_objects:
            # Compute per-object GLIBC
            so_glibc = self._max_glibc_from_requirements(so.gnu_version_requirements)
            if so_glibc is not None:
                so.required_glibc = so_glibc
                if max_glibc is None or so_glibc > max_glibc:
                    max_glibc = so_glibc

            # Compute per-object GLIBCXX
            so_glibcxx = self._max_glibcxx_from_requirements(so.gnu_version_requirements)
            if so_glibcxx is not None:
                so.required_glibcxx = so_glibcxx
                if max_glibcxx is None or _compare_glibcxx(so_glibcxx, max_glibcxx) > 0:
                    max_glibcxx = so_glibcxx

            # Detect CPython ABI usage
            so.has_python_symbols = self._detect_python_symbols(so)

        if max_glibc is not None:
            package_info.required_glibc = max_glibc

        if max_glibcxx is not None:
            package_info.required_glibcxx = max_glibcxx

        return package_info

    # ------------------------------------------------------------------
    # GLIBC version extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _max_glibc_from_requirements(requirements: List[str]) -> Optional[Tuple[int, int]]:
        """Extract the maximum GLIBC version from version requirement strings.

        Parameters
        ----------
        requirements:
            List of strings like ``"libc.so.6(GLIBC_2.17)"``.

        Returns
        -------
        (major, minor) or None
        """
        max_ver: Optional[Tuple[int, int]] = None
        for req in requirements:
            match = _GLIBC_VERSION_RE.search(req)
            if match:
                ver = (int(match.group(1)), int(match.group(2)))
                if max_ver is None or ver > max_ver:
                    max_ver = ver
        return max_ver

    @staticmethod
    def _max_glibcxx_from_requirements(requirements: List[str]) -> Optional[str]:
        """Extract the maximum GLIBCXX version string from version requirements.

        Parameters
        ----------
        requirements:
            List of strings like ``"libstdc++.so.6(GLIBCXX_3.4.29)"``.

        Returns
        -------
        Version string like ``"GLIBCXX_3.4.29"`` or None.
        """
        max_ver: Optional[str] = None
        for req in requirements:
            match = _GLIBCXX_VERSION_RE.search(req)
            if match:
                ver_str = f"GLIBCXX_{match.group(1)}"
                if max_ver is None or _compare_glibcxx(ver_str, max_ver) > 0:
                    max_ver = ver_str
        return max_ver

    # ------------------------------------------------------------------
    # CPython symbol detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_python_symbols(so: SharedObjectInfo) -> bool:
        """Detect whether *so* links against CPython.

        Checks:
        1. DT_NEEDED contains ``libpython*.so*``
        2. DT_SONAME matches ``cpython-*`` pattern (CPython extension naming)
        3. The filename matches the CPython extension naming convention
           (e.g. ``foo.cpython-312-x86_64-linux-gnu.so``)
        """
        # Check DT_NEEDED for libpython
        for lib in so.dt_needed:
            if lib.startswith("libpython"):
                return True

        # Check soname
        if so.dt_soname and "cpython" in so.dt_soname.lower():
            return True

        # Check filename convention: *.cpython-XYZ-*.so
        if ".cpython-" in so.filename:
            return True

        return False


def parse_glibc_version(version_string: str) -> Optional[Tuple[int, int]]:
    """Parse a GLIBC version string into a ``(major, minor)`` tuple.

    Parameters
    ----------
    version_string:
        A string like ``"GLIBC_2.17"`` or just ``"2.17"``.

    Returns
    -------
    (int, int) or None
        The parsed version, or ``None`` if parsing fails.

    Examples
    --------
    >>> parse_glibc_version("GLIBC_2.17")
    (2, 17)
    >>> parse_glibc_version("2.34")
    (2, 34)
    >>> parse_glibc_version("invalid")
    """
    # Try with prefix first
    match = _GLIBC_VERSION_RE.search(version_string)
    if match:
        return (int(match.group(1)), int(match.group(2)))

    # Try bare version
    parts = version_string.strip().split(".")
    if len(parts) >= 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass

    return None


def compute_max_glibc(shared_objects: List[SharedObjectInfo]) -> Optional[Tuple[int, int]]:
    """Compute the maximum required GLIBC version across shared objects.

    Parameters
    ----------
    shared_objects:
        The shared object descriptors, each with a ``required_glibc``
        field that may or may not be set.

    Returns
    -------
    (int, int) or None
        The highest GLIBC version required, or ``None`` if none of the
        objects require GLIBC.
    """
    max_ver: Optional[Tuple[int, int]] = None
    for so in shared_objects:
        if so.required_glibc is not None:
            if max_ver is None or so.required_glibc > max_ver:
                max_ver = so.required_glibc
    return max_ver


def _compare_glibcxx(a: str, b: str) -> int:
    """Compare two GLIBCXX version strings numerically.

    Parameters
    ----------
    a, b:
        Version strings like ``"GLIBCXX_3.4.29"``.

    Returns
    -------
    int
        Negative if *a* < *b*, zero if equal, positive if *a* > *b*.
    """
    def _parts(s: str) -> List[int]:
        match = _GLIBCXX_VERSION_RE.search(s)
        if not match:
            return [0]
        try:
            return [int(x) for x in match.group(1).split(".")]
        except ValueError:
            return [0]

    pa = _parts(a)
    pb = _parts(b)

    # Pad to equal length
    max_len = max(len(pa), len(pb))
    pa.extend([0] * (max_len - len(pa)))
    pb.extend([0] * (max_len - len(pb)))

    for va, vb in zip(pa, pb):
        if va != vb:
            return 1 if va > vb else -1
    return 0
