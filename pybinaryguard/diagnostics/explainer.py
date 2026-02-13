"""Maps error messages and rule IDs to plain-English explanations.

This module provides the ``ErrorPatternDB`` class which matches raw error
messages against a curated database of regex patterns and returns
beginner-friendly diagnoses including root cause, explanation, and fix hints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pybinaryguard.models.finding import Finding


# ---------------------------------------------------------------------------
# Error-pattern database
# ---------------------------------------------------------------------------

ERROR_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": r"version .GLIBC_(\d+\.\d+). not found",
        "root_cause": "GLIBC version too old",
        "rule_id": "GLIBC_VERSION_MISMATCH",
        "explanation": (
            "Your system's C library (GLIBC) is older than what this package "
            "needs. The package was compiled on a newer Linux system."
        ),
        "fix_hint": "Upgrade your OS or use an older package version.",
    },
    {
        "pattern": r"Illegal instruction",
        "root_cause": "CPU instruction set mismatch",
        "rule_id": "ILLEGAL_INSTRUCTION_RISK",
        "explanation": (
            "The package uses CPU instructions (like AVX2) that your "
            "processor doesn't support."
        ),
        "fix_hint": "Install a version built for your CPU, or use a different machine.",
    },
    {
        "pattern": r"libcudart\.so\.(\d+): cannot open",
        "root_cause": "CUDA runtime version mismatch",
        "rule_id": "CUDA_RUNTIME_MISMATCH",
        "explanation": (
            "The package was built for a different CUDA version than "
            "what's installed on your system."
        ),
        "fix_hint": "Install the package variant matching your CUDA version.",
    },
    {
        "pattern": r"libcublas\.so\.(\d+): cannot open",
        "root_cause": "CUDA library missing",
        "rule_id": "CUDA_LIB_MISSING",
        "explanation": (
            "A CUDA math library required by this package is not installed."
        ),
        "fix_hint": (
            "Install the matching CUDA toolkit or reinstall the package "
            "for your CUDA version."
        ),
    },
    {
        "pattern": r"undefined symbol: _Py",
        "root_cause": "Python ABI mismatch",
        "rule_id": "PYTHON_ABI_MISMATCH",
        "explanation": (
            "The package was compiled for a different Python version's "
            "internal API."
        ),
        "fix_hint": "Reinstall the package for your current Python version.",
    },
    {
        "pattern": (
            r"module compiled against API version (0x[0-9a-f]+) "
            r"but this version of numpy is (0x[0-9a-f]+)"
        ),
        "root_cause": "NumPy C API version mismatch",
        "rule_id": "NUMPY_ABI_MISMATCH",
        "explanation": (
            "The package was built against a different NumPy version's C API."
        ),
        "fix_hint": "Reinstall the package or upgrade/downgrade NumPy.",
    },
    {
        "pattern": r"not a supported wheel on this platform",
        "root_cause": "Platform/architecture mismatch",
        "rule_id": "ARCH_MISMATCH",
        "explanation": (
            "This wheel was built for a different CPU architecture or OS."
        ),
        "fix_hint": "Download the correct wheel for your platform.",
    },
    {
        "pattern": r"cudaErrorInsufficientDriver",
        "root_cause": "GPU driver too old",
        "rule_id": "CUDA_DRIVER_TOO_OLD",
        "explanation": (
            "Your GPU driver is too old to support the installed CUDA version."
        ),
        "fix_hint": "Update your NVIDIA GPU driver.",
    },
    {
        "pattern": r"NVML: Driver/library version mismatch",
        "root_cause": "Container driver mismatch",
        "rule_id": "CONTAINER_DRIVER_MISMATCH",
        "explanation": (
            "The NVIDIA driver in your container doesn't match the host "
            "driver. This often happens when the container image has a "
            "newer CUDA than the host supports."
        ),
        "fix_hint": (
            "Use nvidia-container-toolkit and ensure host driver supports "
            "your CUDA version."
        ),
    },
    {
        "pattern": r"libcudnn\.so\.\d+: cannot open",
        "root_cause": "cuDNN missing",
        "rule_id": "CUDNN_VERSION_MISMATCH",
        "explanation": (
            "The cuDNN library required for GPU-accelerated deep learning "
            "is not installed."
        ),
        "fix_hint": "Install the cuDNN version matching your CUDA installation.",
    },
]


# Pre-compile patterns once at import time for performance.
_COMPILED_PATTERNS: List[Tuple[re.Pattern[str], Dict[str, str]]] = [
    (re.compile(entry["pattern"]), entry) for entry in ERROR_PATTERNS
]


# ---------------------------------------------------------------------------
# ErrorPatternDB
# ---------------------------------------------------------------------------


class ErrorPatternDB:
    """A database of error-message patterns with associated root causes.

    The default set of patterns is embedded in the module (``ERROR_PATTERNS``)
    *and* also available as ``error_patterns.json`` for external tooling.  You
    can extend the database at runtime by loading additional patterns from
    JSON files or adding them programmatically.
    """

    def __init__(self) -> None:
        self._patterns: List[Dict[str, str]] = list(ERROR_PATTERNS)
        self._compiled: List[Tuple[re.Pattern[str], Dict[str, str]]] = list(
            _COMPILED_PATTERNS
        )

    # -- mutation -----------------------------------------------------------

    def add_pattern(self, entry: Dict[str, str]) -> None:
        """Add a single pattern entry to the database.

        Parameters
        ----------
        entry:
            A dict with keys ``pattern``, ``root_cause``, ``rule_id``,
            ``explanation``, and ``fix_hint``.

        Raises
        ------
        ValueError
            If any required key is missing.
        """
        required_keys = {"pattern", "root_cause", "rule_id", "explanation", "fix_hint"}
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(f"Pattern entry is missing required keys: {missing}")

        self._patterns.append(entry)
        self._compiled.append((re.compile(entry["pattern"]), entry))

    def load_json(self, path: str) -> int:
        """Load additional patterns from a JSON file.

        Parameters
        ----------
        path:
            Absolute or relative path to a JSON file containing a list of
            pattern entries.

        Returns
        -------
        int
            Number of patterns loaded.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        ValueError
            If any entry is missing required keys.
        """
        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as fh:
            entries: List[Dict[str, str]] = json.load(fh)

        for entry in entries:
            self.add_pattern(entry)
        return len(entries)

    # -- queries ------------------------------------------------------------

    def get_error_patterns(self) -> List[Dict[str, str]]:
        """Return all registered patterns as a list of dicts.

        Returns
        -------
        List[Dict[str, str]]
            A shallow copy of the internal pattern list.
        """
        return list(self._patterns)

    def diagnose_error(self, error_message: str) -> Optional[Dict[str, str]]:
        """Match *error_message* against all known patterns.

        Parameters
        ----------
        error_message:
            The raw error text (e.g. a traceback line or stderr fragment).

        Returns
        -------
        Optional[Dict[str, str]]
            A dict with keys ``root_cause``, ``rule_id``, ``explanation``,
            ``fix_hint``, and ``matched_pattern`` if a match is found; otherwise
            ``None``.
        """
        if not error_message:
            return None

        for compiled, entry in self._compiled:
            match = compiled.search(error_message)
            if match:
                return {
                    "root_cause": entry["root_cause"],
                    "rule_id": entry["rule_id"],
                    "explanation": entry["explanation"],
                    "fix_hint": entry["fix_hint"],
                    "matched_pattern": entry["pattern"],
                }
        return None

    def explain_finding(self, finding: Finding) -> str:
        """Generate a full plain-English explanation paragraph for a Finding.

        The explanation combines the finding's own explanation text with
        any matching pattern data (if the finding carries a
        ``related_error``), and appends a fix suggestion when available.

        Parameters
        ----------
        finding:
            A :class:`Finding` instance to explain.

        Returns
        -------
        str
            A multi-sentence explanation suitable for displaying to
            end users.
        """
        parts: List[str] = []

        # Start with the finding's built-in explanation.
        if finding.explanation:
            parts.append(finding.explanation)

        # If there is a related error string, try to diagnose it for
        # additional context.
        if finding.related_error:
            diagnosis = self.diagnose_error(finding.related_error)
            if diagnosis and diagnosis["explanation"] not in (finding.explanation or ""):
                parts.append(
                    f"Root cause: {diagnosis['root_cause']}. "
                    f"{diagnosis['explanation']}"
                )
                if diagnosis["fix_hint"]:
                    parts.append(f"Suggested fix: {diagnosis['fix_hint']}")

        # Fall back to the finding's own suggestion if nothing else matched.
        if finding.suggestion and not any("Suggested fix" in p for p in parts):
            parts.append(f"Suggested fix: {finding.suggestion}")

        if not parts:
            return f"No detailed explanation available for rule {finding.rule_id}."

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience functions (thin wrappers around a default DB)
# ---------------------------------------------------------------------------

_DEFAULT_DB = ErrorPatternDB()


def diagnose_error(error_message: str) -> Optional[Dict[str, str]]:
    """Match *error_message* against all known error patterns.

    This is a convenience wrapper around
    :meth:`ErrorPatternDB.diagnose_error` using the default built-in
    pattern database.

    Parameters
    ----------
    error_message:
        The raw error text to diagnose.

    Returns
    -------
    Optional[Dict[str, str]]
        Diagnosis dict or ``None``.
    """
    return _DEFAULT_DB.diagnose_error(error_message)


def explain_finding(finding: Finding) -> str:
    """Generate a plain-English explanation for a Finding.

    This is a convenience wrapper around
    :meth:`ErrorPatternDB.explain_finding` using the default built-in
    pattern database.

    Parameters
    ----------
    finding:
        The finding to explain.

    Returns
    -------
    str
        A human-readable explanation paragraph.
    """
    return _DEFAULT_DB.explain_finding(finding)


def get_error_patterns() -> List[Dict[str, str]]:
    """Return all built-in error patterns.

    Returns
    -------
    List[Dict[str, str]]
        A list of pattern dicts.
    """
    return _DEFAULT_DB.get_error_patterns()
