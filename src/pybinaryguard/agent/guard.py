"""Runtime import guard — intercept import failures with structured diagnostics.

Activation::

    # Method 1: Side-effect import (activates immediately)
    import pybinaryguard.agent.guard

    # Method 2: Explicit control
    from pybinaryguard.agent.guard import enable_guard, disable_guard
    enable_guard()
    # ... do imports ...
    disable_guard()

    # Method 3: Context manager
    from pybinaryguard.agent.guard import guarded_imports
    with guarded_imports() as issues:
        import torch
    if issues:
        print(issues)  # List of structured diagnostics

When active, the guard installs a custom ``sys.excepthook`` that intercepts
``ImportError`` and ``OSError`` (missing .so) during imports, runs a fast
diagnostic, and stores structured results in ``guard.captured_issues``.

The guard does NOT suppress exceptions — it captures diagnostics alongside
the normal traceback.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type


# Thread-local storage for guard state
_local = threading.local()

# Module-level state
_original_excepthook: Optional[Callable[..., Any]] = None
_guard_active: bool = False
captured_issues: List[Dict[str, object]] = []


def _diagnose_import_error(exc: ImportError) -> Dict[str, object]:
    """Produce a structured diagnostic from an ImportError."""
    issue: Dict[str, object] = {
        "type": "import_error",
        "module": exc.name or str(exc),
        "message": str(exc),
    }

    msg = str(exc).lower()

    # Detect GLIBC issues
    if "glibc" in msg or "version" in msg and ".so" in msg:
        issue["category"] = "glibc_mismatch"
        issue["fix_hint"] = (
            "The imported library requires a newer GLIBC than your system. "
            "Try: pip install --force-reinstall <package>"
        )
    # Detect missing .so
    elif "cannot open shared object" in msg or ".so" in msg:
        issue["category"] = "missing_shared_library"
        # Extract the library name
        import re
        lib_match = re.search(r"lib\w+\.so[\.\d]*", msg)
        if lib_match:
            issue["missing_library"] = lib_match.group()
        issue["fix_hint"] = (
            "A required shared library is missing. Install the system "
            "dependency or reinstall the package."
        )
    # Detect CUDA errors
    elif "cuda" in msg or "cudart" in msg or "nvrtc" in msg:
        issue["category"] = "cuda_missing"
        issue["fix_hint"] = (
            "CUDA runtime not found. Install a CPU-only version or "
            "set up the CUDA toolkit."
        )
    # Detect architecture issues
    elif "wrong elf class" in msg or "exec format error" in msg:
        issue["category"] = "architecture_mismatch"
        issue["fix_hint"] = (
            "Binary compiled for a different CPU architecture. "
            "Reinstall with: pip install --no-binary :all: <package>"
        )
    else:
        issue["category"] = "unknown"
        issue["fix_hint"] = (
            "Run: pybinaryguard check <package> for detailed diagnostics."
        )

    return issue


def _diagnose_os_error(exc: OSError) -> Optional[Dict[str, object]]:
    """Produce a structured diagnostic from an OSError if it's import-related."""
    msg = str(exc).lower()

    # Only handle .so loading failures
    if ".so" not in msg and "shared object" not in msg:
        return None

    issue: Dict[str, object] = {
        "type": "os_error",
        "message": str(exc),
        "category": "shared_object_load_failure",
    }

    import re
    lib_match = re.search(r"(lib\w+\.so[\.\d]*)", msg)
    if lib_match:
        issue["missing_library"] = lib_match.group(1)

    issue["fix_hint"] = (
        "A shared library failed to load. Check that all system "
        "dependencies are installed."
    )
    return issue


def _guard_excepthook(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Any,
) -> None:
    """Custom excepthook that captures structured diagnostics."""
    global captured_issues

    issue = None

    if isinstance(exc_value, ImportError):
        issue = _diagnose_import_error(exc_value)
    elif isinstance(exc_value, OSError):
        issue = _diagnose_os_error(exc_value)

    if issue is not None:
        captured_issues.append(issue)

    # Always call the original excepthook — we don't suppress errors
    if _original_excepthook is not None:
        _original_excepthook(exc_type, exc_value, exc_tb)
    else:
        sys.__excepthook__(exc_type, exc_value, exc_tb)


def enable_guard() -> None:
    """Activate the import guard.

    Installs a custom ``sys.excepthook`` that captures structured
    diagnostics for import-related failures.
    """
    global _original_excepthook, _guard_active

    if _guard_active:
        return

    _original_excepthook = sys.excepthook
    sys.excepthook = _guard_excepthook
    _guard_active = True


def disable_guard() -> None:
    """Deactivate the import guard, restoring the original excepthook."""
    global _original_excepthook, _guard_active

    if not _guard_active:
        return

    if _original_excepthook is not None:
        sys.excepthook = _original_excepthook
        _original_excepthook = None

    _guard_active = False


def is_active() -> bool:
    """Return whether the import guard is currently active."""
    return _guard_active


def get_captured_issues() -> List[Dict[str, object]]:
    """Return all captured issues since the guard was enabled."""
    return list(captured_issues)


def clear_captured_issues() -> None:
    """Clear all captured issues."""
    captured_issues.clear()


@contextmanager
def guarded_imports():
    """Context manager that captures import diagnostics.

    Usage::

        with guarded_imports() as issues:
            import some_problematic_module

        if issues:
            for issue in issues:
                print(issue)

    Yields a list that will be populated with any import-related
    diagnostics captured during the block.
    """
    issues: List[Dict[str, object]] = []
    was_active = _guard_active
    prev_issues = list(captured_issues)

    enable_guard()
    clear_captured_issues()

    try:
        yield issues
    finally:
        # Capture any issues that occurred
        issues.extend(captured_issues)

        # Restore previous state
        clear_captured_issues()
        captured_issues.extend(prev_issues)

        if not was_active:
            disable_guard()


# Auto-activate on import
enable_guard()
