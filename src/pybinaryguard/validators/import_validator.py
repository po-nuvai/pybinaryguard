"""Import validator — actually tries importing packages to catch real failures.

This is the nuclear option: instead of predicting failures from metadata,
we import each package in an isolated subprocess and capture the exact error.
This catches everything — GLIBC mismatches, missing .so files, CUDA errors,
illegal instructions, segfaults — because we're running the actual code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ImportTestResult:
    """Result of attempting to import a single package."""

    package_name: str
    top_level_name: str
    success: bool
    error_type: Optional[str] = None    # "ImportError", "OSError", "Signal", etc.
    error_message: Optional[str] = None
    category: Optional[str] = None       # "glibc_mismatch", "missing_lib", etc.
    exit_code: int = 0
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "package": self.package_name,
            "import_name": self.top_level_name,
            "success": self.success,
        }
        if not self.success:
            result["error_type"] = self.error_type
            result["error_message"] = self.error_message
            if self.category:
                result["category"] = self.category
            result["exit_code"] = self.exit_code
        result["duration_ms"] = round(self.duration_ms, 1)
        return result


class ImportValidator:
    """Test imports by actually running them in isolated subprocesses.

    Each import test runs in a fresh Python subprocess with a timeout,
    capturing stdout/stderr for error analysis. Signal-based crashes
    (SIGILL, SIGSEGV) are detected from the exit code.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        python_executable: Optional[str] = None,
    ) -> None:
        self._timeout = timeout
        self._python = python_executable or sys.executable

    def test_import(self, package_name: str, top_level: Optional[str] = None) -> ImportTestResult:
        """Test importing a single package in an isolated subprocess.

        Args:
            package_name: The pip package name.
            top_level: The actual importable module name (if different from package_name).

        Returns:
            ImportTestResult with success/failure details.
        """
        import_name = top_level or package_name.replace("-", "_").replace(".", "_")

        # Build the test script
        script = (
            f"import sys; "
            f"try:\n"
            f"    import {import_name}\n"
            f"    print('OK')\n"
            f"except ImportError as e:\n"
            f"    print(f'ImportError: {{e}}', file=sys.stderr)\n"
            f"    sys.exit(10)\n"
            f"except OSError as e:\n"
            f"    print(f'OSError: {{e}}', file=sys.stderr)\n"
            f"    sys.exit(11)\n"
            f"except Exception as e:\n"
            f"    print(f'{{type(e).__name__}}: {{e}}', file=sys.stderr)\n"
            f"    sys.exit(12)\n"
        )

        start = time.monotonic()

        try:
            result = subprocess.run(
                [self._python, "-c", script],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            duration_ms = (time.monotonic() - start) * 1000

            if result.returncode == 0:
                return ImportTestResult(
                    package_name=package_name,
                    top_level_name=import_name,
                    success=True,
                    duration_ms=duration_ms,
                )

            # Parse error
            stderr = result.stderr.strip()
            error_type, error_msg, category = self._classify_error(
                result.returncode, stderr
            )

            return ImportTestResult(
                package_name=package_name,
                top_level_name=import_name,
                success=False,
                error_type=error_type,
                error_message=error_msg,
                category=category,
                exit_code=result.returncode,
                duration_ms=duration_ms,
            )

        except subprocess.TimeoutExpired:
            duration_ms = (time.monotonic() - start) * 1000
            return ImportTestResult(
                package_name=package_name,
                top_level_name=import_name,
                success=False,
                error_type="Timeout",
                error_message=f"Import timed out after {self._timeout}s",
                category="timeout",
                exit_code=-1,
                duration_ms=duration_ms,
            )

    def test_packages(
        self,
        packages: List[Tuple[str, Optional[str]]],
    ) -> List[ImportTestResult]:
        """Test importing multiple packages sequentially.

        Args:
            packages: List of (package_name, top_level_name) tuples.

        Returns:
            List of ImportTestResult for each package.
        """
        results = []
        for pkg_name, top_level in packages:
            results.append(self.test_import(pkg_name, top_level))
        return results

    @staticmethod
    def _classify_error(
        exit_code: int, stderr: str
    ) -> Tuple[str, str, Optional[str]]:
        """Classify the error from exit code and stderr output."""
        stderr_lower = stderr.lower()

        # Signal-based crashes (negative exit codes or 128+signal on Linux)
        signal_codes = {
            -4: "illegal_instruction", 132: "illegal_instruction",   # SIGILL
            -11: "segfault", 139: "segfault",                       # SIGSEGV
            -6: "abort", 134: "abort",                              # SIGABRT
            -9: "killed", 137: "killed",                            # SIGKILL
        }
        if exit_code in signal_codes:
            category = signal_codes[exit_code]
            messages = {
                "illegal_instruction": (
                    "Illegal instruction (SIGILL) — binary compiled for "
                    "incompatible CPU architecture or instruction set"
                ),
                "segfault": "Segmentation fault (SIGSEGV) — binary memory access violation",
                "abort": "Aborted (SIGABRT) — binary assertion or abort() call",
                "killed": "Killed (SIGKILL) — process was killed (out of memory?)",
            }
            return ("Signal", messages.get(category, f"Signal {abs(exit_code)}"), category)
        if exit_code < 0 or exit_code > 128:
            if exit_code < 0:
                sig = abs(exit_code)
            else:
                sig = exit_code - 128
            if sig > 0 and sig < 32:
                return ("Signal", f"Process killed by signal {sig}", "signal")

        # ImportError (exit code 10)
        if exit_code == 10:
            if "glibc" in stderr_lower or "version" in stderr_lower:
                return ("ImportError", stderr, "glibc_mismatch")
            elif "libcuda" in stderr_lower or "cuda" in stderr_lower:
                return ("ImportError", stderr, "cuda_missing")
            elif "cannot open shared object" in stderr_lower:
                return ("ImportError", stderr, "missing_shared_library")
            elif "undefined symbol" in stderr_lower:
                return ("ImportError", stderr, "undefined_symbol")
            return ("ImportError", stderr, "import_error")

        # OSError (exit code 11)
        if exit_code == 11:
            if "cannot open shared object" in stderr_lower:
                return ("OSError", stderr, "missing_shared_library")
            return ("OSError", stderr, "os_error")

        # Other exceptions (exit code 12)
        if exit_code == 12:
            return ("Exception", stderr, "runtime_error")

        return ("Unknown", stderr or f"Exit code {exit_code}", None)

    @staticmethod
    def get_top_level_name(dist_info_path: str, package_name: str) -> str:
        """Get the importable module name from top_level.txt or package name."""
        top_level_file = os.path.join(dist_info_path, "top_level.txt")
        if os.path.isfile(top_level_file):
            try:
                with open(top_level_file, "r") as f:
                    names = [line.strip() for line in f if line.strip()]
                    if names:
                        return names[0]
            except OSError:
                pass
        # Fallback: normalize package name
        return package_name.replace("-", "_").replace(".", "_")
