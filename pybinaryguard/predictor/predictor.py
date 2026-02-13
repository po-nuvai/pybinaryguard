"""Main predictor module for runtime failure prediction.

Combines symbol resolution, dependency analysis, and linker simulation
to predict ImportErrors before they happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.predictor.linker_simulator import LinkerSimulator


@dataclass
class PredictedFailure:
    """A predicted runtime failure."""

    package: str
    module_path: str
    error_type: str  # "ImportError", "SymbolError", "LibraryMissing"
    error_message: str
    missing_symbol: Optional[str] = None
    missing_library: Optional[str] = None
    confidence: float = 0.0


def predict_import_failures(
    package: PackageBinaryInfo,
    profile: SystemProfile
) -> List[PredictedFailure]:
    """Predict import failures for a package.

    Args:
        package: Package binary information
        profile: System profile

    Returns:
        List of predicted failures
    """
    failures: List[PredictedFailure] = []

    # Get library paths from system
    library_paths = profile.library_search_paths or None

    simulator = LinkerSimulator(library_paths)

    # Simulate loading each shared object
    for so in package.shared_objects:
        if not so.path:
            continue

        # Predict import error
        error_msg = simulator.predict_import_error(so.path, f"{package.name}._C")

        if error_msg:
            # Parse error type
            if "undefined symbol" in error_msg:
                error_type = "SymbolError"
                # Extract symbol name
                if "undefined symbol:" in error_msg:
                    parts = error_msg.split("undefined symbol:")
                    if len(parts) > 1:
                        missing_symbol = parts[1].strip()
                    else:
                        missing_symbol = "unknown"
                else:
                    missing_symbol = "unknown"

                failures.append(
                    PredictedFailure(
                        package=package.name,
                        module_path=so.path,
                        error_type=error_type,
                        error_message=error_msg,
                        missing_symbol=missing_symbol,
                        confidence=0.95,
                    )
                )

            elif "cannot open shared object file" in error_msg:
                error_type = "LibraryMissing"
                # Extract library name
                if "file:" in error_msg:
                    parts = error_msg.split("file:")
                    if len(parts) > 1:
                        lib_part = parts[1].split(":")[0].strip()
                        missing_library = lib_part
                    else:
                        missing_library = "unknown"
                else:
                    missing_library = "unknown"

                failures.append(
                    PredictedFailure(
                        package=package.name,
                        module_path=so.path,
                        error_type=error_type,
                        error_message=error_msg,
                        missing_library=missing_library,
                        confidence=0.9,
                    )
                )

            else:
                # Generic import error
                failures.append(
                    PredictedFailure(
                        package=package.name,
                        module_path=so.path,
                        error_type="ImportError",
                        error_message=error_msg,
                        confidence=0.8,
                    )
                )

    return failures


def format_predicted_failure(failure: PredictedFailure) -> str:
    """Format a predicted failure as a human-readable message.

    Args:
        failure: PredictedFailure to format

    Returns:
        Formatted message string
    """
    lines = [
        f"Predicted {failure.error_type} in {failure.package}:",
        f"  {failure.error_message}",
    ]

    if failure.missing_symbol:
        lines.append(f"  Missing symbol: {failure.missing_symbol}")

    if failure.missing_library:
        lines.append(f"  Missing library: {failure.missing_library}")

    lines.append(f"  Confidence: {failure.confidence * 100:.0f}%")

    return "\n".join(lines)
