"""Dynamic linker simulation for predicting runtime failures.

Simulates the behavior of ld.so to predict symbol resolution failures
before they occur at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pybinaryguard.predictor.dependency_graph import DependencyGraph
from pybinaryguard.predictor.resolver import SymbolResolver


@dataclass
class LinkageResult:
    """Result of simulated dynamic linking."""

    success: bool
    missing_symbols: List[str]
    missing_libraries: List[str]
    symbol_sources: Dict[str, Optional[str]]  # symbol -> providing library
    error_message: Optional[str] = None


class LinkerSimulator:
    """Simulates dynamic linker behavior to predict failures."""

    def __init__(self, library_paths: Optional[List[str]] = None):
        """Initialize the linker simulator.

        Args:
            library_paths: Directories to search for libraries
        """
        self.resolver = SymbolResolver(library_paths)
        self.library_paths = library_paths

    def simulate_load(self, binary_path: str) -> LinkageResult:
        """Simulate loading a binary and resolving all symbols.

        Args:
            binary_path: Path to the ELF binary

        Returns:
            LinkageResult with success status and details
        """
        # Build dependency graph
        dep_graph = DependencyGraph(self.library_paths)
        root_node = dep_graph.build_from_binary(binary_path)

        # Check for missing libraries
        missing_libs = dep_graph.get_unresolved_dependencies()

        if missing_libs:
            return LinkageResult(
                success=False,
                missing_symbols=[],
                missing_libraries=missing_libs,
                symbol_sources={},
                error_message=f"Missing libraries: {', '.join(missing_libs)}"
            )

        # Try to resolve symbols
        needed_libs = root_node.dependencies
        symbol_resolution = self.resolver.resolve_undefined_symbols(
            binary_path,
            needed_libs
        )

        # Find unresolved symbols
        missing_symbols = [
            sym for sym, provider in symbol_resolution.items()
            if provider is None
        ]

        if missing_symbols:
            # Format error message similar to actual ImportError
            first_missing = missing_symbols[0]
            error_msg = (
                f"ImportError: {binary_path}: "
                f"undefined symbol: {first_missing}"
            )

            return LinkageResult(
                success=False,
                missing_symbols=missing_symbols,
                missing_libraries=[],
                symbol_sources=symbol_resolution,
                error_message=error_msg
            )

        # Check for circular dependencies
        if dep_graph.has_circular_dependencies():
            return LinkageResult(
                success=True,  # Still successful, but warn
                missing_symbols=[],
                missing_libraries=[],
                symbol_sources=symbol_resolution,
                error_message="Warning: Circular dependencies detected"
            )

        return LinkageResult(
            success=True,
            missing_symbols=[],
            missing_libraries=[],
            symbol_sources=symbol_resolution,
        )

    def predict_import_error(
        self,
        module_path: str,
        extension_name: str
    ) -> Optional[str]:
        """Predict ImportError for a Python extension module.

        Args:
            module_path: Path to the extension module (.so file)
            extension_name: Name of the extension (for error message)

        Returns:
            Predicted error message, or None if import should succeed
        """
        result = self.simulate_load(module_path)

        if not result.success:
            if result.missing_libraries:
                # Format like actual ImportError
                missing = result.missing_libraries[0]
                return (
                    f"ImportError: {extension_name}: "
                    f"cannot open shared object file: {missing}: "
                    f"No such file or directory"
                )

            if result.missing_symbols:
                # Format like actual ImportError
                missing = result.missing_symbols[0]
                return (
                    f"ImportError: {module_path}: "
                    f"undefined symbol: {missing}"
                )

        return None


def simulate_import(
    binary_path: str,
    library_paths: Optional[List[str]] = None
) -> LinkageResult:
    """Convenience function to simulate importing a binary.

    Args:
        binary_path: Path to the ELF binary
        library_paths: Optional library search paths

    Returns:
        LinkageResult with simulation results
    """
    simulator = LinkerSimulator(library_paths)
    return simulator.simulate_load(binary_path)
