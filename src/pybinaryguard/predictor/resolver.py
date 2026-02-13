"""Symbol resolution for ELF binaries.

Resolves undefined symbols against available shared libraries,
simulating the behavior of the dynamic linker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from pybinaryguard.analyzers.elf_analyzer import MinimalELFParser


@dataclass
class Symbol:
    """Represents a symbol from an ELF binary."""

    name: str
    library: Optional[str] = None  # Which library provides this symbol
    version: Optional[str] = None  # Symbol version (e.g., GLIBC_2.34)
    weak: bool = False  # Whether this is a weak symbol
    undefined: bool = False  # Whether this symbol needs to be resolved


@dataclass
class LibrarySymbols:
    """Symbol table for a shared library."""

    path: str
    provided_symbols: Set[str] = field(default_factory=set)
    required_symbols: Set[str] = field(default_factory=set)
    versioned_symbols: Dict[str, str] = field(default_factory=dict)  # symbol -> version


class SymbolResolver:
    """Resolves symbols from ELF binaries against system libraries."""

    def __init__(self, library_paths: Optional[List[str]] = None):
        """Initialize the symbol resolver.

        Args:
            library_paths: List of directories to search for libraries.
                          If None, uses standard system paths.
        """
        if library_paths is None:
            library_paths = self._get_default_library_paths()

        self.library_paths = library_paths
        self._symbol_cache: Dict[str, LibrarySymbols] = {}

    def _get_default_library_paths(self) -> List[str]:
        """Get default library search paths."""
        paths = [
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/local/lib64",
        ]

        # Add LD_LIBRARY_PATH if set
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_lib_path:
            paths.extend(ld_lib_path.split(":"))

        # Filter to existing directories
        return [p for p in paths if os.path.isdir(p)]

    def resolve_undefined_symbols(
        self,
        binary_path: str,
        needed_libraries: List[str]
    ) -> Dict[str, Optional[str]]:
        """Resolve undefined symbols in a binary.

        Args:
            binary_path: Path to the ELF binary
            needed_libraries: List of DT_NEEDED libraries (from ELF)

        Returns:
            Dictionary mapping symbol name to providing library (or None if unresolved)
        """
        # Get undefined symbols from the binary
        undefined_symbols = self._extract_undefined_symbols(binary_path)

        # Build symbol index from needed libraries
        symbol_index: Dict[str, str] = {}
        for lib_name in needed_libraries:
            lib_path = self._find_library(lib_name)
            if lib_path:
                lib_symbols = self._get_library_symbols(lib_path)
                for sym in lib_symbols.provided_symbols:
                    if sym not in symbol_index:
                        symbol_index[sym] = lib_name

        # Resolve each undefined symbol
        resolution: Dict[str, Optional[str]] = {}
        for sym in undefined_symbols:
            resolution[sym] = symbol_index.get(sym)

        return resolution

    def _extract_undefined_symbols(self, binary_path: str) -> Set[str]:
        """Extract undefined symbols from an ELF binary.

        Args:
            binary_path: Path to the ELF binary

        Returns:
            Set of undefined symbol names
        """
        # This is a simplified implementation
        # A full implementation would parse the .dynsym section
        # For now, we return an empty set as a placeholder
        undefined = set()

        try:
            # Parse ELF to get dynamic symbols
            parser = MinimalELFParser(binary_path)
            # TODO: Extend MinimalELFParser to extract symbol table
            # For now, we can extract from version requirements
            for version in parser.get_version_requirements():
                # Version strings like "GLIBC_2.34" indicate required symbols
                undefined.add(version)
        except Exception:
            pass

        return undefined

    def _find_library(self, lib_name: str) -> Optional[str]:
        """Find a library by name in the search paths.

        Args:
            lib_name: Library name (e.g., "libc.so.6")

        Returns:
            Full path to library or None if not found
        """
        for search_path in self.library_paths:
            lib_path = os.path.join(search_path, lib_name)
            if os.path.exists(lib_path):
                return lib_path

        # Also try without directory component if lib_name is absolute
        if lib_name.startswith("/") and os.path.exists(lib_name):
            return lib_name

        return None

    def _get_library_symbols(self, lib_path: str) -> LibrarySymbols:
        """Get symbols provided by a library.

        Args:
            lib_path: Path to the library

        Returns:
            LibrarySymbols object with provided/required symbols
        """
        # Check cache first
        if lib_path in self._symbol_cache:
            return self._symbol_cache[lib_path]

        lib_symbols = LibrarySymbols(path=lib_path)

        try:
            parser = MinimalELFParser(lib_path)

            # Get version requirements (symbols this library requires)
            for version in parser.get_version_requirements():
                lib_symbols.required_symbols.add(version)
                lib_symbols.versioned_symbols[version] = version

            # For provided symbols, we'd need to parse .dynsym
            # This is a simplified version
            # In a full implementation, we'd extract exported symbols
            # For now, mark GLIBC versions as "provided" if this is libc
            if "libc.so" in lib_path:
                for version in parser.get_version_requirements():
                    lib_symbols.provided_symbols.add(version)

        except Exception:
            pass

        # Cache the result
        self._symbol_cache[lib_path] = lib_symbols

        return lib_symbols

    def check_symbol_availability(
        self,
        symbol_name: str,
        version_requirement: Optional[str] = None
    ) -> bool:
        """Check if a symbol is available in system libraries.

        Args:
            symbol_name: Name of the symbol
            version_requirement: Optional version requirement (e.g., "GLIBC_2.34")

        Returns:
            True if symbol is available, False otherwise
        """
        # Search all system libraries
        for search_path in self.library_paths:
            if not os.path.isdir(search_path):
                continue

            try:
                for entry in os.listdir(search_path):
                    if entry.endswith(".so") or ".so." in entry:
                        lib_path = os.path.join(search_path, entry)
                        if os.path.isfile(lib_path):
                            lib_symbols = self._get_library_symbols(lib_path)
                            if symbol_name in lib_symbols.provided_symbols:
                                if version_requirement:
                                    # Check version compatibility
                                    sym_version = lib_symbols.versioned_symbols.get(symbol_name)
                                    if sym_version and self._version_gte(sym_version, version_requirement):
                                        return True
                                else:
                                    return True
            except (OSError, PermissionError):
                continue

        return False

    @staticmethod
    def _version_gte(v1: str, v2: str) -> bool:
        """Compare version strings (simple comparison).

        Args:
            v1: First version (e.g., "GLIBC_2.35")
            v2: Second version (e.g., "GLIBC_2.34")

        Returns:
            True if v1 >= v2
        """
        try:
            # Extract numeric parts (e.g., "GLIBC_2.34" -> [2, 34])
            parts1 = [int(x) for x in v1.split("_")[-1].split(".")]
            parts2 = [int(x) for x in v2.split("_")[-1].split(".")]
            return parts1 >= parts2
        except (ValueError, IndexError):
            return False


def resolve_symbol(
    symbol_name: str,
    library_paths: Optional[List[str]] = None
) -> Optional[str]:
    """Convenience function to resolve a single symbol.

    Args:
        symbol_name: Name of the symbol to resolve
        library_paths: Optional list of library search paths

    Returns:
        Path to library providing the symbol, or None if not found
    """
    resolver = SymbolResolver(library_paths)

    # Search for the symbol
    for search_path in resolver.library_paths:
        if not os.path.isdir(search_path):
            continue

        try:
            for entry in os.listdir(search_path):
                if entry.endswith(".so") or ".so." in entry:
                    lib_path = os.path.join(search_path, entry)
                    if os.path.isfile(lib_path):
                        lib_symbols = resolver._get_library_symbols(lib_path)
                        if symbol_name in lib_symbols.provided_symbols:
                            return lib_path
        except (OSError, PermissionError):
            continue

    return None
