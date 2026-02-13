"""Dependency graph builder for shared library dependencies.

Builds a graph of DT_NEEDED dependencies to visualize and analyze
library dependency chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from pybinaryguard.analyzers.elf_analyzer import MinimalELFParser


@dataclass
class DependencyNode:
    """A node in the dependency graph representing a library."""

    name: str  # Library name (e.g., "libc.so.6")
    path: Optional[str] = None  # Full path if found
    dependencies: List[str] = field(default_factory=list)  # DT_NEEDED entries
    unresolved: bool = False  # Whether this library was found
    circular: bool = False  # Whether this is part of a circular dependency


class DependencyGraph:
    """Dependency graph for analyzing shared library dependencies."""

    def __init__(self, library_paths: Optional[List[str]] = None):
        """Initialize the dependency graph builder.

        Args:
            library_paths: Directories to search for libraries
        """
        if library_paths is None:
            library_paths = [
                "/lib",
                "/lib64",
                "/usr/lib",
                "/usr/lib64",
            ]

        self.library_paths = [Path(p) for p in library_paths if Path(p).exists()]
        self.nodes: Dict[str, DependencyNode] = {}
        self._visited: Set[str] = set()

    def build_from_binary(self, binary_path: str) -> DependencyNode:
        """Build dependency graph starting from a binary.

        Args:
            binary_path: Path to the ELF binary

        Returns:
            Root DependencyNode for the binary
        """
        self.nodes.clear()
        self._visited.clear()

        root_name = Path(binary_path).name
        root_node = DependencyNode(name=root_name, path=binary_path)
        self.nodes[root_name] = root_node

        # Build the graph recursively
        self._build_recursive(binary_path, root_node)

        return root_node

    def _build_recursive(self, lib_path: str, node: DependencyNode) -> None:
        """Recursively build dependency graph.

        Args:
            lib_path: Path to current library
            node: DependencyNode for current library
        """
        # Avoid infinite recursion
        if lib_path in self._visited:
            node.circular = True
            return

        self._visited.add(lib_path)

        try:
            # Parse ELF to get DT_NEEDED
            parser = MinimalELFParser(lib_path)
            needed = parser.get_needed()

            node.dependencies = needed

            # Process each dependency
            for dep_name in needed:
                # Check if we already have this node
                if dep_name in self.nodes:
                    # Mark as circular if we're revisiting
                    if dep_name in self._visited:
                        self.nodes[dep_name].circular = True
                    continue

                # Create new node
                dep_node = DependencyNode(name=dep_name)

                # Try to find the library
                dep_path = self._find_library(dep_name)
                if dep_path:
                    dep_node.path = dep_path
                    self.nodes[dep_name] = dep_node
                    # Recurse
                    self._build_recursive(dep_path, dep_node)
                else:
                    # Unresolved dependency
                    dep_node.unresolved = True
                    self.nodes[dep_name] = dep_node

        except Exception:
            # If we can't parse, mark dependencies as unknown
            node.unresolved = True

    def _find_library(self, lib_name: str) -> Optional[str]:
        """Find a library in the search paths.

        Args:
            lib_name: Library name

        Returns:
            Full path or None
        """
        for search_path in self.library_paths:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)

        return None

    def get_unresolved_dependencies(self) -> List[str]:
        """Get list of unresolved dependencies.

        Returns:
            List of library names that couldn't be found
        """
        return [
            node.name
            for node in self.nodes.values()
            if node.unresolved
        ]

    def get_dependency_chain(self, target_lib: str) -> Optional[List[str]]:
        """Get the dependency chain to a specific library.

        Args:
            target_lib: Target library name

        Returns:
            List of library names in the chain, or None if not found
        """
        # BFS to find shortest path
        from collections import deque

        if not self.nodes:
            return None

        # Start from root (first node)
        root_name = next(iter(self.nodes.keys()))

        queue = deque([(root_name, [root_name])])
        visited = {root_name}

        while queue:
            current, path = queue.popleft()

            if current == target_lib:
                return path

            node = self.nodes.get(current)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        visited.add(dep)
                        queue.append((dep, path + [dep]))

        return None

    def has_circular_dependencies(self) -> bool:
        """Check if the graph has circular dependencies.

        Returns:
            True if circular dependencies exist
        """
        return any(node.circular for node in self.nodes.values())

    def to_dict(self) -> Dict[str, object]:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary with graph structure
        """
        return {
            "nodes": {
                name: {
                    "path": node.path,
                    "dependencies": node.dependencies,
                    "unresolved": node.unresolved,
                    "circular": node.circular,
                }
                for name, node in self.nodes.items()
            },
            "unresolved_count": len(self.get_unresolved_dependencies()),
            "has_circular": self.has_circular_dependencies(),
        }


def build_dependency_graph(
    binary_path: str,
    library_paths: Optional[List[str]] = None
) -> DependencyGraph:
    """Convenience function to build a dependency graph.

    Args:
        binary_path: Path to the ELF binary
        library_paths: Optional library search paths

    Returns:
        DependencyGraph instance
    """
    graph = DependencyGraph(library_paths)
    graph.build_from_binary(binary_path)
    return graph
