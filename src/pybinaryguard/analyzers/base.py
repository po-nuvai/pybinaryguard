"""Abstract base class for all binary analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pybinaryguard.models.package import PackageBinaryInfo


class AnalyzerBase(ABC):
    """Base class for binary analyzers.

    Every analyzer must:
    - Have a unique ``name`` attribute used for identification and logging.
    - Implement ``analyze()`` which enriches a ``PackageBinaryInfo`` in place
      and returns it.
    - Be completely read-only -- analyzers must never modify files on disk.
    - Handle all internal I/O errors gracefully rather than raising.
    """

    name: str  # Analyzer identifier -- must be set by subclasses

    @abstractmethod
    def analyze(self, package_info: PackageBinaryInfo) -> PackageBinaryInfo:
        """Analyze and enrich the *package_info* with findings.

        Parameters
        ----------
        package_info:
            The package descriptor to populate.  The analyzer reads
            binary files referenced by the descriptor, extracts metadata,
            and mutates the descriptor's fields accordingly.

        Returns
        -------
        PackageBinaryInfo
            The same *package_info* instance (mutated in place) so callers
            can chain analyzers conveniently.
        """
        ...
