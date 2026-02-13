"""Abstract base class for all system probes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ProbeBase(ABC):
    """Base class for system probes that collect environment information.

    Every probe must:
    - Have a unique ``name`` attribute used for identification and logging.
    - Implement ``collect()`` which returns a dict whose keys correspond to
      field names on ``SystemProfile``.
    - Be completely read-only -- probes must never modify system state.
    - Handle all internal errors gracefully and return partial data rather
      than raising.
    """

    name: str  # Probe identifier -- must be set by subclasses

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect system information.

        Returns
        -------
        Dict[str, Any]
            A dictionary whose keys are ``SystemProfile`` field names and
            whose values are the detected values for those fields.  If a
            value cannot be determined, it should be omitted from the dict
            rather than set to ``None`` (unless ``None`` is the intended
            sentinel).
        """
        ...

    def is_applicable(self) -> bool:
        """Return whether this probe can run on the current system.

        The default implementation returns ``True``.  Subclasses may
        override to skip probing when the current platform is
        incompatible (e.g. GPU probe on a system with no GPU devices).
        """
        return True
