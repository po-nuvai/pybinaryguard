"""Base rule interface for compatibility checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pybinaryguard.models.system import SystemProfile
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.finding import Finding


class Rule(ABC):
    """Base class for all binary compatibility rules.

    Every rule has a unique ``rule_id`` (e.g. ``"GLIBC_VERSION_MISMATCH"``),
    a human-readable ``description``, and an ``evaluate`` method that inspects
    the system profile together with a list of package binary analyses and
    returns zero or more :class:`Finding` objects.

    Subclasses may override :meth:`is_applicable` to skip evaluation when the
    rule cannot possibly apply (e.g. CUDA rules when no GPU is present).
    """

    rule_id: str
    description: str

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Whether this rule should run given the system profile.

        Override in subclasses to gate on system capabilities such as the
        presence of a GPU, a specific libc implementation, etc.

        Args:
            profile: The current system profile.

        Returns:
            ``True`` if the rule should be evaluated; ``False`` to skip.
        """
        return True

    @abstractmethod
    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Evaluate this rule and return findings.

        Args:
            profile: The current system profile.
            packages: List of packages with their binary analysis results.

        Returns:
            A (possibly empty) list of findings produced by this rule.
        """
        ...
