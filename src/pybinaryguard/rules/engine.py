"""Rule engine -- orchestrates evaluation of all registered rules."""

from __future__ import annotations

from typing import List, Set

from pybinaryguard.models.system import SystemProfile
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.finding import Finding
from pybinaryguard.rules.base import Rule


class RuleEngine:
    """Evaluates registered rules against a system profile and package list.

    The engine holds a list of :class:`Rule` instances, runs each applicable
    rule, collects findings, and returns them sorted by severity (critical
    first).

    Example::

        engine = RuleEngine.with_builtin_rules()
        findings = engine.evaluate(profile, packages)
    """

    def __init__(self, ignored_rules: Set[str] | None = None) -> None:
        self._rules: List[Rule] = []
        self._ignored_rules: Set[str] = ignored_rules or set()

    # -- Registration -------------------------------------------------------

    def register(self, rule: Rule) -> None:
        """Add a rule to the engine.

        Args:
            rule: An instance of a :class:`Rule` subclass.

        Raises:
            TypeError: If *rule* is not a :class:`Rule` instance.
        """
        if not isinstance(rule, Rule):
            raise TypeError(
                f"Expected a Rule instance, got {type(rule).__name__}"
            )
        self._rules.append(rule)

    @property
    def rules(self) -> List[Rule]:
        """All currently registered rules."""
        return list(self._rules)

    @property
    def ignored_rules(self) -> Set[str]:
        """Rule IDs that will be skipped during evaluation."""
        return set(self._ignored_rules)

    @ignored_rules.setter
    def ignored_rules(self, value: Set[str]) -> None:
        self._ignored_rules = set(value)

    # -- Evaluation ---------------------------------------------------------

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Run all applicable rules and return sorted findings.

        Rules whose ``rule_id`` is in :attr:`ignored_rules` are skipped.
        Rules whose :meth:`~Rule.is_applicable` returns ``False`` for
        *profile* are also skipped.

        Args:
            profile: The current system profile.
            packages: List of packages with binary analysis data.

        Returns:
            Findings sorted by severity (CRITICAL first, then WARNING,
            INFO, PASSED).
        """
        findings: List[Finding] = []
        for rule in self._rules:
            if rule.rule_id in self._ignored_rules:
                continue
            if not rule.is_applicable(profile):
                continue
            try:
                rule_findings = rule.evaluate(profile, packages)
                findings.extend(rule_findings)
            except Exception:
                # Individual rule failures must not crash the whole scan.
                # In production this would be logged; for now we silently
                # skip the broken rule.
                pass
        findings.sort(key=lambda f: f.severity)
        return findings

    # -- Factory ------------------------------------------------------------

    @classmethod
    def with_builtin_rules(
        cls,
        ignored_rules: Set[str] | None = None,
    ) -> "RuleEngine":
        """Create an engine pre-loaded with all built-in rules.

        Args:
            ignored_rules: Optional set of rule IDs to skip.

        Returns:
            A fully-configured :class:`RuleEngine`.
        """
        engine = cls(ignored_rules=ignored_rules)
        engine.load_builtin_rules()
        return engine

    def load_builtin_rules(self) -> None:
        """Instantiate and register all built-in rules."""
        from pybinaryguard.rules.builtin import get_all_builtin_rules

        for rule in get_all_builtin_rules():
            self.register(rule)
