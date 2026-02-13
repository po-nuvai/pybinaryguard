"""Tests for the RuleEngine class."""

from __future__ import annotations

from typing import List

import pytest

from pybinaryguard.models.enums import Architecture, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule
from pybinaryguard.rules.engine import RuleEngine


# ---------------------------------------------------------------------------
# Stub rules for testing the engine
# ---------------------------------------------------------------------------


class AlwaysPassRule(Rule):
    rule_id = "ALWAYS_PASS"
    description = "Always returns PASSED"

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        return [
            Finding(
                rule_id=self.rule_id,
                severity=Severity.PASSED,
                title="Everything OK",
                explanation="No issues found.",
            )
        ]


class AlwaysWarnRule(Rule):
    rule_id = "ALWAYS_WARN"
    description = "Always returns WARNING"

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        return [
            Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title="Minor issue",
                explanation="Something is slightly off.",
            )
        ]


class AlwaysCriticalRule(Rule):
    rule_id = "ALWAYS_CRITICAL"
    description = "Always returns CRITICAL"

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        return [
            Finding(
                rule_id=self.rule_id,
                severity=Severity.CRITICAL,
                title="Major issue",
                explanation="Something is seriously wrong.",
            )
        ]


class NotApplicableRule(Rule):
    rule_id = "NOT_APPLICABLE"
    description = "Never applicable"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return False

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        return [
            Finding(
                rule_id=self.rule_id,
                severity=Severity.CRITICAL,
                title="Should never appear",
                explanation="This should never be in the results.",
            )
        ]


class BrokenRule(Rule):
    rule_id = "BROKEN_RULE"
    description = "A rule that always raises"

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        raise RuntimeError("This rule is broken!")


def _make_profile() -> SystemProfile:
    return SystemProfile(
        python_version=(3, 12, 0),
        architecture=Architecture.X86_64,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuleEngine:
    def test_register_and_list_rules(self) -> None:
        engine = RuleEngine()
        engine.register(AlwaysPassRule())
        engine.register(AlwaysWarnRule())
        assert len(engine.rules) == 2

    def test_register_rejects_non_rule(self) -> None:
        engine = RuleEngine()
        with pytest.raises(TypeError, match="Expected a Rule instance"):
            engine.register("not a rule")  # type: ignore[arg-type]

    def test_evaluate_returns_findings(self) -> None:
        engine = RuleEngine()
        engine.register(AlwaysPassRule())
        engine.register(AlwaysWarnRule())

        profile = _make_profile()
        findings = engine.evaluate(profile, [])
        assert len(findings) == 2

    def test_evaluate_sorts_by_severity(self) -> None:
        engine = RuleEngine()
        engine.register(AlwaysPassRule())
        engine.register(AlwaysWarnRule())
        engine.register(AlwaysCriticalRule())

        profile = _make_profile()
        findings = engine.evaluate(profile, [])

        # CRITICAL should come first, then WARNING, then PASSED
        assert findings[0].severity == Severity.CRITICAL
        assert findings[1].severity == Severity.WARNING
        assert findings[2].severity == Severity.PASSED

    def test_ignored_rules_are_skipped(self) -> None:
        engine = RuleEngine(ignored_rules={"ALWAYS_WARN"})
        engine.register(AlwaysPassRule())
        engine.register(AlwaysWarnRule())

        profile = _make_profile()
        findings = engine.evaluate(profile, [])

        assert len(findings) == 1
        assert findings[0].rule_id == "ALWAYS_PASS"

    def test_ignored_rules_property_getter(self) -> None:
        engine = RuleEngine(ignored_rules={"SOME_RULE"})
        assert "SOME_RULE" in engine.ignored_rules

    def test_ignored_rules_property_setter(self) -> None:
        engine = RuleEngine()
        engine.ignored_rules = {"ALWAYS_CRITICAL"}
        engine.register(AlwaysCriticalRule())

        profile = _make_profile()
        findings = engine.evaluate(profile, [])
        assert len(findings) == 0

    def test_not_applicable_rules_are_skipped(self) -> None:
        engine = RuleEngine()
        engine.register(AlwaysPassRule())
        engine.register(NotApplicableRule())

        profile = _make_profile()
        findings = engine.evaluate(profile, [])

        assert len(findings) == 1
        assert findings[0].rule_id == "ALWAYS_PASS"

    def test_broken_rule_does_not_crash_engine(self) -> None:
        engine = RuleEngine()
        engine.register(AlwaysPassRule())
        engine.register(BrokenRule())

        profile = _make_profile()
        # Should not raise, broken rule is silently skipped
        findings = engine.evaluate(profile, [])
        assert len(findings) == 1
        assert findings[0].rule_id == "ALWAYS_PASS"

    def test_with_builtin_rules_loads_rules(self) -> None:
        engine = RuleEngine.with_builtin_rules()
        assert len(engine.rules) > 0
        # Check that some well-known rules are present
        rule_ids = {r.rule_id for r in engine.rules}
        assert "GLIBC_VERSION_MISMATCH" in rule_ids
        assert "ARCH_MISMATCH" in rule_ids

    def test_with_builtin_rules_respects_ignored_rules(self) -> None:
        engine = RuleEngine.with_builtin_rules(
            ignored_rules={"GLIBC_VERSION_MISMATCH"}
        )
        assert "GLIBC_VERSION_MISMATCH" in engine.ignored_rules

    def test_evaluate_empty_engine(self) -> None:
        engine = RuleEngine()
        profile = _make_profile()
        findings = engine.evaluate(profile, [])
        assert findings == []

    def test_load_builtin_rules_populates_engine(self) -> None:
        engine = RuleEngine()
        assert len(engine.rules) == 0
        engine.load_builtin_rules()
        assert len(engine.rules) > 10  # We know there are 19 built-in rules

    def test_rules_property_returns_copy(self) -> None:
        engine = RuleEngine()
        engine.register(AlwaysPassRule())
        rules = engine.rules
        rules.clear()  # Should not affect the engine
        assert len(engine.rules) == 1
