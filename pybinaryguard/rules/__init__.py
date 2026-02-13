"""Rules subsystem for PyBinaryGuard.

This package provides the rule engine and all built-in compatibility rules
that PyBinaryGuard uses to detect binary incompatibilities.

Quick start::

    from pybinaryguard.rules import RuleEngine

    engine = RuleEngine.with_builtin_rules()
    findings = engine.evaluate(system_profile, packages)
"""

from __future__ import annotations

from pybinaryguard.rules.base import Rule
from pybinaryguard.rules.builtin import get_all_builtin_rules
from pybinaryguard.rules.engine import RuleEngine

__all__ = [
    "Rule",
    "RuleEngine",
    "get_all_builtin_rules",
]
