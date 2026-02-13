"""Tests for the plugin system: HookRegistry and plugin loader."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.plugins.hooks import HookRegistry
from pybinaryguard.plugins.loader import (
    _load_builtin_contrib_plugins,
    discover_plugins,
)
from pybinaryguard.probes.base import ProbeBase
from pybinaryguard.rules.base import Rule


# ---------------------------------------------------------------------------
# Stub probe and rule for registry tests
# ---------------------------------------------------------------------------


class StubProbe(ProbeBase):
    name = "stub_probe"

    def is_applicable(self) -> bool:
        return True

    def collect(self) -> Dict[str, Any]:
        return {"test_key": "test_value"}


class AnotherStubProbe(ProbeBase):
    name = "another_stub_probe"

    def is_applicable(self) -> bool:
        return True

    def collect(self) -> Dict[str, Any]:
        return {}


class StubRule(Rule):
    rule_id = "STUB_RULE"
    description = "A stub rule for testing"

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        return []


class AnotherStubRule(Rule):
    rule_id = "ANOTHER_STUB_RULE"
    description = "Another stub rule for testing"

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        return []


# ---------------------------------------------------------------------------
# HookRegistry -- probe registration
# ---------------------------------------------------------------------------


class TestHookRegistryProbes:
    def test_add_probe(self) -> None:
        registry = HookRegistry()
        probe = StubProbe()
        registry.add_probe(probe)
        assert len(registry.probes) == 1
        assert registry.probes[0].name == "stub_probe"

    def test_add_probe_rejects_non_probe(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="ProbeBase instance"):
            registry.add_probe("not_a_probe")  # type: ignore[arg-type]

    def test_add_duplicate_probe_name_raises(self) -> None:
        registry = HookRegistry()
        registry.add_probe(StubProbe())
        with pytest.raises(ValueError, match="already registered"):
            registry.add_probe(StubProbe())

    def test_probe_count(self) -> None:
        registry = HookRegistry()
        assert registry.probe_count == 0
        registry.add_probe(StubProbe())
        assert registry.probe_count == 1

    def test_probes_returns_copy(self) -> None:
        registry = HookRegistry()
        registry.add_probe(StubProbe())
        probes = registry.probes
        probes.clear()
        assert registry.probe_count == 1  # Original not affected


# ---------------------------------------------------------------------------
# HookRegistry -- rule registration
# ---------------------------------------------------------------------------


class TestHookRegistryRules:
    def test_add_rule(self) -> None:
        registry = HookRegistry()
        rule = StubRule()
        registry.add_rule(rule)
        assert len(registry.rules) == 1
        assert registry.rules[0].rule_id == "STUB_RULE"

    def test_add_rule_rejects_non_rule(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="Rule instance"):
            registry.add_rule("not_a_rule")  # type: ignore[arg-type]

    def test_add_duplicate_rule_id_raises(self) -> None:
        registry = HookRegistry()
        registry.add_rule(StubRule())
        with pytest.raises(ValueError, match="already registered"):
            registry.add_rule(StubRule())

    def test_rule_count(self) -> None:
        registry = HookRegistry()
        assert registry.rule_count == 0
        registry.add_rule(StubRule())
        assert registry.rule_count == 1


# ---------------------------------------------------------------------------
# HookRegistry -- callables
# ---------------------------------------------------------------------------


class TestHookRegistryCallables:
    def test_add_board_detector(self) -> None:
        registry = HookRegistry()
        detector = lambda data: "TestBoard" if "test" in data else None
        registry.add_board_detector(detector)
        assert len(registry.board_detectors) == 1

    def test_add_board_detector_rejects_non_callable(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="callable"):
            registry.add_board_detector(42)  # type: ignore[arg-type]

    def test_add_framework_checker(self) -> None:
        registry = HookRegistry()
        checker = lambda profile, packages: []
        registry.add_framework_checker(checker)
        assert len(registry.framework_checkers) == 1

    def test_add_framework_checker_rejects_non_callable(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="callable"):
            registry.add_framework_checker("not_callable")  # type: ignore[arg-type]

    def test_add_reporter(self) -> None:
        registry = HookRegistry()
        reporter = lambda findings, profile: "report"
        registry.add_reporter(reporter)
        assert len(registry.reporters) == 1

    def test_add_reporter_rejects_non_callable(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="callable"):
            registry.add_reporter(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HookRegistry -- scan hooks
# ---------------------------------------------------------------------------


class TestHookRegistryScanHooks:
    def test_pre_scan_hook(self) -> None:
        registry = HookRegistry()
        hook = lambda profile: None
        registry.pre_scan(hook)
        assert len(registry.pre_scan_hooks) == 1

    def test_post_scan_hook(self) -> None:
        registry = HookRegistry()
        hook = lambda profile: None
        registry.post_scan(hook)
        assert len(registry.post_scan_hooks) == 1

    def test_pre_scan_rejects_non_callable(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="callable"):
            registry.pre_scan("not_a_hook")  # type: ignore[arg-type]

    def test_post_scan_rejects_non_callable(self) -> None:
        registry = HookRegistry()
        with pytest.raises(TypeError, match="callable"):
            registry.post_scan(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HookRegistry -- summary and repr
# ---------------------------------------------------------------------------


class TestHookRegistrySummary:
    def test_summary_empty(self) -> None:
        registry = HookRegistry()
        s = registry.summary()
        assert s["probes"] == 0
        assert s["rules"] == 0
        assert s["board_detectors"] == 0

    def test_summary_with_items(self) -> None:
        registry = HookRegistry()
        registry.add_probe(StubProbe())
        registry.add_rule(StubRule())
        s = registry.summary()
        assert s["probes"] == 1
        assert s["rules"] == 1

    def test_repr_empty(self) -> None:
        registry = HookRegistry()
        r = repr(registry)
        assert "HookRegistry" in r

    def test_repr_with_items(self) -> None:
        registry = HookRegistry()
        registry.add_probe(StubProbe())
        r = repr(registry)
        assert "probes=1" in r


# ---------------------------------------------------------------------------
# discover_plugins
# ---------------------------------------------------------------------------


class TestDiscoverPlugins:
    def test_discover_returns_registry(self) -> None:
        # With both contrib and external disabled, should get empty registry
        registry = discover_plugins(load_contrib=False, load_external=False)
        assert isinstance(registry, HookRegistry)
        assert registry.probe_count == 0
        assert registry.rule_count == 0

    def test_discover_with_contrib_disabled(self) -> None:
        registry = discover_plugins(load_contrib=False, load_external=False)
        assert isinstance(registry, HookRegistry)

    @mock.patch("pybinaryguard.plugins.loader._load_entry_point_plugins")
    @mock.patch("pybinaryguard.plugins.loader._load_builtin_contrib_plugins")
    def test_discover_calls_both_loaders(
        self, mock_contrib: Any, mock_external: Any
    ) -> None:
        discover_plugins(load_contrib=True, load_external=True)
        mock_external.assert_called_once()
        mock_contrib.assert_called_once()

    @mock.patch("pybinaryguard.plugins.loader._load_entry_point_plugins")
    @mock.patch("pybinaryguard.plugins.loader._load_builtin_contrib_plugins")
    def test_discover_skips_external_when_disabled(
        self, mock_contrib: Any, mock_external: Any
    ) -> None:
        discover_plugins(load_contrib=True, load_external=False)
        mock_external.assert_not_called()
        mock_contrib.assert_called_once()


# ---------------------------------------------------------------------------
# _load_builtin_contrib_plugins
# ---------------------------------------------------------------------------


class TestLoadBuiltinContribPlugins:
    @mock.patch("importlib.import_module", side_effect=ImportError("no module"))
    def test_handles_missing_contrib_modules(self, mock_import: Any) -> None:
        registry = HookRegistry()
        # Should not raise even if all contrib modules are missing
        _load_builtin_contrib_plugins(registry)
        # Since all imports fail, nothing should be registered
        assert registry.probe_count == 0

    @mock.patch("importlib.import_module")
    def test_handles_contrib_without_register(self, mock_import: Any) -> None:
        # Create a module mock that has no 'register' attribute
        mock_module = mock.MagicMock(spec=[])
        del mock_module.register  # Ensure no register attribute
        mock_import.return_value = mock_module

        registry = HookRegistry()
        _load_builtin_contrib_plugins(registry)
        # Should not raise, just skip modules without register
        assert registry.probe_count == 0

    @mock.patch("importlib.import_module")
    def test_handles_contrib_register_that_raises(self, mock_import: Any) -> None:
        mock_module = mock.MagicMock()
        mock_module.register.side_effect = RuntimeError("Registration failed!")
        mock_import.return_value = mock_module

        registry = HookRegistry()
        # Should not raise even if register() raises
        _load_builtin_contrib_plugins(registry)


# ---------------------------------------------------------------------------
# Integration: multiple probes and rules registered
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    def test_multiple_probes_and_rules(self) -> None:
        registry = HookRegistry()
        registry.add_probe(StubProbe())
        registry.add_probe(AnotherStubProbe())
        registry.add_rule(StubRule())
        registry.add_rule(AnotherStubRule())

        assert registry.probe_count == 2
        assert registry.rule_count == 2
        assert len(registry.probes) == 2
        assert len(registry.rules) == 2

    def test_full_registry_summary(self) -> None:
        registry = HookRegistry()
        registry.add_probe(StubProbe())
        registry.add_rule(StubRule())
        registry.add_board_detector(lambda data: None)
        registry.add_framework_checker(lambda p, pkgs: [])
        registry.add_reporter(lambda f, p: "")
        registry.pre_scan(lambda p: None)
        registry.post_scan(lambda p: None)

        s = registry.summary()
        assert s["probes"] == 1
        assert s["rules"] == 1
        assert s["board_detectors"] == 1
        assert s["framework_checkers"] == 1
        assert s["reporters"] == 1
        assert s["pre_scan_hooks"] == 1
        assert s["post_scan_hooks"] == 1
