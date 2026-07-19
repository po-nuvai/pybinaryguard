"""Tests for diagnostic utilities: findings helpers, explainer, and suggestions."""

from __future__ import annotations

from typing import Any

import pytest

from pybinaryguard.models.enums import Architecture, ContainerRuntime, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.diagnostics.findings import (
    deduplicate_findings,
    filter_findings,
    group_by_package,
    group_by_severity,
    sort_findings,
)
from pybinaryguard.diagnostics.explainer import (
    ErrorPatternDB,
    diagnose_error,
    explain_finding,
)
from pybinaryguard.diagnostics.suggestions import suggest_fix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finding(
    rule_id: str = "TEST",
    severity: Severity = Severity.WARNING,
    package: str | None = "pkg",
    **kwargs: Any,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=severity,
        title=f"{rule_id} title",
        explanation=f"{rule_id} explanation",
        package=package,
        **kwargs,
    )


def _make_profile(**kwargs: object) -> SystemProfile:
    defaults = dict(
        python_version=(3, 12, 0),
        python_abi_tag="cp312",
        python_implementation="cpython",
        python_executable="/usr/bin/python3",
        os_name="Ubuntu",
        os_version="22.04",
        architecture=Architecture.X86_64,
        glibc_version=(2, 35),
        gpu_available=False,
    )
    defaults.update(kwargs)
    return SystemProfile(**defaults)


# ---------------------------------------------------------------------------
# filter_findings
# ---------------------------------------------------------------------------


class TestFilterFindings:
    def test_filter_critical_only(self) -> None:
        findings = [
            _finding(severity=Severity.CRITICAL),
            _finding(severity=Severity.WARNING),
            _finding(severity=Severity.INFO),
            _finding(severity=Severity.PASSED),
        ]
        result = filter_findings(findings, Severity.CRITICAL)
        assert len(result) == 1
        assert result[0].severity == Severity.CRITICAL

    def test_filter_warning_includes_critical(self) -> None:
        findings = [
            _finding(severity=Severity.CRITICAL),
            _finding(severity=Severity.WARNING),
            _finding(severity=Severity.INFO),
        ]
        result = filter_findings(findings, Severity.WARNING)
        assert len(result) == 2
        severities = {f.severity for f in result}
        assert Severity.CRITICAL in severities
        assert Severity.WARNING in severities

    def test_filter_info_includes_all_except_passed(self) -> None:
        findings = [
            _finding(severity=Severity.CRITICAL),
            _finding(severity=Severity.WARNING),
            _finding(severity=Severity.INFO),
            _finding(severity=Severity.PASSED),
        ]
        result = filter_findings(findings, Severity.INFO)
        assert len(result) == 3

    def test_filter_passed_includes_everything(self) -> None:
        findings = [
            _finding(severity=Severity.CRITICAL),
            _finding(severity=Severity.PASSED),
        ]
        result = filter_findings(findings, Severity.PASSED)
        assert len(result) == 2

    def test_filter_raises_for_invalid_severity(self) -> None:
        with pytest.raises(TypeError, match="Severity enum member"):
            filter_findings([], "critical")  # type: ignore[arg-type]

    def test_filter_empty_list(self) -> None:
        result = filter_findings([], Severity.CRITICAL)
        assert result == []


# ---------------------------------------------------------------------------
# sort_findings
# ---------------------------------------------------------------------------


class TestSortFindings:
    def test_sorts_by_severity_critical_first(self) -> None:
        findings = [
            _finding(severity=Severity.PASSED, package="z"),
            _finding(severity=Severity.CRITICAL, package="a"),
            _finding(severity=Severity.WARNING, package="m"),
        ]
        result = sort_findings(findings)
        assert result[0].severity == Severity.CRITICAL
        assert result[1].severity == Severity.WARNING
        assert result[2].severity == Severity.PASSED

    def test_sorts_by_package_name_within_severity(self) -> None:
        findings = [
            _finding(severity=Severity.CRITICAL, package="torch"),
            _finding(severity=Severity.CRITICAL, package="numpy"),
        ]
        result = sort_findings(findings)
        assert result[0].package == "numpy"
        assert result[1].package == "torch"

    def test_none_package_sorts_last(self) -> None:
        findings = [
            _finding(severity=Severity.WARNING, package=None),
            _finding(severity=Severity.WARNING, package="alpha"),
        ]
        result = sort_findings(findings)
        assert result[0].package == "alpha"
        assert result[1].package is None

    def test_sort_empty_list(self) -> None:
        assert sort_findings([]) == []


# ---------------------------------------------------------------------------
# group_by_package
# ---------------------------------------------------------------------------


class TestGroupByPackage:
    def test_groups_by_package(self) -> None:
        findings = [
            _finding(package="torch"),
            _finding(package="torch"),
            _finding(package="numpy"),
        ]
        groups = group_by_package(findings)
        assert len(groups["torch"]) == 2
        assert len(groups["numpy"]) == 1

    def test_none_package_grouped_as_unknown(self) -> None:
        findings = [_finding(package=None)]
        groups = group_by_package(findings)
        assert "<unknown>" in groups
        assert len(groups["<unknown>"]) == 1

    def test_empty_findings(self) -> None:
        groups = group_by_package([])
        assert groups == {}


# ---------------------------------------------------------------------------
# group_by_severity
# ---------------------------------------------------------------------------


class TestGroupBySeverity:
    def test_groups_by_severity(self) -> None:
        findings = [
            _finding(severity=Severity.CRITICAL),
            _finding(severity=Severity.WARNING),
            _finding(severity=Severity.WARNING),
        ]
        groups = group_by_severity(findings)
        assert len(groups[Severity.CRITICAL]) == 1
        assert len(groups[Severity.WARNING]) == 2
        assert Severity.INFO not in groups

    def test_empty_findings(self) -> None:
        groups = group_by_severity([])
        assert groups == {}


# ---------------------------------------------------------------------------
# deduplicate_findings
# ---------------------------------------------------------------------------


class TestDeduplicateFindings:
    def test_removes_duplicates(self) -> None:
        findings = [
            _finding(rule_id="A", package="torch"),
            _finding(rule_id="A", package="torch"),
            _finding(rule_id="B", package="torch"),
        ]
        result = deduplicate_findings(findings)
        assert len(result) == 2

    def test_preserves_first_occurrence(self) -> None:
        f1 = _finding(rule_id="A", package="pkg", severity=Severity.CRITICAL)
        f2 = _finding(rule_id="A", package="pkg", severity=Severity.WARNING)
        result = deduplicate_findings([f1, f2])
        assert len(result) == 1
        assert result[0].severity == Severity.CRITICAL  # First occurrence kept

    def test_different_packages_not_deduplicated(self) -> None:
        findings = [
            _finding(rule_id="A", package="torch"),
            _finding(rule_id="A", package="numpy"),
        ]
        result = deduplicate_findings(findings)
        assert len(result) == 2

    def test_none_package_deduplication(self) -> None:
        findings = [
            _finding(rule_id="A", package=None),
            _finding(rule_id="A", package=None),
        ]
        result = deduplicate_findings(findings)
        assert len(result) == 1

    def test_empty_findings(self) -> None:
        assert deduplicate_findings([]) == []


# ---------------------------------------------------------------------------
# ErrorPatternDB
# ---------------------------------------------------------------------------


class TestErrorPatternDB:
    def test_diagnose_glibc_error(self) -> None:
        db = ErrorPatternDB()
        result = db.diagnose_error("ImportError: version `GLIBC_2.34' not found")
        assert result is not None
        assert result["rule_id"] == "GLIBC_VERSION_MISMATCH"
        assert "root_cause" in result

    def test_diagnose_illegal_instruction(self) -> None:
        db = ErrorPatternDB()
        result = db.diagnose_error("Illegal instruction (core dumped)")
        assert result is not None
        assert result["rule_id"] == "ILLEGAL_INSTRUCTION_RISK"

    def test_diagnose_cuda_runtime_error(self) -> None:
        db = ErrorPatternDB()
        result = db.diagnose_error("libcudart.so.12: cannot open shared object file")
        assert result is not None
        assert result["rule_id"] == "CUDA_RUNTIME_MISMATCH"

    def test_diagnose_unknown_error_returns_none(self) -> None:
        db = ErrorPatternDB()
        result = db.diagnose_error("Some completely unrelated error")
        assert result is None

    def test_diagnose_empty_string_returns_none(self) -> None:
        db = ErrorPatternDB()
        assert db.diagnose_error("") is None

    def test_add_custom_pattern(self) -> None:
        db = ErrorPatternDB()
        db.add_pattern({
            "pattern": r"CustomError: (\d+)",
            "root_cause": "Custom error",
            "rule_id": "CUSTOM_RULE",
            "explanation": "A custom error occurred.",
            "fix_hint": "Fix the custom error.",
        })
        result = db.diagnose_error("CustomError: 42")
        assert result is not None
        assert result["rule_id"] == "CUSTOM_RULE"

    def test_add_pattern_missing_key_raises(self) -> None:
        db = ErrorPatternDB()
        with pytest.raises(ValueError, match="missing required keys"):
            db.add_pattern({
                "pattern": r"foo",
                "root_cause": "bar",
                # Missing rule_id, explanation, fix_hint
            })

    def test_get_error_patterns_returns_list(self) -> None:
        db = ErrorPatternDB()
        patterns = db.get_error_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_diagnose_error_convenience(self) -> None:
        result = diagnose_error("version `GLIBC_2.34' not found")
        assert result is not None
        assert result["rule_id"] == "GLIBC_VERSION_MISMATCH"

    def test_explain_finding_with_explanation(self) -> None:
        f = Finding(
            rule_id="TEST",
            severity=Severity.WARNING,
            title="Test title",
            explanation="Test explanation.",
            package="pkg",
        )
        text = explain_finding(f)
        assert "Test explanation" in text

    def test_explain_finding_with_related_error(self) -> None:
        f = Finding(
            rule_id="TEST",
            severity=Severity.CRITICAL,
            title="Test",
            explanation="Base explanation.",
            related_error="version `GLIBC_2.34' not found",
        )
        text = explain_finding(f)
        assert "Base explanation" in text
        # The related error should add context from the error pattern DB
        assert "GLIBC" in text or "Root cause" in text


# ---------------------------------------------------------------------------
# suggest_fix
# ---------------------------------------------------------------------------


class TestSuggestFix:
    def test_suggest_glibc_fix(self) -> None:
        profile = _make_profile(glibc_version=(2, 17))
        finding = _finding(
            rule_id="GLIBC_VERSION_MISMATCH",
            package="numpy",
        )
        suggestion = suggest_fix(finding, profile)
        assert "GLIBC" in suggestion or "glibc" in suggestion.lower()
        assert "numpy" in suggestion

    def test_suggest_arch_fix(self) -> None:
        profile = _make_profile(architecture=Architecture.X86_64)
        finding = _finding(
            rule_id="ARCH_MISMATCH",
            package="torch",
        )
        suggestion = suggest_fix(finding, profile)
        assert "x86_64" in suggestion
        assert "torch" in suggestion

    def test_suggest_python_abi_fix(self) -> None:
        profile = _make_profile()
        finding = _finding(
            rule_id="PYTHON_ABI_MISMATCH",
            package="scipy",
        )
        suggestion = suggest_fix(finding, profile)
        assert "scipy" in suggestion

    def test_suggest_fix_unknown_rule_uses_fallback(self) -> None:
        profile = _make_profile()
        finding = _finding(
            rule_id="UNKNOWN_RULE_XYZ",
            suggestion="Custom suggestion text",
        )
        suggestion = suggest_fix(finding, profile)
        assert "Custom suggestion text" in suggestion

    def test_suggest_fix_unknown_rule_no_suggestion(self) -> None:
        profile = _make_profile()
        finding = Finding(
            rule_id="TOTALLY_UNKNOWN",
            severity=Severity.WARNING,
            title="Unknown",
            explanation="Unknown issue",
        )
        suggestion = suggest_fix(finding, profile)
        assert "No specific fix" in suggestion

    def test_suggest_fix_raises_for_invalid_finding(self) -> None:
        profile = _make_profile()
        with pytest.raises(TypeError, match="Finding instance"):
            suggest_fix("not_a_finding", profile)  # type: ignore[arg-type]

    def test_suggest_fix_raises_for_invalid_profile(self) -> None:
        finding = _finding()
        with pytest.raises(TypeError, match="SystemProfile instance"):
            suggest_fix(finding, "not_a_profile")  # type: ignore[arg-type]

    def test_suggest_fix_detects_ubuntu_pkg_manager(self) -> None:
        profile = _make_profile(os_name="Ubuntu")
        finding = _finding(rule_id="GLIBC_VERSION_MISMATCH", package="pkg")
        suggestion = suggest_fix(finding, profile)
        assert "apt" in suggestion

    def test_suggest_fix_detects_fedora_pkg_manager(self) -> None:
        profile = _make_profile(os_name="Fedora")
        finding = _finding(rule_id="GLIBC_VERSION_MISMATCH", package="pkg")
        suggestion = suggest_fix(finding, profile)
        assert "yum" in suggestion
