"""Tests for the PyBinaryGuard Agent SDK."""

from __future__ import annotations

import json
from unittest import mock

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding


def _finding(rule_id: str, severity: Severity = Severity.CRITICAL, **kw) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=severity,
        title=f"Test: {rule_id}",
        explanation="Test explanation",
        **kw,
    )


# =========================================================================
# ActionRecommender
# =========================================================================

class TestActionRecommender:
    def test_recommend_cuda_mismatch(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [_finding("CUDA_RUNTIME_MISMATCH", package="torch")]
        actions = r.recommend(findings)
        assert len(actions) >= 1
        assert actions[0].action_type == "reinstall"
        assert actions[0].target == "torch"

    def test_recommend_glibc_mismatch(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [_finding("GLIBC_VERSION_MISMATCH", package="numpy")]
        actions = r.recommend(findings)
        assert len(actions) >= 1
        assert actions[0].action_type == "downgrade"

    def test_recommend_passed_finding_skipped(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [_finding("GLIBC_VERSION_MISMATCH", severity=Severity.PASSED)]
        actions = r.recommend(findings)
        assert len(actions) == 0

    def test_recommend_deduplicates_targets(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [
            _finding("CUDA_RUNTIME_MISMATCH", package="torch"),
            _finding("CUDA_RUNTIME_MISMATCH", package="torch"),
        ]
        actions = r.recommend(findings)
        assert len(actions) == 1

    def test_recommend_safety_classification(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [_finding("CUDA_DRIVER_TOO_OLD", package="torch")]
        actions = r.recommend(findings)
        assert len(actions) >= 1
        assert actions[0].safety == "dangerous"

    def test_recommend_fallback_to_suggestion(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [
            _finding(
                "SOME_UNKNOWN_RULE_XYZ",
                package="foo",
                severity=Severity.WARNING,
            ),
        ]
        # Set a suggestion on the finding
        findings[0].suggestion = "pip install foo==1.0"
        actions = r.recommend(findings)
        assert len(actions) == 1
        assert actions[0].action_type == "fix"

    def test_action_to_dict(self):
        from pybinaryguard.agent.recommender import RecommendedAction

        a = RecommendedAction(
            action_type="reinstall",
            target="torch",
            command="pip install torch",
            reason="test",
            safety="safe",
        )
        d = a.to_dict()
        assert d["action_type"] == "reinstall"
        assert d["target"] == "torch"
        assert d["safety"] == "safe"

    def test_priority_ordering(self):
        from pybinaryguard.agent.recommender import ActionRecommender

        r = ActionRecommender()
        findings = [
            _finding("GLIBC_VERSION_MISMATCH", severity=Severity.WARNING, package="a"),
            _finding("CUDA_RUNTIME_MISMATCH", severity=Severity.CRITICAL, package="b"),
        ]
        actions = r.recommend(findings)
        # Critical should be first (lower priority number)
        assert actions[0].priority < actions[1].priority


# =========================================================================
# Tool Schema Export
# =========================================================================

class TestToolSchema:
    def test_openai_format(self):
        from pybinaryguard.agent.schema import get_tool_descriptors

        tools = get_tool_descriptors(format="openai")
        assert isinstance(tools, list)
        assert len(tools) >= 4
        # OpenAI format has type: "function"
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_mcp_format(self):
        from pybinaryguard.agent.schema import get_tool_descriptors

        tools = get_tool_descriptors(format="mcp")
        assert isinstance(tools, list)
        for tool in tools:
            assert "name" in tool
            assert "inputSchema" in tool

    def test_json_schema_format(self):
        from pybinaryguard.agent.schema import get_tool_descriptors

        tools = get_tool_descriptors(format="json_schema")
        assert isinstance(tools, list)
        for tool in tools:
            assert "name" in tool
            assert "parameters" in tool

    def test_invalid_format_raises(self):
        import pytest
        from pybinaryguard.agent.schema import get_tool_descriptors

        with pytest.raises(ValueError, match="Unknown format"):
            get_tool_descriptors(format="invalid")

    def test_export_tool_schema_returns_json_string(self):
        from pybinaryguard.agent.schema import export_tool_schema

        result = export_tool_schema(format="openai")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_scan_tool_has_parameters(self):
        from pybinaryguard.agent.schema import get_tool_descriptors

        tools = get_tool_descriptors(format="json_schema")
        scan_tool = next(t for t in tools if t["name"] == "pybinaryguard_scan")
        props = scan_tool["parameters"]["properties"]
        assert "scan_mode" in props
        assert "packages" in props

    def test_simulate_tool_exists(self):
        from pybinaryguard.agent.schema import get_tool_descriptors

        tools = get_tool_descriptors(format="json_schema")
        names = [t["name"] for t in tools]
        assert "pybinaryguard_simulate_install" in names


# =========================================================================
# Simulator (simulate_install)
# =========================================================================

class TestSimulator:
    def test_parse_wheel_filename(self):
        from pybinaryguard.agent.simulator import _parse_wheel_filename

        w = _parse_wheel_filename(
            "torch-2.4.0+cu124-cp312-cp312-manylinux_2_17_x86_64.whl"
        )
        assert w is not None
        assert w.name == "torch"
        assert w.version == "2.4.0+cu124"
        assert w.python_tag == "cp312"
        assert w.abi_tag == "cp312"
        assert w.cuda_variant == (12, 4)
        assert w.required_glibc == (2, 17)
        assert w.target_arch == "x86_64"

    def test_parse_wheel_filename_no_cuda(self):
        from pybinaryguard.agent.simulator import _parse_wheel_filename

        w = _parse_wheel_filename(
            "numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.whl"
        )
        assert w is not None
        assert w.name == "numpy"
        assert w.cuda_variant is None

    def test_parse_wheel_filename_aarch64(self):
        from pybinaryguard.agent.simulator import _parse_wheel_filename

        w = _parse_wheel_filename(
            "torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.whl"
        )
        assert w is not None
        assert w.target_arch == "aarch64"

    def test_parse_wheel_filename_invalid(self):
        from pybinaryguard.agent.simulator import _parse_wheel_filename

        assert _parse_wheel_filename("not_a_wheel.tar.gz") is None

    def test_parse_version_spec_with_cuda(self):
        from pybinaryguard.agent.simulator import _parse_version_spec

        name, version, cuda = _parse_version_spec("torch==2.4.0+cu124")
        assert name == "torch"
        assert version == "2.4.0+cu124"
        assert cuda == (12, 4)

    def test_parse_version_spec_no_version(self):
        from pybinaryguard.agent.simulator import _parse_version_spec

        name, version, cuda = _parse_version_spec("torch")
        assert name == "torch"
        assert version == ""
        assert cuda is None

    def test_check_python_compat_match(self):
        from pybinaryguard.agent.simulator import _check_python_compat

        result = _check_python_compat("cp312", (3, 12, 1))
        assert result is None  # Compatible

    def test_check_python_compat_mismatch(self):
        from pybinaryguard.agent.simulator import _check_python_compat

        result = _check_python_compat("cp311", (3, 12, 1))
        assert result is not None
        assert result["type"] == "python_version_mismatch"

    def test_check_python_compat_universal(self):
        from pybinaryguard.agent.simulator import _check_python_compat

        result = _check_python_compat("py3", (3, 12, 1))
        assert result is None

    def test_check_arch_compat_match(self):
        from pybinaryguard.agent.simulator import _check_arch_compat

        assert _check_arch_compat("x86_64", "x86_64") is None

    def test_check_arch_compat_mismatch(self):
        from pybinaryguard.agent.simulator import _check_arch_compat

        result = _check_arch_compat("x86_64", "aarch64")
        assert result is not None
        assert result["severity"] == "critical"

    def test_check_glibc_compat_ok(self):
        from pybinaryguard.agent.simulator import _check_glibc_compat

        assert _check_glibc_compat((2, 17), (2, 35)) is None

    def test_check_glibc_compat_too_old(self):
        from pybinaryguard.agent.simulator import _check_glibc_compat

        result = _check_glibc_compat((2, 35), (2, 17))
        assert result is not None
        assert result["type"] == "glibc_too_old"

    def test_check_cuda_compat_no_gpu(self):
        from pybinaryguard.agent.simulator import _check_cuda_compat

        result = _check_cuda_compat((12, 4), None, gpu_available=False)
        assert result is not None
        assert result["type"] == "no_gpu"

    def test_check_cuda_compat_major_mismatch(self):
        from pybinaryguard.agent.simulator import _check_cuda_compat

        result = _check_cuda_compat((12, 4), (11, 8), gpu_available=True)
        assert result is not None
        assert result["type"] == "cuda_major_mismatch"

    def test_check_cuda_compat_ok(self):
        from pybinaryguard.agent.simulator import _check_cuda_compat

        result = _check_cuda_compat((12, 1), (12, 4), gpu_available=True)
        assert result is None

    def test_check_musl_glibc_conflict(self):
        from pybinaryguard.agent.simulator import _check_musl_compat

        result = _check_musl_compat(
            "manylinux_2_17_x86_64",
            system_glibc=None,
            system_musl=(1, 2),
        )
        assert result is not None
        assert result["type"] == "glibc_on_musl"

    def test_simulate_wheel_filename(self):
        from pybinaryguard.agent.simulator import simulate
        from pybinaryguard.models.system import SystemProfile

        with mock.patch("pybinaryguard.scanner.Scanner") as MockScanner:
            instance = MockScanner.return_value
            instance.get_profile.return_value = SystemProfile(
                python_version=(3, 12, 1),
                architecture=mock.MagicMock(value="x86_64"),
                glibc_version=(2, 35),
                gpu_available=True,
                cuda_runtime_version=(12, 4),
            )

            result = simulate(
                "torch-2.4.0+cu124-cp312-cp312-manylinux_2_17_x86_64.whl"
            )
            assert result.predicted_compatible is True
            assert result.confidence >= 0.9
            assert result.parsed_tags is not None
            assert result.parsed_tags["name"] == "torch"

    def test_simulate_incompatible_wheel(self):
        from pybinaryguard.agent.simulator import simulate
        from pybinaryguard.models.system import SystemProfile

        with mock.patch("pybinaryguard.scanner.Scanner") as MockScanner:
            instance = MockScanner.return_value
            instance.get_profile.return_value = SystemProfile(
                python_version=(3, 11, 0),  # Wrong Python
                architecture=mock.MagicMock(value="aarch64"),  # Wrong arch
                glibc_version=(2, 17),
                gpu_available=False,
            )

            result = simulate(
                "torch-2.4.0+cu124-cp312-cp312-manylinux_2_17_x86_64.whl"
            )
            assert result.predicted_compatible is False
            assert len(result.blockers) >= 1

    def test_simulate_name_only(self):
        from pybinaryguard.agent.simulator import simulate
        from pybinaryguard.models.system import SystemProfile

        with mock.patch("pybinaryguard.scanner.Scanner") as MockScanner:
            instance = MockScanner.return_value
            instance.get_profile.return_value = SystemProfile(
                python_version=(3, 12, 1),
                glibc_version=(2, 35),
            )

            result = simulate("torch")
            assert result.predicted_compatible is True
            assert result.confidence <= 0.6
            assert result.parsed_tags["name"] == "torch"


# =========================================================================
# Tool Interface (scan, check, doctor)
# =========================================================================

class TestToolInterface:
    def test_actionable_report_to_dict(self):
        from pybinaryguard.agent.tool_interface import ActionableReport

        report = ActionableReport(
            health_score=84,
            risk_level="medium",
            total_packages=100,
            packages_scanned=50,
            scan_duration_ms=1500.0,
        )
        d = report.to_dict()
        assert d["health_score"] == 84
        assert d["risk_level"] == "medium"
        assert d["issue_count"] == 0

    def test_actionable_report_to_json(self):
        from pybinaryguard.agent.tool_interface import ActionableReport

        report = ActionableReport(
            health_score=100,
            risk_level="none",
            total_packages=10,
            packages_scanned=10,
            scan_duration_ms=500.0,
        )
        result = json.loads(report.to_json())
        assert result["health_score"] == 100

    def test_agent_check_result_to_dict(self):
        from pybinaryguard.agent.tool_interface import AgentCheckResult

        result = AgentCheckResult(
            package="torch",
            compatible=True,
            risk_level="none",
        )
        d = result.to_dict()
        assert d["package"] == "torch"
        assert d["compatible"] is True

    def test_agent_simulate_result_to_dict(self):
        from pybinaryguard.agent.tool_interface import AgentSimulateResult

        result = AgentSimulateResult(
            package_spec="torch==2.4.0",
            predicted_compatible=True,
            confidence=0.9,
            risk_level="none",
        )
        d = result.to_dict()
        assert d["predicted_compatible"] is True
        assert d["confidence"] == 0.9

    def test_agent_doctor_result_to_dict(self):
        from pybinaryguard.agent.tool_interface import AgentDoctorResult

        result = AgentDoctorResult(
            error_message="GLIBC not found",
            diagnosis="GLIBC mismatch",
            auto_fix_safe=True,
            fix_plan=[{"step": 1, "action": "pip install foo", "safety": "safe"}],
        )
        d = result.to_dict()
        assert d["auto_fix_safe"] is True
        assert len(d["fix_plan"]) == 1

    def test_risk_level_calculation(self):
        from pybinaryguard.agent.tool_interface import _risk_level

        # No findings = none
        assert _risk_level([]) == "none"

        # 1 critical = high
        assert _risk_level([_finding("X")]) == "high"

        # 3 criticals = critical
        assert _risk_level([_finding("X"), _finding("Y"), _finding("Z")]) == "critical"

        # Warnings only
        assert _risk_level([
            _finding("X", severity=Severity.WARNING),
        ]) == "low"

        # 3+ warnings = medium
        assert _risk_level([
            _finding("X", severity=Severity.WARNING),
            _finding("Y", severity=Severity.WARNING),
            _finding("Z", severity=Severity.WARNING),
        ]) == "medium"

    def test_findings_to_issues_skips_passed(self):
        from pybinaryguard.agent.tool_interface import _findings_to_issues

        findings = [
            _finding("A", severity=Severity.PASSED),
            _finding("B", severity=Severity.CRITICAL),
        ]
        issues = _findings_to_issues(findings)
        assert len(issues) == 1
        assert issues[0]["type"] == "B"

    def test_classify_actions(self):
        from pybinaryguard.agent.recommender import RecommendedAction
        from pybinaryguard.agent.tool_interface import _classify_actions

        actions = [
            RecommendedAction("reinstall", "a", "pip install a", "r", "safe"),
            RecommendedAction("configure", "b", "# fix", "r", "dangerous"),
            RecommendedAction("upgrade", "c", "pip install c", "r", "review"),
        ]
        safe, review, dangerous = _classify_actions(actions)
        assert len(safe) == 1
        assert len(review) == 1
        assert len(dangerous) == 1

    def test_as_agent_tool(self):
        from pybinaryguard.agent.tool_interface import as_agent_tool

        tools = as_agent_tool()
        assert "pybinaryguard_scan" in tools
        assert "pybinaryguard_check" in tools
        assert "pybinaryguard_simulate_install" in tools
        assert tools["pybinaryguard_scan"]["handler"] is not None
        assert "schema" in tools["pybinaryguard_scan"]


# =========================================================================
# Import Guard
# =========================================================================

class TestImportGuard:
    def test_enable_disable(self):
        from pybinaryguard.agent import guard

        # It auto-enables on import, so disable first
        guard.disable_guard()
        assert guard.is_active() is False

        guard.enable_guard()
        assert guard.is_active() is True

        guard.disable_guard()
        assert guard.is_active() is False

    def test_double_enable_is_safe(self):
        from pybinaryguard.agent import guard

        guard.disable_guard()
        guard.enable_guard()
        guard.enable_guard()  # Should not error
        assert guard.is_active() is True
        guard.disable_guard()

    def test_double_disable_is_safe(self):
        from pybinaryguard.agent import guard

        guard.disable_guard()
        guard.disable_guard()  # Should not error
        assert guard.is_active() is False

    def test_diagnose_import_error_glibc(self):
        from pybinaryguard.agent.guard import _diagnose_import_error

        exc = ImportError("GLIBC_2.34 not found in libfoo.so")
        exc.name = "foo"
        result = _diagnose_import_error(exc)
        assert result["category"] == "glibc_mismatch"
        assert result["type"] == "import_error"

    def test_diagnose_import_error_missing_so(self):
        from pybinaryguard.agent.guard import _diagnose_import_error

        exc = ImportError("cannot open shared object file: libcudart.so.12")
        exc.name = "torch"
        result = _diagnose_import_error(exc)
        assert result["category"] == "missing_shared_library"

    def test_diagnose_import_error_cuda(self):
        from pybinaryguard.agent.guard import _diagnose_import_error

        exc = ImportError("libcudart.so.12: cannot open shared object file")
        exc.name = "torch"
        result = _diagnose_import_error(exc)
        assert result["category"] in ("missing_shared_library", "cuda_missing")

    def test_diagnose_import_error_unknown(self):
        from pybinaryguard.agent.guard import _diagnose_import_error

        exc = ImportError("something weird happened")
        exc.name = "foo"
        result = _diagnose_import_error(exc)
        assert result["category"] == "unknown"

    def test_diagnose_os_error_so(self):
        from pybinaryguard.agent.guard import _diagnose_os_error

        exc = OSError("libfoo.so: shared object not found")
        result = _diagnose_os_error(exc)
        assert result is not None
        assert result["category"] == "shared_object_load_failure"

    def test_diagnose_os_error_unrelated(self):
        from pybinaryguard.agent.guard import _diagnose_os_error

        exc = OSError("Permission denied")
        result = _diagnose_os_error(exc)
        assert result is None

    def test_guarded_imports_context_manager(self):
        from pybinaryguard.agent.guard import guarded_imports, disable_guard

        disable_guard()

        with guarded_imports() as issues:
            pass  # No import errors

        assert len(issues) == 0

    def test_captured_issues_clear(self):
        from pybinaryguard.agent import guard

        guard.clear_captured_issues()
        assert len(guard.get_captured_issues()) == 0


# =========================================================================
# CLI commands for agent SDK
# =========================================================================

class TestAgentCLI:
    def test_export_tool_schema_parser(self):
        from pybinaryguard.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["export-tool-schema"])
        assert args.command == "export-tool-schema"

    def test_export_tool_schema_with_format(self):
        from pybinaryguard.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["export-tool-schema", "--schema-format", "mcp"])
        assert args.schema_format == "mcp"

    def test_simulate_parser(self):
        from pybinaryguard.cli.main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["simulate", "torch==2.4.0+cu124"])
        assert args.command == "simulate"
        assert args.package_spec == "torch==2.4.0+cu124"

    def test_cmd_export_tool_schema(self, capsys):
        import argparse
        from pybinaryguard.cli.commands import cmd_export_tool_schema

        args = argparse.Namespace(schema_format="openai")
        exit_code = cmd_export_tool_schema(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)

    def test_cmd_simulate(self, capsys):
        import argparse
        from pybinaryguard.cli.commands import cmd_simulate
        from pybinaryguard.models.system import SystemProfile

        with mock.patch("pybinaryguard.scanner.Scanner") as MockScanner:
            instance = MockScanner.return_value
            instance.get_profile.return_value = SystemProfile(
                python_version=(3, 12, 1),
                architecture=mock.MagicMock(value="x86_64"),
                glibc_version=(2, 35),
            )

            args = argparse.Namespace(
                package_spec="numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.whl",
                format="table",
            )
            exit_code = cmd_simulate(args)
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "COMPATIBLE" in captured.out


# =========================================================================
# Agent module __init__ imports
# =========================================================================

class TestAgentModuleImports:
    def test_import_agent_module(self):
        import pybinaryguard.agent as agent
        assert hasattr(agent, "scan")
        assert hasattr(agent, "check")
        assert hasattr(agent, "simulate_install")
        assert hasattr(agent, "doctor")
        assert hasattr(agent, "export_tool_schema")
        assert hasattr(agent, "as_agent_tool")

    def test_import_guard_module(self):
        from pybinaryguard.agent import guard
        assert hasattr(guard, "enable_guard")
        assert hasattr(guard, "disable_guard")
        assert hasattr(guard, "guarded_imports")
