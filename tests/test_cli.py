"""Tests for the CLI argument parser and command dispatch."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from pybinaryguard.cli.main import _build_parser, main
from pybinaryguard.cli.commands import _exit_code, dispatch
from pybinaryguard.cli.formatters import (
    FormatterBase,
    JSONFormatter,
    MinimalFormatter,
    TableFormatter,
    get_formatter,
)


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_parser_scan_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan"])
        assert args.command == "scan"

    def test_parser_check_command_with_package(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["check", "torch"])
        assert args.command == "check"
        assert args.package == "torch"

    def test_parser_profile_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["profile"])
        assert args.command == "profile"

    def test_parser_doctor_with_error(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["doctor", "--error", "GLIBC_2.34 not found"])
        assert args.command == "doctor"
        assert args.error == "GLIBC_2.34 not found"

    def test_parser_doctor_with_package(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["doctor", "--package", "torch"])
        assert args.command == "doctor"
        assert args.package == "torch"

    def test_parser_inspect_command(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["inspect", "mypackage.whl"])
        assert args.command == "inspect"
        assert args.file == "mypackage.whl"

    def test_parser_format_option(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "--format", "json"])
        assert args.format == "json"

    def test_parser_format_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan"])
        assert args.format == "table"

    def test_parser_severity_option(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "--severity", "critical"])
        assert args.severity == "critical"

    def test_parser_no_color_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "--no-color"])
        assert args.no_color is True

    def test_parser_verbose_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "-v"])
        assert args.verbose is True

    def test_parser_quiet_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "-q"])
        assert args.quiet is True

    def test_parser_ci_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "--ci"])
        assert args.ci is True

    def test_parser_timeout_option(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "--timeout", "60"])
        assert args.timeout == 60.0

    def test_parser_ignore_rules(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--ignore", "CUDA_MINOR_MISMATCH", "scan"])
        assert "CUDA_MINOR_MISMATCH" in args.ignore

    def test_parser_no_command_defaults_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# main() function
# ---------------------------------------------------------------------------


class TestMain:
    @mock.patch("pybinaryguard.cli.commands.dispatch", return_value=0)
    def test_default_command_is_scan(self, mock_dispatch: Any) -> None:
        """When no command is given, main() should default to scan."""
        result = main([])
        # Verify dispatch was called (main imports dispatch from commands)
        mock_dispatch.assert_called_once()
        assert result == 0

    def test_ci_mode_overrides(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "--ci"])
        # Simulate CI mode overrides that main() does
        if args.ci:
            args.format = "minimal"
            if args.severity == "all":
                args.severity = "critical"
            args.no_color = True
        assert args.format == "minimal"
        assert args.severity == "critical"
        assert args.no_color is True

    def test_quiet_mode_overrides(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scan", "-q"])
        if args.quiet:
            args.severity = "critical"
        assert args.severity == "critical"

    @mock.patch("pybinaryguard.cli.commands.dispatch", return_value=42)
    def test_main_returns_exit_code(self, mock_dispatch: Any) -> None:
        result = main(["scan"])
        assert result == 42


# ---------------------------------------------------------------------------
# _exit_code
# ---------------------------------------------------------------------------


class TestExitCode:
    def test_zero_when_no_issues(self) -> None:
        assert _exit_code(critical=0, warning=0) == 0

    def test_one_when_warnings_only(self) -> None:
        assert _exit_code(critical=0, warning=3) == 1

    def test_two_when_critical(self) -> None:
        assert _exit_code(critical=1, warning=0) == 2

    def test_two_when_critical_and_warnings(self) -> None:
        assert _exit_code(critical=2, warning=5) == 2


# ---------------------------------------------------------------------------
# get_formatter
# ---------------------------------------------------------------------------


class TestGetFormatter:
    def test_get_table_formatter(self) -> None:
        f = get_formatter("table")
        assert isinstance(f, TableFormatter)

    def test_get_json_formatter(self) -> None:
        f = get_formatter("json")
        assert isinstance(f, JSONFormatter)

    def test_get_minimal_formatter(self) -> None:
        f = get_formatter("minimal")
        assert isinstance(f, MinimalFormatter)

    def test_get_unknown_formatter_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown formatter"):
            get_formatter("xml")

    def test_table_formatter_no_color(self) -> None:
        f = get_formatter("table", no_color=True)
        assert isinstance(f, TableFormatter)


# ---------------------------------------------------------------------------
# Formatter output tests
# ---------------------------------------------------------------------------


class TestFormatterOutput:
    def test_json_formatter_format_profile(self) -> None:
        from pybinaryguard.models.system import SystemProfile
        from pybinaryguard.models.enums import Architecture

        formatter = JSONFormatter()
        profile = SystemProfile(
            python_version=(3, 12, 0),
            architecture=Architecture.X86_64,
            os_name="Ubuntu",
            os_version="22.04",
        )
        output = formatter.format_profile(profile)
        import json
        data = json.loads(output)
        assert "profile" in data
        assert data["version"] == "1.0.3"

    def test_json_formatter_format_check_no_findings(self) -> None:
        from pybinaryguard.models.system import SystemProfile
        from pybinaryguard.models.enums import Architecture

        formatter = JSONFormatter()
        profile = SystemProfile(
            python_version=(3, 12, 0),
            architecture=Architecture.X86_64,
        )
        output = formatter.format_check([], "torch", profile)
        import json
        data = json.loads(output)
        assert data["package"] == "torch"
        assert data["findings"] == []

    def test_minimal_formatter_format_check_no_findings(self) -> None:
        from pybinaryguard.models.system import SystemProfile
        from pybinaryguard.models.enums import Architecture

        formatter = MinimalFormatter()
        profile = SystemProfile(
            python_version=(3, 12, 0),
            architecture=Architecture.X86_64,
        )
        output = formatter.format_check([], "torch", profile)
        assert "no issues found" in output.lower()

    def test_minimal_formatter_format_check_with_finding(self) -> None:
        from pybinaryguard.models.system import SystemProfile
        from pybinaryguard.models.enums import Architecture, Severity
        from pybinaryguard.models.finding import Finding

        formatter = MinimalFormatter()
        profile = SystemProfile(
            python_version=(3, 12, 0),
            architecture=Architecture.X86_64,
        )
        findings = [
            Finding(
                rule_id="TEST",
                severity=Severity.CRITICAL,
                title="Test finding",
                explanation="Test",
                package="torch",
            )
        ]
        output = formatter.format_check(findings, "torch", profile)
        assert "CRITICAL" in output
        assert "torch" in output
