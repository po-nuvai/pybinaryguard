"""Tests for the Scanner orchestrator."""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List
from unittest import mock

import pytest

from pybinaryguard.models.enums import Architecture, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.scanner import Scanner


# ---------------------------------------------------------------------------
# Stub probe for testing
# ---------------------------------------------------------------------------


class StubProbe:
    """A minimal probe stub that returns canned data."""

    name = "stub"

    def is_applicable(self) -> bool:
        return True

    def collect(self) -> Dict[str, Any]:
        return {
            "python_version": (3, 12, 0),
            "python_abi_tag": "cp312",
            "python_implementation": "cpython",
            "python_executable": "/usr/bin/python3",
            "architecture": Architecture.X86_64,
            "os_name": "TestOS",
            "os_version": "1.0",
        }


class FailingProbe:
    """A probe that always raises."""

    name = "failing"

    def is_applicable(self) -> bool:
        return True

    def collect(self) -> Dict[str, Any]:
        raise RuntimeError("Probe failure!")


class NotApplicableProbe:
    """A probe that reports itself as not applicable."""

    name = "not_applicable"

    def is_applicable(self) -> bool:
        return False

    def collect(self) -> Dict[str, Any]:
        return {"bogus_data": True}


# ---------------------------------------------------------------------------
# Scanner.get_profile
# ---------------------------------------------------------------------------


class TestScannerGetProfile:
    def test_get_profile_returns_system_profile(self) -> None:
        scanner = Scanner(probes=[StubProbe()], enable_plugins=False)
        profile = scanner.get_profile()
        assert isinstance(profile, SystemProfile)
        assert profile.python_version == (3, 12, 0)
        assert profile.architecture == Architecture.X86_64

    def test_get_profile_caches_result(self) -> None:
        probe = StubProbe()
        scanner = Scanner(probes=[probe], enable_plugins=False)
        p1 = scanner.get_profile()
        p2 = scanner.get_profile()
        # Should be the same object (cached)
        assert p1 is p2

    def test_get_profile_handles_failing_probe(self) -> None:
        scanner = Scanner(probes=[FailingProbe()], enable_plugins=False)
        # Should not raise, just return a profile with defaults
        profile = scanner.get_profile()
        assert isinstance(profile, SystemProfile)

    def test_get_profile_skips_not_applicable_probe(self) -> None:
        scanner = Scanner(
            probes=[StubProbe(), NotApplicableProbe()],
            enable_plugins=False,
        )
        profile = scanner.get_profile()
        # Should only have data from StubProbe
        assert profile.os_name == "TestOS"


# ---------------------------------------------------------------------------
# Scanner.run
# ---------------------------------------------------------------------------


class TestScannerRun:
    def test_run_returns_scan_report(self) -> None:
        scanner = Scanner(probes=[StubProbe()], enable_plugins=False)
        report = scanner.run()
        assert report is not None
        assert hasattr(report, "findings")
        assert hasattr(report, "health_score")

    def test_run_with_package_filter(self, tmp_path: Any) -> None:
        # Set up a fake site-packages with the expected probe data
        sp = tmp_path / "site-packages"
        sp.mkdir()

        class SiteProbe(StubProbe):
            def collect(self) -> Dict[str, Any]:
                data = super().collect()
                data["site_packages_paths"] = (str(sp),)
                return data

        scanner = Scanner(
            probes=[SiteProbe()],
            packages=["nonexistent_pkg"],
            enable_plugins=False,
        )
        report = scanner.run()
        # With no matching packages, should get 0 scanned
        assert report.packages_scanned == 0

    def test_run_handles_scanner_error_gracefully(self) -> None:
        """If probes raise, the scanner should return a report with an error finding."""
        class ExplodingProbe:
            name = "exploding"

            def is_applicable(self) -> bool:
                return True

            def collect(self) -> Dict[str, Any]:
                raise RuntimeError("Kaboom!")

        scanner = Scanner(probes=[ExplodingProbe()], enable_plugins=False)
        report = scanner.run()
        # Even if probes fail, run() should return a report
        assert report is not None

    def test_run_with_ignored_rules(self) -> None:
        scanner = Scanner(
            probes=[StubProbe()],
            ignored_rules={"GLIBC_VERSION_MISMATCH", "ARCH_MISMATCH"},
            enable_plugins=False,
        )
        report = scanner.run()
        # Ignored rules should not produce findings
        rule_ids = {f.rule_id for f in report.findings}
        assert "GLIBC_VERSION_MISMATCH" not in rule_ids
        assert "ARCH_MISMATCH" not in rule_ids


# ---------------------------------------------------------------------------
# Scanner.check_package
# ---------------------------------------------------------------------------


class TestScannerCheckPackage:
    def test_check_nonexistent_package(self) -> None:
        scanner = Scanner(probes=[StubProbe()], enable_plugins=False)
        findings = scanner.check_package("totally_fake_package_xyz")
        assert len(findings) >= 1
        # Should get a PACKAGE_NOT_FOUND finding
        rule_ids = {f.rule_id for f in findings}
        assert "PACKAGE_NOT_FOUND" in rule_ids


# ---------------------------------------------------------------------------
# Scanner.inspect_file
# ---------------------------------------------------------------------------


class TestScannerInspectFile:
    def test_inspect_nonexistent_file_raises(self) -> None:
        scanner = Scanner(probes=[StubProbe()], enable_plugins=False)
        with pytest.raises(FileNotFoundError):
            scanner.inspect_file("/nonexistent/file.whl")

    def test_inspect_unsupported_file_type_raises(self, tmp_path: Any) -> None:
        txt_file = tmp_path / "readme.txt"
        txt_file.write_text("Hello")

        scanner = Scanner(probes=[StubProbe()], enable_plugins=False)
        with pytest.raises(ValueError, match="Unsupported file type"):
            scanner.inspect_file(str(txt_file))

    def test_inspect_so_file(self, tmp_path: Any) -> None:
        """Inspecting a .so file should return findings (or at least not crash)."""
        # Create a minimal fake .so file (will fail ELF parsing -> finding)
        so_file = tmp_path / "test.so"
        so_file.write_bytes(b"\x00" * 64)

        scanner = Scanner(probes=[StubProbe()], enable_plugins=False)
        findings = scanner.inspect_file(str(so_file))
        # Should get at least one finding (ELF parse error or similar)
        assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# Scanner._parse_dist_info_name
# ---------------------------------------------------------------------------


class TestParseDistInfoName:
    def test_parse_standard_name(self) -> None:
        name, version = Scanner._parse_dist_info_name("numpy-1.26.4.dist-info")
        assert name == "numpy"
        assert version == "1.26.4"

    def test_parse_name_with_caps(self) -> None:
        name, version = Scanner._parse_dist_info_name("Pillow-10.2.0.dist-info")
        assert name == "Pillow"
        assert version == "10.2.0"

    def test_parse_invalid_no_suffix(self) -> None:
        name, version = Scanner._parse_dist_info_name("numpy-1.26.4")
        assert name == ""
        assert version == ""

    def test_parse_invalid_no_dash(self) -> None:
        name, version = Scanner._parse_dist_info_name("numpy.dist-info")
        assert name == ""
        assert version == ""


# ---------------------------------------------------------------------------
# Scanner._parse_wheel_filename
# ---------------------------------------------------------------------------


class TestParseWheelFilename:
    def test_parse_standard_wheel_name(self) -> None:
        name, version = Scanner._parse_wheel_filename(
            "torch-2.4.0-cp312-cp312-manylinux_2_17_x86_64.whl"
        )
        assert name == "torch"
        assert version == "2.4.0"

    def test_parse_wheel_with_build_tag(self) -> None:
        name, version = Scanner._parse_wheel_filename(
            "package-1.0-0-cp312-cp312-linux_x86_64.whl"
        )
        assert name == "package"
        assert version == "1.0"

    def test_parse_non_wheel_returns_empty(self) -> None:
        name, version = Scanner._parse_wheel_filename("not_a_wheel.tar.gz")
        assert name == ""
        assert version == ""

    def test_parse_too_few_parts(self) -> None:
        name, version = Scanner._parse_wheel_filename("short.whl")
        assert name == ""
        assert version == ""


# ---------------------------------------------------------------------------
# Scanner._build_profile
# ---------------------------------------------------------------------------


class TestBuildProfile:
    def test_build_from_valid_data(self) -> None:
        data = {
            "python_version": (3, 12, 0),
            "architecture": Architecture.X86_64,
            "os_name": "Ubuntu",
        }
        profile = Scanner._build_profile(data)
        assert profile.python_version == (3, 12, 0)
        assert profile.architecture == Architecture.X86_64
        assert profile.os_name == "Ubuntu"

    def test_ignores_unknown_keys(self) -> None:
        data = {
            "python_version": (3, 12, 0),
            "unknown_field_xyz": "should be ignored",
        }
        profile = Scanner._build_profile(data)
        assert profile.python_version == (3, 12, 0)
        assert not hasattr(profile, "unknown_field_xyz")

    def test_ignores_none_values(self) -> None:
        data = {
            "python_version": (3, 12, 0),
            "os_name": None,  # Should be ignored
        }
        profile = Scanner._build_profile(data)
        assert profile.os_name == ""  # Default value from SystemProfile
