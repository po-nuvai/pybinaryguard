"""Tests for all data models in pybinaryguard.models."""

from __future__ import annotations

import pytest

from pybinaryguard.models.enums import Architecture, ContainerRuntime, Severity
from pybinaryguard.models.finding import Finding, ScanReport
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo, WheelTag
from pybinaryguard.models.system import SystemProfile


# ---------------------------------------------------------------------------
# Severity ordering
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_severity_lt_critical_less_than_warning(self) -> None:
        assert Severity.CRITICAL < Severity.WARNING

    def test_severity_lt_warning_less_than_info(self) -> None:
        assert Severity.WARNING < Severity.INFO

    def test_severity_lt_info_less_than_passed(self) -> None:
        assert Severity.INFO < Severity.PASSED

    def test_severity_lt_not_less_than_itself(self) -> None:
        assert not (Severity.CRITICAL < Severity.CRITICAL)

    def test_severity_le_equal(self) -> None:
        assert Severity.CRITICAL <= Severity.CRITICAL

    def test_severity_le_less(self) -> None:
        assert Severity.CRITICAL <= Severity.WARNING

    def test_severity_le_not_greater(self) -> None:
        assert not (Severity.PASSED <= Severity.CRITICAL)

    def test_severity_lt_returns_not_implemented_for_non_severity(self) -> None:
        result = Severity.CRITICAL.__lt__("not_a_severity")
        assert result is NotImplemented

    def test_severity_le_returns_not_implemented_for_non_severity(self) -> None:
        result = Severity.CRITICAL.__le__("not_a_severity")
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_from_machine_x86_64(self) -> None:
        assert Architecture.from_machine("x86_64") == Architecture.X86_64

    def test_from_machine_amd64(self) -> None:
        assert Architecture.from_machine("AMD64") == Architecture.X86_64

    def test_from_machine_aarch64(self) -> None:
        assert Architecture.from_machine("aarch64") == Architecture.AARCH64

    def test_from_machine_arm64(self) -> None:
        assert Architecture.from_machine("arm64") == Architecture.AARCH64

    def test_from_machine_armv7l(self) -> None:
        assert Architecture.from_machine("armv7l") == Architecture.ARMV7L

    def test_from_machine_armv6l(self) -> None:
        assert Architecture.from_machine("armv6l") == Architecture.ARMV7L

    def test_from_machine_i686(self) -> None:
        assert Architecture.from_machine("i686") == Architecture.I686

    def test_from_machine_i386(self) -> None:
        assert Architecture.from_machine("i386") == Architecture.I686

    def test_from_machine_ppc64le(self) -> None:
        assert Architecture.from_machine("ppc64le") == Architecture.PPC64LE

    def test_from_machine_s390x(self) -> None:
        assert Architecture.from_machine("s390x") == Architecture.S390X

    def test_from_machine_unknown(self) -> None:
        assert Architecture.from_machine("riscv64") == Architecture.UNKNOWN

    def test_elf_machine_x86_64(self) -> None:
        assert Architecture.X86_64.elf_machine == 62

    def test_elf_machine_aarch64(self) -> None:
        assert Architecture.AARCH64.elf_machine == 183

    def test_elf_machine_arm(self) -> None:
        assert Architecture.ARMV7L.elf_machine == 40

    def test_elf_machine_i686(self) -> None:
        assert Architecture.I686.elf_machine == 3

    def test_elf_machine_unknown_returns_zero(self) -> None:
        assert Architecture.UNKNOWN.elf_machine == 0


# ---------------------------------------------------------------------------
# SystemProfile.summary()
# ---------------------------------------------------------------------------


class TestSystemProfileSummary:
    def test_summary_full_profile(self, sample_profile: SystemProfile) -> None:
        s = sample_profile.summary()
        assert "Python" in s
        assert "3.12.0" in s["Python"]
        assert "OS" in s
        assert "Ubuntu" in s["OS"]
        assert "Architecture" in s
        assert s["Architecture"] == "x86_64"
        assert "GLIBC" in s
        assert s["GLIBC"] == "2.35"
        assert "CUDA Runtime" in s
        assert s["CUDA Runtime"] == "12.2"
        assert "GPU Driver" in s
        assert "GPU" in s
        assert "compute 8.9" in s["GPU"]

    def test_summary_musl_profile(self, alpine_profile: SystemProfile) -> None:
        s = alpine_profile.summary()
        assert "musl" in s
        assert s["musl"] == "1.2"
        assert "GLIBC" not in s
        assert "Container" in s
        assert s["Container"] == "docker"

    def test_summary_jetson_profile(self, jetson_profile: SystemProfile) -> None:
        s = jetson_profile.summary()
        assert "Board" in s
        assert "Jetson" in s["Board"]
        assert "JetPack" in s
        assert s["JetPack"] == "6.0"

    def test_summary_no_gpu_profile(self, no_gpu_profile: SystemProfile) -> None:
        s = no_gpu_profile.summary()
        assert "GPU" not in s
        assert "CUDA Runtime" not in s
        assert "GPU Driver" not in s

    def test_summary_empty_profile(self) -> None:
        profile = SystemProfile()
        s = profile.summary()
        # No fields should be populated
        assert len(s) == 0


# ---------------------------------------------------------------------------
# PackageBinaryInfo properties
# ---------------------------------------------------------------------------


class TestPackageBinaryInfo:
    def test_has_binaries_true(self, sample_package: PackageBinaryInfo) -> None:
        assert sample_package.has_binaries is True

    def test_has_binaries_false(self, pure_python_package: PackageBinaryInfo) -> None:
        assert pure_python_package.has_binaries is False

    def test_so_count(self, sample_package: PackageBinaryInfo) -> None:
        assert sample_package.so_count == 2

    def test_so_count_zero(self, pure_python_package: PackageBinaryInfo) -> None:
        assert pure_python_package.so_count == 0

    def test_manylinux_tag(self, sample_package: PackageBinaryInfo) -> None:
        assert sample_package.manylinux_tag == "manylinux_2_17_x86_64"

    def test_manylinux_tag_none(self, pure_python_package: PackageBinaryInfo) -> None:
        assert pure_python_package.manylinux_tag is None

    def test_manylinux_glibc_standard(self, sample_package: PackageBinaryInfo) -> None:
        assert sample_package.manylinux_glibc == (2, 17)

    def test_manylinux_glibc_none(self, pure_python_package: PackageBinaryInfo) -> None:
        assert pure_python_package.manylinux_glibc is None

    def test_manylinux_glibc_legacy_manylinux1(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="old",
            package_version="1.0",
            install_path="/tmp/old",
            wheel_tags=[WheelTag("cp39", "cp39", "manylinux1_x86_64")],
        )
        assert pkg.manylinux_glibc == (2, 5)

    def test_manylinux_glibc_legacy_manylinux2014(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="mid",
            package_version="1.0",
            install_path="/tmp/mid",
            wheel_tags=[WheelTag("cp39", "cp39", "manylinux2014_x86_64")],
        )
        assert pkg.manylinux_glibc == (2, 17)


# ---------------------------------------------------------------------------
# Finding.as_dict()
# ---------------------------------------------------------------------------


class TestFinding:
    def test_as_dict_minimal(self) -> None:
        f = Finding(
            rule_id="TEST_RULE",
            severity=Severity.WARNING,
            title="Test Title",
            explanation="Test explanation",
        )
        d = f.as_dict()
        assert d["rule_id"] == "TEST_RULE"
        assert d["severity"] == "warning"
        assert d["title"] == "Test Title"
        assert d["explanation"] == "Test explanation"
        assert "technical_detail" not in d
        assert "suggestion" not in d
        assert "package" not in d

    def test_as_dict_with_all_fields(self) -> None:
        f = Finding(
            rule_id="FULL_RULE",
            severity=Severity.CRITICAL,
            title="Full Test",
            explanation="Full explanation",
            technical_detail="detail here",
            suggestion="fix it",
            package="numpy",
            package_version="1.26.0",
            confidence=0.85,
            related_error="some error",
        )
        d = f.as_dict()
        assert d["technical_detail"] == "detail here"
        assert d["suggestion"] == "fix it"
        assert d["package"] == "numpy"
        assert d["package_version"] == "1.26.0"
        assert d["confidence"] == 0.85
        assert d["related_error"] == "some error"

    def test_as_dict_confidence_1_not_included(self) -> None:
        f = Finding(
            rule_id="X",
            severity=Severity.INFO,
            title="T",
            explanation="E",
            confidence=1.0,
        )
        d = f.as_dict()
        assert "confidence" not in d


# ---------------------------------------------------------------------------
# ScanReport health_score and health_label
# ---------------------------------------------------------------------------


class TestScanReport:
    def test_health_score_100_no_findings(self) -> None:
        report = ScanReport(findings=[], packages_scanned=10, total_packages=10)
        assert report.health_score == 100

    def test_health_score_75_one_critical(self) -> None:
        findings = [
            Finding(rule_id="A", severity=Severity.CRITICAL, title="X", explanation="Y"),
        ]
        report = ScanReport(findings=findings)
        assert report.health_score == 75

    def test_health_score_50_two_critical(self) -> None:
        findings = [
            Finding(rule_id="A", severity=Severity.CRITICAL, title="X", explanation="Y"),
            Finding(rule_id="B", severity=Severity.CRITICAL, title="X", explanation="Y"),
        ]
        report = ScanReport(findings=findings)
        assert report.health_score == 50

    def test_health_score_0_clamped(self) -> None:
        findings = [
            Finding(rule_id=f"R{i}", severity=Severity.CRITICAL, title="X", explanation="Y")
            for i in range(10)
        ]
        report = ScanReport(findings=findings)
        assert report.health_score == 0

    def test_health_score_with_warnings(self) -> None:
        findings = [
            Finding(rule_id="W1", severity=Severity.WARNING, title="X", explanation="Y"),
            Finding(rule_id="W2", severity=Severity.WARNING, title="X", explanation="Y"),
        ]
        report = ScanReport(findings=findings)
        assert report.health_score == 90

    def test_health_label_excellent(self) -> None:
        report = ScanReport(findings=[])
        assert report.health_label == "Excellent"

    def test_health_label_good(self) -> None:
        findings = [
            Finding(rule_id="A", severity=Severity.CRITICAL, title="X", explanation="Y"),
        ]
        report = ScanReport(findings=findings)
        assert report.health_label == "Good"

    def test_health_label_needs_attention(self) -> None:
        findings = [
            Finding(rule_id="A", severity=Severity.CRITICAL, title="X", explanation="Y"),
            Finding(rule_id="B", severity=Severity.CRITICAL, title="X", explanation="Y"),
        ]
        report = ScanReport(findings=findings)
        assert report.health_label == "Needs Attention"

    def test_health_label_critical(self) -> None:
        findings = [
            Finding(rule_id=f"R{i}", severity=Severity.CRITICAL, title="X", explanation="Y")
            for i in range(5)
        ]
        report = ScanReport(findings=findings)
        assert report.health_label == "Critical"

    def test_summary_dict(self) -> None:
        findings = [
            Finding(rule_id="C", severity=Severity.CRITICAL, title="X", explanation="Y"),
            Finding(rule_id="W", severity=Severity.WARNING, title="X", explanation="Y"),
            Finding(rule_id="I", severity=Severity.INFO, title="X", explanation="Y"),
            Finding(rule_id="P", severity=Severity.PASSED, title="X", explanation="Y"),
        ]
        report = ScanReport(findings=findings, packages_scanned=4, total_packages=10)
        d = report.summary_dict()
        assert d["critical"] == 1
        assert d["warning"] == 1
        assert d["info"] == 1
        assert d["passed"] == 1
        assert d["total_packages"] == 10
        assert d["packages_scanned"] == 4
