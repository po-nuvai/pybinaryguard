"""Tests for gap-closing features: toolchain probe, venv probe, source build rules,
dependency conflict rules, venv rules, and import validator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, mock_open
import os
import sys

import pytest

from pybinaryguard.models.enums import Architecture, ContainerRuntime, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo
from pybinaryguard.models.system import SystemProfile


# ---------------------------------------------------------------------------
# Toolchain Probe
# ---------------------------------------------------------------------------

class TestToolchainProbe:
    def test_collect_returns_dict(self):
        from pybinaryguard.probes.toolchain_probe import ToolchainProbe
        probe = ToolchainProbe()
        data = probe.collect()
        assert isinstance(data, dict)
        assert "toolchain_versions" in data
        assert "has_build_tools" in data
        assert "has_python_dev_headers" in data

    def test_name(self):
        from pybinaryguard.probes.toolchain_probe import ToolchainProbe
        assert ToolchainProbe().name == "toolchain"

    @patch("pybinaryguard.probes.toolchain_probe.shutil.which")
    @patch("pybinaryguard.probes.toolchain_probe.subprocess.run")
    def test_detect_gcc(self, mock_run, mock_which):
        from pybinaryguard.probes.toolchain_probe import ToolchainProbe
        mock_which.return_value = "/usr/bin/gcc"
        mock_run.return_value = MagicMock(
            stdout="gcc (Ubuntu 11.4.0-1ubuntu1) 11.4.0", stderr=""
        )
        version = ToolchainProbe._detect_tool("gcc", r"(\d+\.\d+\.\d+)")
        assert version == "11.4.0"

    @patch("pybinaryguard.probes.toolchain_probe.shutil.which")
    def test_detect_missing_tool(self, mock_which):
        from pybinaryguard.probes.toolchain_probe import ToolchainProbe
        mock_which.return_value = None
        version = ToolchainProbe._detect_tool("gcc", r"(\d+\.\d+\.\d+)")
        assert version is None

    def test_has_python_headers_method(self):
        from pybinaryguard.probes.toolchain_probe import ToolchainProbe
        # Just check it runs without error
        result = ToolchainProbe._has_python_headers()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Venv Probe
# ---------------------------------------------------------------------------

class TestVenvProbe:
    def test_collect_returns_dict(self):
        from pybinaryguard.probes.venv_probe import VenvProbe
        probe = VenvProbe()
        data = probe.collect()
        assert isinstance(data, dict)
        assert "venv_type" in data
        assert "is_system_python" in data
        assert "is_virtual_env" in data

    def test_name(self):
        from pybinaryguard.probes.venv_probe import VenvProbe
        assert VenvProbe().name == "venv"

    @patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "base"})
    def test_detect_conda(self):
        from pybinaryguard.probes.venv_probe import VenvProbe
        assert VenvProbe._detect_venv_type() == "conda"

    @patch.dict(os.environ, {"POETRY_ACTIVE": "1"}, clear=False)
    def test_detect_poetry(self):
        from pybinaryguard.probes.venv_probe import VenvProbe
        # Clear conda env vars first
        env = {k: v for k, v in os.environ.items()
               if k not in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX")}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"POETRY_ACTIVE": "1"}):
                assert VenvProbe._detect_venv_type() == "poetry"

    @patch.dict(os.environ, {"PIPENV_ACTIVE": "1"}, clear=False)
    def test_detect_pipenv(self):
        from pybinaryguard.probes.venv_probe import VenvProbe
        env = {k: v for k, v in os.environ.items()
               if k not in ("CONDA_DEFAULT_ENV", "CONDA_PREFIX", "POETRY_ACTIVE")}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"PIPENV_ACTIVE": "1"}):
                assert VenvProbe._detect_venv_type() == "pipenv"

    def test_mixed_env_detection(self):
        from pybinaryguard.probes.venv_probe import VenvProbe
        result = VenvProbe._detect_mixed_env()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SystemProfile new fields
# ---------------------------------------------------------------------------

class TestSystemProfileNewFields:
    def test_toolchain_fields(self):
        p = SystemProfile(
            toolchain_versions={"gcc": "11.4.0"},
            has_build_tools=True,
            has_python_dev_headers=True,
        )
        assert p.toolchain_versions["gcc"] == "11.4.0"
        assert p.has_build_tools is True

    def test_venv_fields(self):
        p = SystemProfile(
            venv_type="venv",
            is_system_python=False,
            is_virtual_env=True,
            mixed_env_risk=False,
        )
        assert p.venv_type == "venv"
        assert p.is_virtual_env is True

    def test_summary_includes_compiler(self):
        p = SystemProfile(
            python_version=(3, 11, 0),
            toolchain_versions={"gcc": "11.4.0"},
        )
        summary = p.summary()
        assert "Compiler" in summary
        assert "gcc 11.4.0" in summary["Compiler"]

    def test_summary_includes_venv(self):
        p = SystemProfile(
            python_version=(3, 11, 0),
            venv_type="conda",
        )
        summary = p.summary()
        assert "Environment" in summary
        assert summary["Environment"] == "conda"


# ---------------------------------------------------------------------------
# Source Build Rules
# ---------------------------------------------------------------------------

def _make_profile(**kwargs):
    defaults = dict(
        python_version=(3, 11, 0),
        architecture=Architecture.X86_64,
        glibc_version=(2, 35),
        has_build_tools=True,
        has_python_dev_headers=True,
    )
    defaults.update(kwargs)
    return SystemProfile(**defaults)


def _make_pkg(name, version="1.0.0", pure=False, has_so=False, has_wheel_tags=True):
    so_list = []
    if has_so:
        so_list = [SharedObjectInfo(path=f"/lib/{name}.so", filename=f"{name}.so")]
    tags = []
    if has_wheel_tags:
        from pybinaryguard.models.package import WheelTag
        tags = [WheelTag(interpreter="cp311", abi="cp311", platform="manylinux_2_17_x86_64")]
    return PackageBinaryInfo(
        package_name=name,
        package_version=version,
        install_path=f"/fake/{name}-{version}.dist-info",
        is_pure_python=pure,
        shared_objects=so_list,
        wheel_tags=tags if has_wheel_tags else [],
    )


class TestSourceBuildRules:
    def test_detect_source_built_package(self):
        from pybinaryguard.rules.builtin.source_build_rules import SourceBuildDetectionRule
        rule = SourceBuildDetectionRule()
        profile = _make_profile()
        pkg = _make_pkg("custom_lib", has_so=True, has_wheel_tags=False, pure=False)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].rule_id == "SOURCE_BUILD_DETECTED"

    def test_skip_pure_python(self):
        from pybinaryguard.rules.builtin.source_build_rules import SourceBuildDetectionRule
        rule = SourceBuildDetectionRule()
        profile = _make_profile()
        pkg = _make_pkg("pure_pkg", pure=True)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_skip_wheel_installed(self):
        from pybinaryguard.rules.builtin.source_build_rules import SourceBuildDetectionRule
        rule = SourceBuildDetectionRule()
        profile = _make_profile()
        pkg = _make_pkg("torch", has_so=True, has_wheel_tags=True)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 0

    def test_no_compiler_warning(self):
        from pybinaryguard.rules.builtin.source_build_rules import SourceBuildNoCompilerRule
        rule = SourceBuildNoCompilerRule()
        profile = _make_profile(has_build_tools=False)
        assert rule.is_applicable(profile)
        pkg = _make_pkg("custom_lib", has_so=True, has_wheel_tags=False, pure=False)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].rule_id == "SOURCE_BUILD_NO_COMPILER"

    def test_no_compiler_not_applicable(self):
        from pybinaryguard.rules.builtin.source_build_rules import SourceBuildNoCompilerRule
        rule = SourceBuildNoCompilerRule()
        profile = _make_profile(has_build_tools=True)
        assert not rule.is_applicable(profile)

    def test_missing_python_headers(self):
        from pybinaryguard.rules.builtin.source_build_rules import MissingPythonHeadersRule
        rule = MissingPythonHeadersRule()
        profile = _make_profile(has_python_dev_headers=False, has_build_tools=True)
        assert rule.is_applicable(profile)
        pkg = _make_pkg("custom_lib", has_so=True, has_wheel_tags=False, pure=False)
        findings = rule.evaluate(profile, [pkg])
        assert len(findings) == 1
        assert findings[0].rule_id == "SOURCE_BUILD_NO_PYTHON_HEADERS"


# ---------------------------------------------------------------------------
# Dependency Conflict Rules
# ---------------------------------------------------------------------------

class TestDependencyConflictRules:
    def test_version_matches_equal(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("1.2.3", "==1.2.3")
        assert not _version_matches("1.2.4", "==1.2.3")

    def test_version_matches_gte(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("2.0.0", ">=1.0.0")
        assert _version_matches("1.0.0", ">=1.0.0")
        assert not _version_matches("0.9.0", ">=1.0.0")

    def test_version_matches_lt(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("1.0.0", "<2.0.0")
        assert not _version_matches("2.0.0", "<2.0.0")

    def test_version_matches_compound(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("1.5.0", ">=1.0.0,<2.0.0")
        assert not _version_matches("2.1.0", ">=1.0.0,<2.0.0")

    def test_version_matches_not_equal(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("1.2.4", "!=1.2.3")
        assert not _version_matches("1.2.3", "!=1.2.3")

    def test_version_matches_compatible(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("1.4.5", "~=1.4.2")
        assert not _version_matches("1.5.0", "~=1.4.2")

    def test_version_matches_wildcard(self):
        from pybinaryguard.rules.builtin.dependency_rules import _version_matches
        assert _version_matches("1.0.5", "==1.0.*")
        assert not _version_matches("1.1.0", "==1.0.*")

    def test_parse_requirement(self):
        from pybinaryguard.rules.builtin.dependency_rules import _parse_requirement
        result = _parse_requirement("numpy>=1.20.0")
        assert result is not None
        name, spec, marker = result
        assert name == "numpy"
        assert ">=1.20.0" in spec

    def test_parse_requirement_with_extras(self):
        from pybinaryguard.rules.builtin.dependency_rules import _parse_requirement
        result = _parse_requirement("torch[cuda12]>=2.0")
        assert result is not None
        name, spec, _ = result
        assert name == "torch"

    def test_parse_requirement_skips_extra_markers(self):
        from pybinaryguard.rules.builtin.dependency_rules import _parse_requirement
        result = _parse_requirement('sphinx; extra == "docs"')
        assert result is None

    def test_conflict_detection(self):
        from pybinaryguard.rules.builtin.dependency_rules import DependencyConflictRule
        import tempfile

        rule = DependencyConflictRule()
        profile = _make_profile()

        # Create fake dist-info with a requirement
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_info = os.path.join(tmpdir, "pkgA-1.0.0.dist-info")
            os.makedirs(dist_info)
            with open(os.path.join(dist_info, "METADATA"), "w") as f:
                f.write("Metadata-Version: 2.1\n")
                f.write("Name: pkgA\n")
                f.write("Version: 1.0.0\n")
                f.write("Requires-Dist: numpy>=2.0.0\n")
                f.write("\n")

            pkgA = PackageBinaryInfo(
                package_name="pkgA", package_version="1.0.0",
                install_path=dist_info, is_pure_python=True,
            )
            numpy = PackageBinaryInfo(
                package_name="numpy", package_version="1.24.0",
                install_path="/fake/numpy-1.24.0.dist-info",
                is_pure_python=False,
            )

            findings = rule.evaluate(profile, [pkgA, numpy])
            assert len(findings) == 1
            assert findings[0].rule_id == "DEPENDENCY_VERSION_CONFLICT"
            assert "numpy" in findings[0].title

    def test_no_conflict(self):
        from pybinaryguard.rules.builtin.dependency_rules import DependencyConflictRule
        import tempfile

        rule = DependencyConflictRule()
        profile = _make_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            dist_info = os.path.join(tmpdir, "pkgA-1.0.0.dist-info")
            os.makedirs(dist_info)
            with open(os.path.join(dist_info, "METADATA"), "w") as f:
                f.write("Metadata-Version: 2.1\n")
                f.write("Name: pkgA\n")
                f.write("Requires-Dist: numpy>=1.20.0\n")
                f.write("\n")

            pkgA = PackageBinaryInfo(
                package_name="pkgA", package_version="1.0.0",
                install_path=dist_info,
            )
            numpy = PackageBinaryInfo(
                package_name="numpy", package_version="1.24.0",
                install_path="/fake/numpy-1.24.0.dist-info",
            )

            findings = rule.evaluate(profile, [pkgA, numpy])
            assert len(findings) == 0

    def test_missing_dependency(self):
        from pybinaryguard.rules.builtin.dependency_rules import MissingDependencyRule
        import tempfile

        rule = MissingDependencyRule()
        profile = _make_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            dist_info = os.path.join(tmpdir, "pkgA-1.0.0.dist-info")
            os.makedirs(dist_info)
            with open(os.path.join(dist_info, "METADATA"), "w") as f:
                f.write("Metadata-Version: 2.1\n")
                f.write("Requires-Dist: nonexistent-pkg\n")
                f.write("\n")

            pkgA = PackageBinaryInfo(
                package_name="pkgA", package_version="1.0.0",
                install_path=dist_info,
            )

            findings = rule.evaluate(profile, [pkgA])
            assert len(findings) == 1
            assert findings[0].rule_id == "DEPENDENCY_MISSING"


# ---------------------------------------------------------------------------
# Venv Rules
# ---------------------------------------------------------------------------

class TestVenvRules:
    def test_system_python_warning(self):
        from pybinaryguard.rules.builtin.venv_rules import SystemPythonWarningRule
        rule = SystemPythonWarningRule()
        profile = _make_profile(is_system_python=True)
        assert rule.is_applicable(profile)
        pkgs = [_make_pkg(f"pkg{i}") for i in range(15)]
        findings = rule.evaluate(profile, pkgs)
        assert len(findings) == 1
        assert findings[0].rule_id == "VENV_SYSTEM_PYTHON"

    def test_system_python_few_packages(self):
        from pybinaryguard.rules.builtin.venv_rules import SystemPythonWarningRule
        rule = SystemPythonWarningRule()
        profile = _make_profile(is_system_python=True)
        pkgs = [_make_pkg(f"pkg{i}") for i in range(5)]
        findings = rule.evaluate(profile, pkgs)
        assert len(findings) == 0

    def test_mixed_env(self):
        from pybinaryguard.rules.builtin.venv_rules import MixedEnvironmentRule
        rule = MixedEnvironmentRule()
        profile = _make_profile(mixed_env_risk=True)
        assert rule.is_applicable(profile)
        findings = rule.evaluate(profile, [])
        assert len(findings) == 1
        assert findings[0].rule_id == "VENV_MIXED_ENVIRONMENT"

    def test_user_site_leak(self):
        from pybinaryguard.rules.builtin.venv_rules import UserSiteLeakRule
        rule = UserSiteLeakRule()
        profile = _make_profile(
            is_virtual_env=True, pip_user_site_enabled=True, venv_type="venv"
        )
        assert rule.is_applicable(profile)
        findings = rule.evaluate(profile, [])
        assert len(findings) == 1
        assert findings[0].rule_id == "VENV_USER_SITE_LEAK"

    def test_user_site_not_applicable(self):
        from pybinaryguard.rules.builtin.venv_rules import UserSiteLeakRule
        rule = UserSiteLeakRule()
        profile = _make_profile(is_virtual_env=False, pip_user_site_enabled=False)
        assert not rule.is_applicable(profile)

    def test_conda_pip_mixing(self):
        from pybinaryguard.rules.builtin.venv_rules import CondaPipMixingRule
        rule = CondaPipMixingRule()
        profile = _make_profile(venv_type="conda")
        assert rule.is_applicable(profile)
        pkgs = [_make_pkg(f"pkg{i}", has_so=True) for i in range(10)]
        findings = rule.evaluate(profile, pkgs)
        assert len(findings) == 1
        assert findings[0].rule_id == "VENV_CONDA_PIP_MIXING"

    def test_conda_not_applicable(self):
        from pybinaryguard.rules.builtin.venv_rules import CondaPipMixingRule
        rule = CondaPipMixingRule()
        profile = _make_profile(venv_type="venv")
        assert not rule.is_applicable(profile)


# ---------------------------------------------------------------------------
# Import Validator
# ---------------------------------------------------------------------------

class TestImportValidator:
    def test_init(self):
        from pybinaryguard.validators.import_validator import ImportValidator
        v = ImportValidator(timeout=5.0)
        assert v._timeout == 5.0

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_successful_import(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(returncode=0, stdout="OK\n", stderr="")
        v = ImportValidator()
        result = v.test_import("json")
        assert result.success is True
        assert result.package_name == "json"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_import_error(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(
            returncode=10, stdout="", stderr="ImportError: cannot open shared object"
        )
        v = ImportValidator()
        result = v.test_import("fake_pkg")
        assert result.success is False
        assert result.error_type == "ImportError"
        assert result.category == "missing_shared_library"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_glibc_mismatch(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(
            returncode=10, stdout="",
            stderr="ImportError: /lib/libm.so.6: version `GLIBC_2.34' not found"
        )
        v = ImportValidator()
        result = v.test_import("torch")
        assert result.success is False
        assert result.category == "glibc_mismatch"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_illegal_instruction(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(returncode=132, stdout="", stderr="")
        v = ImportValidator()
        result = v.test_import("torch")
        assert result.success is False
        assert result.category == "illegal_instruction"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_segfault(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(returncode=139, stdout="", stderr="")
        v = ImportValidator()
        result = v.test_import("broken_pkg")
        assert result.success is False
        assert result.category == "segfault"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_timeout(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="python", timeout=5)
        v = ImportValidator(timeout=5.0)
        result = v.test_import("slow_pkg")
        assert result.success is False
        assert result.error_type == "Timeout"
        assert result.category == "timeout"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_cuda_missing(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(
            returncode=10, stdout="",
            stderr="ImportError: libcudart.so.12: cannot open shared object file"
        )
        v = ImportValidator()
        result = v.test_import("torch")
        assert result.success is False
        assert result.category == "cuda_missing"

    def test_result_as_dict(self):
        from pybinaryguard.validators.import_validator import ImportTestResult
        r = ImportTestResult(
            package_name="torch", top_level_name="torch",
            success=False, error_type="ImportError",
            error_message="GLIBC not found", category="glibc_mismatch",
            exit_code=10, duration_ms=50.0,
        )
        d = r.as_dict()
        assert d["package"] == "torch"
        assert d["success"] is False
        assert d["category"] == "glibc_mismatch"

    def test_get_top_level_name(self):
        from pybinaryguard.validators.import_validator import ImportValidator
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "top_level.txt"), "w") as f:
                f.write("torch\n")
            result = ImportValidator.get_top_level_name(tmpdir, "pytorch")
            assert result == "torch"

    def test_get_top_level_name_fallback(self):
        from pybinaryguard.validators.import_validator import ImportValidator
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ImportValidator.get_top_level_name(tmpdir, "my-package")
            assert result == "my_package"

    @patch("pybinaryguard.validators.import_validator.subprocess.run")
    def test_test_packages(self, mock_run):
        from pybinaryguard.validators.import_validator import ImportValidator
        mock_run.return_value = MagicMock(returncode=0, stdout="OK\n", stderr="")
        v = ImportValidator()
        results = v.test_packages([("json", None), ("os", None)])
        assert len(results) == 2
        assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# Probe Registration
# ---------------------------------------------------------------------------

class TestProbeRegistration:
    def test_all_probes_includes_new(self):
        from pybinaryguard.probes import get_all_probes
        probes = get_all_probes()
        names = [p.name for p in probes]
        assert "toolchain" in names
        assert "venv" in names
        assert len(probes) == 9  # 7 original + 2 new

    def test_all_rules_includes_new(self):
        from pybinaryguard.rules.builtin import get_all_builtin_rules
        rules = get_all_builtin_rules()
        rule_ids = [r.rule_id for r in rules]
        # Source build rules
        assert "SOURCE_BUILD_DETECTED" in rule_ids
        assert "SOURCE_BUILD_NO_COMPILER" in rule_ids
        assert "SOURCE_BUILD_NO_PYTHON_HEADERS" in rule_ids
        # Dependency rules
        assert "DEPENDENCY_VERSION_CONFLICT" in rule_ids
        assert "DEPENDENCY_MISSING" in rule_ids
        # Venv rules
        assert "VENV_SYSTEM_PYTHON" in rule_ids
        assert "VENV_MIXED_ENVIRONMENT" in rule_ids
        assert "VENV_USER_SITE_LEAK" in rule_ids
        assert "VENV_CONDA_PIP_MIXING" in rule_ids
