"""Tests for Fast vs Deep Scan Modes."""

from __future__ import annotations

import argparse
from unittest import mock

from pybinaryguard.models.enums import ScanMode, Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo


# ---------------------------------------------------------------------------
# ScanMode enum
# ---------------------------------------------------------------------------

class TestScanModeEnum:
    def test_fast_value(self):
        assert ScanMode.FAST.value == "fast"

    def test_standard_value(self):
        assert ScanMode.STANDARD.value == "standard"

    def test_deep_value(self):
        assert ScanMode.DEEP.value == "deep"

    def test_default_is_standard(self):
        from pybinaryguard.scanner import Scanner
        scanner = Scanner()
        assert scanner._scan_mode == ScanMode.STANDARD


# ---------------------------------------------------------------------------
# Scanner scan mode initialization
# ---------------------------------------------------------------------------

class TestScannerModeInit:
    def test_scanner_accepts_fast_mode(self):
        from pybinaryguard.scanner import Scanner
        scanner = Scanner(scan_mode=ScanMode.FAST)
        assert scanner._scan_mode == ScanMode.FAST

    def test_scanner_accepts_deep_mode(self):
        from pybinaryguard.scanner import Scanner
        scanner = Scanner(scan_mode=ScanMode.DEEP)
        assert scanner._scan_mode == ScanMode.DEEP


# ---------------------------------------------------------------------------
# FAST mode behaviour
# ---------------------------------------------------------------------------

class TestFastMode:
    def test_fast_mode_skips_elf_analyzer_import(self):
        """In FAST mode, ELFAnalyzer should not be imported during package discovery."""
        from pybinaryguard.scanner import Scanner
        from pybinaryguard.models.system import SystemProfile

        scanner = Scanner(scan_mode=ScanMode.FAST)
        # Use a non-existent path so no packages are found
        profile = SystemProfile(site_packages_paths=("/nonexistent/path",))
        scanner._profile = profile

        # Should not raise even if ELF analyzer is broken
        packages = scanner._discover_and_analyze_packages(profile)
        assert packages == []

    def test_fast_mode_skips_elf_dependent_rules(self):
        """FAST mode should skip rules that depend on ELF analysis."""
        from pybinaryguard.scanner import Scanner

        scanner = Scanner(scan_mode=ScanMode.FAST)
        assert "GLIBC_VERSION_MISMATCH" in scanner._ELF_DEPENDENT_RULES
        assert "MISSING_SHARED_LIB" in scanner._ELF_DEPENDENT_RULES

    def test_has_shared_objects_detects_so_files(self):
        """_has_shared_objects should detect .so files in a directory."""
        import tempfile
        import os
        from pybinaryguard.scanner import Scanner

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake .so file
            so_path = os.path.join(tmpdir, "test.cpython-312-x86_64-linux-gnu.so")
            with open(so_path, "w") as f:
                f.write("fake")

            assert Scanner._has_shared_objects(tmpdir) is True

    def test_has_shared_objects_negative(self):
        """_has_shared_objects returns False for pure-Python dirs."""
        import tempfile
        import os
        from pybinaryguard.scanner import Scanner

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only .py files
            py_path = os.path.join(tmpdir, "module.py")
            with open(py_path, "w") as f:
                f.write("pass")

            assert Scanner._has_shared_objects(tmpdir) is False

    def test_has_shared_objects_nonexistent_dir(self):
        from pybinaryguard.scanner import Scanner
        assert Scanner._has_shared_objects("/nonexistent/path") is False


# ---------------------------------------------------------------------------
# DEEP mode behaviour
# ---------------------------------------------------------------------------

class TestDeepMode:
    def test_compute_binary_hashes(self):
        """DEEP mode should populate SHA256 hashes on shared objects."""
        import tempfile
        import hashlib
        import os
        from pybinaryguard.scanner import Scanner

        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = os.path.join(tmpdir, "test.so")
            content = b"fake binary content"
            with open(so_path, "wb") as f:
                f.write(content)

            expected_hash = hashlib.sha256(content).hexdigest()

            so_info = SharedObjectInfo(path=so_path, filename="test.so")
            pkg = PackageBinaryInfo(
                package_name="test",
                package_version="1.0",
                install_path=tmpdir,
                shared_objects=[so_info],
            )

            Scanner._compute_binary_hashes(pkg)

            assert so_info.sha256 == expected_hash

    def test_compute_binary_hashes_missing_file(self):
        """Hash computation should handle missing files gracefully."""
        from pybinaryguard.scanner import Scanner

        so_info = SharedObjectInfo(path="/nonexistent/test.so", filename="test.so")
        pkg = PackageBinaryInfo(
            package_name="test",
            package_version="1.0",
            install_path="/nonexistent",
            shared_objects=[so_info],
        )

        Scanner._compute_binary_hashes(pkg)
        assert so_info.sha256 is None

    def test_compute_binary_hashes_no_path(self):
        """Hash computation should handle shared objects without paths."""
        from pybinaryguard.scanner import Scanner

        so_info = SharedObjectInfo(path="", filename="test.so")
        pkg = PackageBinaryInfo(
            package_name="test",
            package_version="1.0",
            install_path="/tmp",
            shared_objects=[so_info],
        )

        Scanner._compute_binary_hashes(pkg)
        assert so_info.sha256 is None


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIScanMode:
    def test_resolve_scan_mode_default(self):
        from pybinaryguard.cli.commands import _resolve_scan_mode
        args = argparse.Namespace(fast=False, deep=False)
        assert _resolve_scan_mode(args) == ScanMode.STANDARD

    def test_resolve_scan_mode_fast(self):
        from pybinaryguard.cli.commands import _resolve_scan_mode
        args = argparse.Namespace(fast=True, deep=False)
        assert _resolve_scan_mode(args) == ScanMode.FAST

    def test_resolve_scan_mode_deep(self):
        from pybinaryguard.cli.commands import _resolve_scan_mode
        args = argparse.Namespace(fast=False, deep=True)
        assert _resolve_scan_mode(args) == ScanMode.DEEP

    def test_resolve_scan_mode_missing_attrs(self):
        """Should handle args without fast/deep attrs."""
        from pybinaryguard.cli.commands import _resolve_scan_mode
        args = argparse.Namespace()
        assert _resolve_scan_mode(args) == ScanMode.STANDARD

    def test_cli_parser_fast_flag(self):
        from pybinaryguard.cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["scan", "--fast"])
        assert args.fast is True
        assert args.deep is False

    def test_cli_parser_deep_flag(self):
        from pybinaryguard.cli.main import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["scan", "--deep"])
        assert args.deep is True
        assert args.fast is False

    def test_cli_parser_fast_and_deep_mutually_exclusive(self):
        """--fast and --deep cannot be used together."""
        import pytest
        from pybinaryguard.cli.main import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["scan", "--fast", "--deep"])


# ---------------------------------------------------------------------------
# Package model additions
# ---------------------------------------------------------------------------

class TestPackageModelAdditions:
    def test_shared_object_sha256_default_none(self):
        so = SharedObjectInfo(path="/test.so", filename="test.so")
        assert so.sha256 is None

    def test_package_name_property(self):
        pkg = PackageBinaryInfo(
            package_name="torch", package_version="2.0", install_path="/tmp"
        )
        assert pkg.name == "torch"

    def test_package_version_property(self):
        pkg = PackageBinaryInfo(
            package_name="torch", package_version="2.0", install_path="/tmp"
        )
        assert pkg.version == "2.0"

    def test_cuda_version_property(self):
        pkg = PackageBinaryInfo(
            package_name="torch", package_version="2.0", install_path="/tmp",
            cuda_build_version=(12, 4),
        )
        assert pkg.cuda_version == "12.4"

    def test_cuda_version_none_when_no_cuda(self):
        pkg = PackageBinaryInfo(
            package_name="numpy", package_version="1.26", install_path="/tmp",
        )
        assert pkg.cuda_version is None
