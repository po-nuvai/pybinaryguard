"""Tests for WheelAnalyzer and related helper functions."""

from __future__ import annotations

import os
import textwrap
from typing import Any

import pytest

from pybinaryguard.analyzers.wheel_analyzer import (
    WheelAnalyzer,
    _normalize_name,
    find_dist_info_dirs,
)
from pybinaryguard.models.package import PackageBinaryInfo, WheelTag


# ---------------------------------------------------------------------------
# _normalize_name helper
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_lowercase_simple(self) -> None:
        assert _normalize_name("NumPy") == "numpy"

    def test_replace_underscores(self) -> None:
        assert _normalize_name("my_package") == "my-package"

    def test_replace_dots(self) -> None:
        assert _normalize_name("zope.interface") == "zope-interface"

    def test_consecutive_separators(self) -> None:
        assert _normalize_name("some__pkg") == "some-pkg"

    def test_mixed_separators(self) -> None:
        assert _normalize_name("My-Cool_Package.v2") == "my-cool-package-v2"


# ---------------------------------------------------------------------------
# WheelAnalyzer._parse_wheel_file
# ---------------------------------------------------------------------------


class TestParseWheelFile:
    def test_parse_single_tag(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: setuptools\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp312-cp312-manylinux_2_17_x86_64\n"
        )

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0.0",
            install_path=str(tmp_path / "pkg"),
        )
        WheelAnalyzer._parse_wheel_file(str(dist_info), pkg)

        assert len(pkg.wheel_tags) == 1
        assert pkg.wheel_tags[0].interpreter == "cp312"
        assert pkg.wheel_tags[0].abi == "cp312"
        assert pkg.wheel_tags[0].platform == "manylinux_2_17_x86_64"

    def test_parse_multiple_tags(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Tag: cp311-cp311-manylinux_2_17_x86_64\n"
            "Tag: cp312-cp312-manylinux_2_17_x86_64\n"
        )

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0.0",
            install_path=str(tmp_path / "pkg"),
        )
        WheelAnalyzer._parse_wheel_file(str(dist_info), pkg)

        assert len(pkg.wheel_tags) == 2
        interpreters = {t.interpreter for t in pkg.wheel_tags}
        assert "cp311" in interpreters
        assert "cp312" in interpreters

    def test_parse_missing_wheel_file(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        # No WHEEL file created

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0.0",
            install_path=str(tmp_path / "pkg"),
        )
        WheelAnalyzer._parse_wheel_file(str(dist_info), pkg)

        # Should not raise, just have no tags
        assert len(pkg.wheel_tags) == 0

    def test_parse_compound_platform_tag(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "WHEEL").write_text(
            "Tag: cp312-cp312-manylinux_2_17_x86_64-manylinux2014_x86_64\n"
        )

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0.0",
            install_path=str(tmp_path / "pkg"),
        )
        WheelAnalyzer._parse_wheel_file(str(dist_info), pkg)

        assert len(pkg.wheel_tags) == 1
        # Compound platform tag: parts[2:] joined with hyphens
        assert "manylinux_2_17_x86_64" in pkg.wheel_tags[0].platform


# ---------------------------------------------------------------------------
# WheelAnalyzer._parse_metadata_file
# ---------------------------------------------------------------------------


class TestParseMetadataFile:
    def test_parse_name_and_version(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            "Name: MyPackage\n"
            "Version: 2.5.3\n"
            "Summary: A test package\n"
        )

        pkg = PackageBinaryInfo(
            package_name="placeholder",
            package_version="placeholder",
            install_path=str(tmp_path / "pkg"),
        )
        WheelAnalyzer._parse_metadata_file(str(dist_info), pkg)

        assert pkg.package_name == "MyPackage"
        assert pkg.package_version == "2.5.3"

    def test_parse_missing_metadata_file(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        # No METADATA file

        pkg = PackageBinaryInfo(
            package_name="original",
            package_version="1.0",
            install_path=str(tmp_path / "pkg"),
        )
        WheelAnalyzer._parse_metadata_file(str(dist_info), pkg)

        # Should not modify existing values
        assert pkg.package_name == "original"
        assert pkg.package_version == "1.0"


# ---------------------------------------------------------------------------
# WheelAnalyzer._detect_pure_python
# ---------------------------------------------------------------------------


class TestDetectPurePython:
    def test_pure_python_from_tags(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "RECORD").write_text("pkg/__init__.py,sha256=abc,100\n")

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0",
            install_path=str(tmp_path / "pkg"),
            wheel_tags=[WheelTag(interpreter="py3", abi="none", platform="any")],
        )
        WheelAnalyzer._detect_pure_python(str(dist_info), pkg)

        assert pkg.is_pure_python is True

    def test_not_pure_python_from_platform_tag(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0",
            install_path=str(tmp_path / "pkg"),
            wheel_tags=[WheelTag(interpreter="cp312", abi="cp312", platform="manylinux_2_17_x86_64")],
        )
        WheelAnalyzer._detect_pure_python(str(dist_info), pkg)

        assert pkg.is_pure_python is False

    def test_pure_python_from_record_no_tags(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "RECORD").write_text(
            "pkg/__init__.py,sha256=abc,100\n"
            "pkg/utils.py,sha256=def,200\n"
        )

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0",
            install_path=str(tmp_path / "pkg"),
            wheel_tags=[],  # No tags -> fallback to RECORD scan
        )
        WheelAnalyzer._detect_pure_python(str(dist_info), pkg)

        assert pkg.is_pure_python is True

    def test_not_pure_python_from_record_so_file(self, tmp_path: Any) -> None:
        dist_info = tmp_path / "pkg-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "RECORD").write_text(
            "pkg/__init__.py,sha256=abc,100\n"
            "pkg/core.cpython-312-x86_64-linux-gnu.so,sha256=def,50000\n"
        )

        pkg = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0",
            install_path=str(tmp_path / "pkg"),
            wheel_tags=[],  # No tags -> fallback to RECORD scan
        )
        WheelAnalyzer._detect_pure_python(str(dist_info), pkg)

        assert pkg.is_pure_python is False


# ---------------------------------------------------------------------------
# WheelAnalyzer._extract_cuda_version
# ---------------------------------------------------------------------------


class TestExtractCudaVersion:
    def test_extract_cu124(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="torch",
            package_version="2.4.0+cu124",
            install_path="/tmp/torch",
        )
        WheelAnalyzer._extract_cuda_version(pkg)
        assert pkg.cuda_build_version == (12, 4)

    def test_extract_cu118(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="torch",
            package_version="2.1.0+cu118",
            install_path="/tmp/torch",
        )
        WheelAnalyzer._extract_cuda_version(pkg)
        assert pkg.cuda_build_version == (11, 8)

    def test_extract_cu121(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="torch",
            package_version="2.2.0+cu121",
            install_path="/tmp/torch",
        )
        WheelAnalyzer._extract_cuda_version(pkg)
        assert pkg.cuda_build_version == (12, 1)

    def test_no_cuda_suffix(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="numpy",
            package_version="1.26.4",
            install_path="/tmp/numpy",
        )
        WheelAnalyzer._extract_cuda_version(pkg)
        assert pkg.cuda_build_version is None

    def test_cpu_suffix_no_cuda(self) -> None:
        pkg = PackageBinaryInfo(
            package_name="torch",
            package_version="2.1.0+cpu",
            install_path="/tmp/torch",
        )
        WheelAnalyzer._extract_cuda_version(pkg)
        assert pkg.cuda_build_version is None


# ---------------------------------------------------------------------------
# find_dist_info_dirs
# ---------------------------------------------------------------------------


class TestFindDistInfoDirs:
    def test_finds_dist_info(self, tmp_path: Any) -> None:
        sp = tmp_path / "site-packages"
        sp.mkdir()
        dist = sp / "mypackage-1.0.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Name: mypackage\nVersion: 1.0.0\n"
        )

        results = find_dist_info_dirs(str(sp))
        assert len(results) == 1
        assert results[0][1] == "mypackage"
        assert results[0][2] == "1.0.0"

    def test_empty_site_packages(self, tmp_path: Any) -> None:
        sp = tmp_path / "site-packages"
        sp.mkdir()

        results = find_dist_info_dirs(str(sp))
        assert results == []

    def test_nonexistent_directory(self) -> None:
        results = find_dist_info_dirs("/nonexistent/path")
        assert results == []

    def test_multiple_dist_infos(self, tmp_path: Any) -> None:
        sp = tmp_path / "site-packages"
        sp.mkdir()

        for name, ver in [("numpy", "1.26.4"), ("pandas", "2.1.0")]:
            dist = sp / f"{name}-{ver}.dist-info"
            dist.mkdir()
            (dist / "METADATA").write_text(
                f"Name: {name}\nVersion: {ver}\n"
            )

        results = find_dist_info_dirs(str(sp))
        assert len(results) == 2
        names = {r[1] for r in results}
        assert "numpy" in names
        assert "pandas" in names
