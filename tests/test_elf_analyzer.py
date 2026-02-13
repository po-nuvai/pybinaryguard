"""Tests for the MinimalELFParser and ELFAnalyzer."""

from __future__ import annotations

import os
import struct
from typing import Any

import pytest

from pybinaryguard.analyzers.elf_analyzer import (
    ELFAnalyzer,
    ELFParseError,
    MinimalELFParser,
    _parse_glibc_version,
)
from pybinaryguard.models.enums import Architecture
from pybinaryguard.models.package import PackageBinaryInfo, SharedObjectInfo


# ---------------------------------------------------------------------------
# ELF header builder helper
# ---------------------------------------------------------------------------


def make_elf_header(
    bits: int = 64,
    arch: str = "x86_64",
    endian: str = "little",
) -> bytes:
    """Create a minimal valid ELF file header.

    This produces a minimal ELF binary with just the ELF header, no sections
    or program headers. Suitable for basic parse tests.
    """
    magic = b"\x7fELF"
    ei_class = 2 if bits == 64 else 1
    ei_data = 1 if endian == "little" else 2
    ei_version = 1
    ei_osabi = 0
    ei_abiversion = 0
    ei_pad = b"\x00" * 7

    ident = magic + bytes([ei_class, ei_data, ei_version, ei_osabi, ei_abiversion]) + ei_pad

    arch_map = {
        "x86_64": 62,
        "aarch64": 183,
        "arm": 40,
        "i686": 3,
        "ppc64le": 21,
        "s390x": 22,
    }
    e_machine = arch_map.get(arch, 0)

    prefix = "<" if endian == "little" else ">"

    e_type = 3  # ET_DYN (shared object)
    e_version = 1

    if bits == 64:
        # Elf64_Ehdr after ident (48 bytes):
        # e_type(2) e_machine(2) e_version(4) e_entry(8) e_phoff(8)
        # e_shoff(8) e_flags(4) e_ehsize(2) e_phentsize(2) e_phnum(2)
        # e_shentsize(2) e_shnum(2) e_shstrndx(2)
        header = struct.pack(
            prefix + "HHIQQQIHHHHHH",
            e_type,
            e_machine,
            e_version,
            0,  # e_entry
            0,  # e_phoff (no program headers)
            0,  # e_shoff (no section headers)
            0,  # e_flags
            64,  # e_ehsize
            56,  # e_phentsize
            0,  # e_phnum
            64,  # e_shentsize
            0,  # e_shnum
            0,  # e_shstrndx
        )
    else:
        # Elf32_Ehdr after ident (36 bytes):
        header = struct.pack(
            prefix + "HHIIIIIHHHHHH",
            e_type,
            e_machine,
            e_version,
            0,  # e_entry
            0,  # e_phoff
            0,  # e_shoff
            0,  # e_flags
            52,  # e_ehsize
            32,  # e_phentsize
            0,  # e_phnum
            40,  # e_shentsize
            0,  # e_shnum
            0,  # e_shstrndx
        )

    return ident + header


# ---------------------------------------------------------------------------
# MinimalELFParser tests
# ---------------------------------------------------------------------------


class TestMinimalELFParser:
    def test_parse_x86_64_elf(self, tmp_path: Any) -> None:
        elf_data = make_elf_header(bits=64, arch="x86_64", endian="little")
        elf_path = str(tmp_path / "test.so")
        with open(elf_path, "wb") as f:
            f.write(elf_data)

        parser = MinimalELFParser(elf_path)
        info = parser.parse()
        assert info["ei_class"] == 64
        assert info["endianness"] == "little"
        assert info["e_machine"] == 62

    def test_parse_aarch64_elf(self, tmp_path: Any) -> None:
        elf_data = make_elf_header(bits=64, arch="aarch64", endian="little")
        elf_path = str(tmp_path / "test_arm.so")
        with open(elf_path, "wb") as f:
            f.write(elf_data)

        parser = MinimalELFParser(elf_path)
        info = parser.parse()
        assert info["ei_class"] == 64
        assert info["e_machine"] == 183

    def test_parse_32bit_arm_elf(self, tmp_path: Any) -> None:
        elf_data = make_elf_header(bits=32, arch="arm", endian="little")
        elf_path = str(tmp_path / "test_arm32.so")
        with open(elf_path, "wb") as f:
            f.write(elf_data)

        parser = MinimalELFParser(elf_path)
        info = parser.parse()
        assert info["ei_class"] == 32
        assert info["e_machine"] == 40

    def test_parse_invalid_magic_raises_error(self, tmp_path: Any) -> None:
        elf_path = str(tmp_path / "bad.so")
        with open(elf_path, "wb") as f:
            f.write(b"\x00\x00\x00\x00" + b"\x00" * 60)

        parser = MinimalELFParser(elf_path)
        with pytest.raises(ELFParseError, match="Not an ELF file"):
            parser.parse()

    def test_parse_corrupt_elf_truncated(self, tmp_path: Any) -> None:
        elf_path = str(tmp_path / "truncated.so")
        with open(elf_path, "wb") as f:
            f.write(b"\x7fELF\x02\x01")  # truncated after class+data

        parser = MinimalELFParser(elf_path)
        with pytest.raises(ELFParseError):
            parser.parse()

    def test_parse_nonexistent_file_raises_oserror(self) -> None:
        parser = MinimalELFParser("/nonexistent/path/to/file.so")
        with pytest.raises(OSError):
            parser.parse()

    def test_get_needed_empty_for_minimal_elf(self, tmp_path: Any) -> None:
        elf_data = make_elf_header(bits=64, arch="x86_64")
        elf_path = str(tmp_path / "minimal.so")
        with open(elf_path, "wb") as f:
            f.write(elf_data)

        parser = MinimalELFParser(elf_path)
        parser.parse()
        assert parser.get_needed() == []

    def test_get_version_requirements_empty_for_minimal_elf(self, tmp_path: Any) -> None:
        elf_data = make_elf_header(bits=64, arch="x86_64")
        elf_path = str(tmp_path / "minimal2.so")
        with open(elf_path, "wb") as f:
            f.write(elf_data)

        parser = MinimalELFParser(elf_path)
        parser.parse()
        assert parser.get_version_requirements() == []

    def test_parse_non_elf_text_file(self, tmp_path: Any) -> None:
        path = str(tmp_path / "text.so")
        with open(path, "w") as f:
            f.write("This is not an ELF file\n")

        parser = MinimalELFParser(path)
        with pytest.raises(ELFParseError, match="Not an ELF file"):
            parser.parse()

    def test_parse_big_endian_elf(self, tmp_path: Any) -> None:
        elf_data = make_elf_header(bits=64, arch="s390x", endian="big")
        elf_path = str(tmp_path / "big_endian.so")
        with open(elf_path, "wb") as f:
            f.write(elf_data)

        parser = MinimalELFParser(elf_path)
        info = parser.parse()
        assert info["endianness"] == "big"
        assert info["e_machine"] == 22


# ---------------------------------------------------------------------------
# ELFAnalyzer tests
# ---------------------------------------------------------------------------


class TestELFAnalyzer:
    def test_analyze_populates_shared_objects(self, tmp_path: Any) -> None:
        # Create a fake package directory with a .so file
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        so_path = pkg_dir / "core.cpython-312-x86_64-linux-gnu.so"
        elf_data = make_elf_header(bits=64, arch="x86_64")
        so_path.write_bytes(elf_data)

        pkg_info = PackageBinaryInfo(
            package_name="mypackage",
            package_version="1.0.0",
            install_path=str(pkg_dir),
        )

        analyzer = ELFAnalyzer()
        result = analyzer.analyze(pkg_info)

        assert result.is_pure_python is False
        assert len(result.shared_objects) == 1
        assert result.shared_objects[0].architecture == Architecture.X86_64
        assert result.shared_objects[0].elf_class == 64

    def test_analyze_nonexistent_install_path(self) -> None:
        pkg_info = PackageBinaryInfo(
            package_name="noexist",
            package_version="1.0",
            install_path="/nonexistent/path",
        )
        analyzer = ELFAnalyzer()
        result = analyzer.analyze(pkg_info)
        # Should return gracefully without raising
        assert result.is_pure_python is True

    def test_analyze_skips_non_elf_files(self, tmp_path: Any) -> None:
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        # A .so file that is not actually ELF
        fake_so = pkg_dir / "fake.so"
        fake_so.write_text("not an elf file")

        pkg_info = PackageBinaryInfo(
            package_name="pkg",
            package_version="1.0",
            install_path=str(pkg_dir),
        )
        analyzer = ELFAnalyzer()
        result = analyzer.analyze(pkg_info)
        # Should not raise, just skip the bad file
        assert result.so_count == 0

    def test_analyze_multiple_so_files(self, tmp_path: Any) -> None:
        pkg_dir = tmp_path / "multi"
        pkg_dir.mkdir()
        for i in range(3):
            so_path = pkg_dir / f"lib{i}.so"
            so_path.write_bytes(make_elf_header(bits=64, arch="x86_64"))

        pkg_info = PackageBinaryInfo(
            package_name="multi",
            package_version="1.0",
            install_path=str(pkg_dir),
        )
        analyzer = ELFAnalyzer()
        result = analyzer.analyze(pkg_info)
        assert result.so_count == 3


# ---------------------------------------------------------------------------
# _parse_glibc_version helper
# ---------------------------------------------------------------------------


class TestParseGlibcVersion:
    def test_parse_glibc_2_17(self) -> None:
        assert _parse_glibc_version("GLIBC_2.17") == (2, 17)

    def test_parse_glibc_2_35(self) -> None:
        assert _parse_glibc_version("GLIBC_2.35") == (2, 35)

    def test_parse_invalid(self) -> None:
        assert _parse_glibc_version("INVALID") is None

    def test_parse_no_prefix(self) -> None:
        assert _parse_glibc_version("2.28") is None  # requires GLIBC_ prefix

    def test_parse_partial_version(self) -> None:
        assert _parse_glibc_version("GLIBC_2") is None
