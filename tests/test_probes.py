"""Tests for system probes with mocking."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict
from unittest import mock

import pytest

from pybinaryguard.models.enums import Architecture, ContainerRuntime
from pybinaryguard.probes.python_probe import PythonProbe
from pybinaryguard.probes.glibc_probe import GlibcProbe
from pybinaryguard.probes.cpu_probe import CpuProbe
from pybinaryguard.probes.os_probe import OsProbe
from pybinaryguard.probes.library_probe import LibraryProbe
from pybinaryguard.probes.board_probe import BoardProbe
from pybinaryguard.probes.gpu_probe import GpuProbe


# ---------------------------------------------------------------------------
# PythonProbe
# ---------------------------------------------------------------------------


class TestPythonProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = PythonProbe()
        assert probe.is_applicable() is True

    def test_collect_returns_python_version(self) -> None:
        probe = PythonProbe()
        data = probe.collect()
        assert "python_version" in data
        assert isinstance(data["python_version"], tuple)
        assert len(data["python_version"]) == 3

    def test_collect_returns_implementation(self) -> None:
        probe = PythonProbe()
        data = probe.collect()
        assert "python_implementation" in data
        assert data["python_implementation"] in ("cpython", "pypy")

    def test_collect_returns_abi_tag(self) -> None:
        probe = PythonProbe()
        data = probe.collect()
        assert "python_abi_tag" in data
        assert isinstance(data["python_abi_tag"], str)

    def test_collect_returns_executable(self) -> None:
        probe = PythonProbe()
        data = probe.collect()
        assert "python_executable" in data

    def test_collect_stable_abi_for_cpython_3x(self) -> None:
        probe = PythonProbe()
        data = probe.collect()
        if sys.implementation.name == "cpython" and sys.version_info >= (3, 2):
            assert data["stable_abi_supported"] is True


# ---------------------------------------------------------------------------
# GlibcProbe
# ---------------------------------------------------------------------------


class TestGlibcProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = GlibcProbe()
        assert probe.is_applicable() is True

    @mock.patch.object(GlibcProbe, "_detect_musl", return_value=None)
    @mock.patch.object(GlibcProbe, "_glibc_via_confstr", return_value=(2, 35))
    def test_collect_glibc_via_confstr(self, mock_confstr: Any, mock_musl: Any) -> None:
        probe = GlibcProbe()
        data = probe.collect()
        assert data["glibc_version"] == (2, 35)

    @mock.patch.object(GlibcProbe, "_detect_musl", return_value=None)
    @mock.patch.object(GlibcProbe, "_glibc_via_confstr", return_value=None)
    @mock.patch.object(GlibcProbe, "_glibc_via_ctypes", return_value=(2, 31))
    def test_collect_glibc_via_ctypes_fallback(
        self, mock_ctypes: Any, mock_confstr: Any, mock_musl: Any
    ) -> None:
        probe = GlibcProbe()
        data = probe.collect()
        assert data["glibc_version"] == (2, 31)

    @mock.patch.object(GlibcProbe, "_detect_musl", return_value=(1, 2))
    def test_collect_musl_detected(self, mock_musl: Any) -> None:
        probe = GlibcProbe()
        data = probe.collect()
        assert data["musl_version"] == (1, 2)
        assert "glibc_version" not in data

    def test_parse_glibc_version_string(self) -> None:
        assert GlibcProbe._parse_glibc_version_string("glibc 2.35") == (2, 35)

    def test_parse_glibc_version_string_no_match(self) -> None:
        assert GlibcProbe._parse_glibc_version_string("unknown") is None

    def test_parse_musl_version_output(self) -> None:
        output = "musl libc (x86_64)\nVersion 1.2.3\n"
        assert GlibcProbe._parse_musl_version_output(output) == (1, 2)


# ---------------------------------------------------------------------------
# CpuProbe
# ---------------------------------------------------------------------------

_X86_CPUINFO = """\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i7-12700K
flags\t\t: fpu vme de pse avx avx2 avx512f sse4_2
"""

_ARM_CPUINFO = """\
processor\t: 0
BogoMIPS\t: 48.00
Features\t: fp asimd neon sha1 sha2 crc32
CPU implementer\t: 0x41
CPU part\t: 0xd05
Hardware\t: BCM2835
"""


class TestCpuProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = CpuProbe()
        assert probe.is_applicable() is True

    @mock.patch.object(CpuProbe, "_get_machine", return_value="x86_64")
    @mock.patch.object(CpuProbe, "_read_cpuinfo", return_value=_X86_CPUINFO)
    def test_collect_x86_architecture(self, mock_cpuinfo: Any, mock_machine: Any) -> None:
        probe = CpuProbe()
        data = probe.collect()
        assert data["architecture"] == Architecture.X86_64
        assert data["cpu_model"] == "Intel(R) Core(TM) i7-12700K"
        assert data["has_avx"] is True
        assert data["has_avx2"] is True
        assert data["has_avx512"] is True
        assert data["has_sse42"] is True

    @mock.patch.object(CpuProbe, "_get_machine", return_value="aarch64")
    @mock.patch.object(CpuProbe, "_read_cpuinfo", return_value=_ARM_CPUINFO)
    def test_collect_arm_architecture(self, mock_cpuinfo: Any, mock_machine: Any) -> None:
        probe = CpuProbe()
        data = probe.collect()
        assert data["architecture"] == Architecture.AARCH64
        assert data["has_neon"] is True
        assert data["has_avx"] is False
        assert data["has_avx2"] is False

    @mock.patch.object(CpuProbe, "_get_machine", return_value="x86_64")
    @mock.patch.object(CpuProbe, "_read_cpuinfo", return_value="")
    def test_collect_empty_cpuinfo(self, mock_cpuinfo: Any, mock_machine: Any) -> None:
        probe = CpuProbe()
        data = probe.collect()
        assert data["cpu_model"] == ""
        assert data["cpu_flags"] == frozenset()

    def test_extract_cpu_model_arm_fallback_hardware(self) -> None:
        model = CpuProbe._extract_cpu_model(_ARM_CPUINFO, "aarch64")
        assert model == "BCM2835"


# ---------------------------------------------------------------------------
# OsProbe
# ---------------------------------------------------------------------------


class TestOsProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = OsProbe()
        assert probe.is_applicable() is True

    def test_parse_release_file_ubuntu(self) -> None:
        content = 'NAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n'
        result = OsProbe._parse_release_file(content)
        assert result["name"] == "Ubuntu"
        assert result["version_id"] == "22.04"

    def test_parse_release_file_alpine(self) -> None:
        content = "NAME='Alpine Linux'\nVERSION_ID=3.19\nID=alpine\n"
        result = OsProbe._parse_release_file(content)
        assert result["name"] == "Alpine Linux"
        assert result["version_id"] == "3.19"

    def test_parse_release_file_skips_comments(self) -> None:
        content = "# comment\nNAME=Test\n"
        result = OsProbe._parse_release_file(content)
        assert result["name"] == "Test"

    @mock.patch("os.path.exists", return_value=True)
    def test_detect_container_dockerenv(self, mock_exists: Any) -> None:
        is_container, runtime = OsProbe._detect_container()
        assert is_container is True
        assert runtime == ContainerRuntime.DOCKER

    @mock.patch("os.path.exists", return_value=False)
    @mock.patch.dict(os.environ, {"container": "podman"})
    def test_detect_container_podman_env(self, mock_exists: Any) -> None:
        is_container, runtime = OsProbe._detect_container()
        assert is_container is True
        assert runtime == ContainerRuntime.PODMAN

    @mock.patch("os.path.exists", return_value=False)
    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch.object(OsProbe, "_detect_container_from_cgroup", return_value=None)
    @mock.patch.object(OsProbe, "_detect_container_from_mountinfo", return_value=None)
    def test_detect_container_not_in_container(
        self, mock_mountinfo: Any, mock_cgroup: Any, mock_exists: Any
    ) -> None:
        is_container, runtime = OsProbe._detect_container()
        assert is_container is False
        assert runtime == ContainerRuntime.NONE


# ---------------------------------------------------------------------------
# LibraryProbe
# ---------------------------------------------------------------------------


class TestLibraryProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = LibraryProbe()
        assert probe.is_applicable() is True

    @mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": "/usr/local/lib:/opt/lib"})
    def test_get_ld_library_path(self) -> None:
        result = LibraryProbe._get_ld_library_path()
        assert "/usr/local/lib" in result
        assert "/opt/lib" in result

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_ld_library_path_empty(self) -> None:
        result = LibraryProbe._get_ld_library_path()
        assert result == ()

    def test_parse_ldconfig_output(self) -> None:
        output = (
            "42 libs found in cache `/etc/ld.so.cache'\n"
            "\tlibz.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libz.so.1\n"
            "\tlibm.so.6 (libc6,x86-64) => /lib/x86_64-linux-gnu/libm.so.6\n"
        )
        result = LibraryProbe._parse_ldconfig_output(output)
        assert result["libz.so.1"] == "/lib/x86_64-linux-gnu/libz.so.1"
        assert result["libm.so.6"] == "/lib/x86_64-linux-gnu/libm.so.6"

    def test_parse_ldconfig_output_empty(self) -> None:
        result = LibraryProbe._parse_ldconfig_output("")
        assert result == {}


# ---------------------------------------------------------------------------
# BoardProbe
# ---------------------------------------------------------------------------


class TestBoardProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = BoardProbe()
        assert probe.is_applicable() is True

    @mock.patch.object(BoardProbe, "_read_device_tree_model", return_value="")
    @mock.patch.object(BoardProbe, "_read_text_file", return_value="")
    def test_collect_no_board_detected(self, mock_read: Any, mock_dt: Any) -> None:
        probe = BoardProbe()
        data = probe.collect()
        assert data["is_embedded_board"] is False

    def test_detect_raspberry_pi(self) -> None:
        result = BoardProbe._detect_raspberry_pi("Raspberry Pi 4 Model B Rev 1.5")
        assert result == "Raspberry Pi 4 Model B Rev 1.5"

    def test_detect_raspberry_pi_none(self) -> None:
        result = BoardProbe._detect_raspberry_pi("Some Other Device")
        assert result is None

    def test_detect_beaglebone(self) -> None:
        result = BoardProbe._detect_beaglebone("BeagleBone Black Rev C")
        assert result == "BeagleBone Black Rev C"

    def test_detect_beaglebone_none(self) -> None:
        result = BoardProbe._detect_beaglebone("Not a Beagle")
        assert result is None

    @mock.patch.object(BoardProbe, "_read_text_file", return_value="# R36 (release), REVISION: 2.0")
    def test_detect_jetson_via_tegra_release(self, mock_read: Any) -> None:
        result = BoardProbe._detect_jetson("NVIDIA Jetson AGX Orin")
        assert result is not None
        assert result["name"] == "NVIDIA Jetson AGX Orin"

    def test_detect_jetson_via_dt_model(self) -> None:
        # No tegra_release file, but dt_model contains "jetson"
        with mock.patch.object(BoardProbe, "_read_text_file", return_value=""):
            result = BoardProbe._detect_jetson("NVIDIA Jetson Nano Developer Kit")
            assert result is not None
            assert result["name"] == "NVIDIA Jetson Nano Developer Kit"


# ---------------------------------------------------------------------------
# GpuProbe
# ---------------------------------------------------------------------------


class TestGpuProbe:
    def test_is_applicable_always_true(self) -> None:
        probe = GpuProbe()
        assert probe.is_applicable() is True

    @mock.patch.object(GpuProbe, "_check_device_nodes", return_value=False)
    @mock.patch.object(GpuProbe, "_parse_proc_driver_version", return_value=None)
    @mock.patch.object(GpuProbe, "_query_nvml", return_value=None)
    @mock.patch.object(GpuProbe, "_query_cuda_runtime", return_value=None)
    @mock.patch.object(GpuProbe, "_detect_cuda_toolkit", return_value=None)
    @mock.patch.object(GpuProbe, "_detect_jetson_gpu", return_value=None)
    @mock.patch.object(GpuProbe, "_detect_cudnn", return_value=None)
    def test_collect_no_gpu(self, *mocks: Any) -> None:
        probe = GpuProbe()
        data = probe.collect()
        assert data["gpu_available"] is False

    @mock.patch.object(GpuProbe, "_check_device_nodes", return_value=True)
    @mock.patch.object(GpuProbe, "_parse_proc_driver_version", return_value="535.129.03")
    @mock.patch.object(GpuProbe, "_query_nvml", return_value=None)
    @mock.patch.object(GpuProbe, "_query_cuda_runtime", return_value=(12, 2))
    @mock.patch.object(GpuProbe, "_detect_cuda_toolkit", return_value=None)
    @mock.patch.object(GpuProbe, "_detect_jetson_gpu", return_value=None)
    @mock.patch.object(GpuProbe, "_detect_cudnn", return_value=None)
    def test_collect_with_gpu(self, *mocks: Any) -> None:
        probe = GpuProbe()
        data = probe.collect()
        assert data["gpu_available"] is True
        assert data["gpu_driver_version"] == "535.129.03"
        assert data["cuda_runtime_version"] == (12, 2)

    def test_parse_proc_driver_version_content(self) -> None:
        content = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.129.03  Wed Aug  2 14:30:28 UTC 2023\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=content)):
            result = GpuProbe._parse_proc_driver_version()
            assert result == "535.129.03"

    @mock.patch("builtins.open", side_effect=FileNotFoundError)
    def test_parse_proc_driver_version_no_file(self, mock_open: Any) -> None:
        result = GpuProbe._parse_proc_driver_version()
        assert result is None
