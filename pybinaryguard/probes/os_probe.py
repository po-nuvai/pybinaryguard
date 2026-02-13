"""Probe for OS, kernel, and virtualisation environment information."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional

from pybinaryguard.models.enums import ContainerRuntime

from .base import ProbeBase


class OsProbe(ProbeBase):
    """Collects operating-system and runtime-environment metadata.

    - Distro name and version from ``/etc/os-release``.
    - Kernel version from ``platform.release()``.
    - Container detection: ``/.dockerenv``, ``/proc/1/cgroup``, env vars.
    - VM detection: ``/sys/class/dmi/id/`` sysfs entries.
    """

    name = "os"

    def collect(self) -> Dict[str, Any]:
        """Return OS, kernel, container, and VM information."""
        data: Dict[str, Any] = {}

        os_info = self._parse_os_release()
        data["os_name"] = os_info.get("name", "")
        data["os_version"] = os_info.get("version_id", os_info.get("version", ""))
        data["kernel_version"] = self._get_kernel_version()

        container_info = self._detect_container()
        data["is_container"] = container_info[0]
        data["container_runtime"] = container_info[1]

        data["is_virtual_machine"] = self._detect_vm()

        return data

    # ------------------------------------------------------------------
    # OS release
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_os_release() -> Dict[str, str]:
        """Parse ``/etc/os-release`` (or ``/usr/lib/os-release`` fallback).

        Returns a dictionary with keys lower-cased.  Values are unquoted.
        """
        for path in ("/etc/os-release", "/usr/lib/os-release"):
            try:
                with open(path, "r") as fh:
                    return OsProbe._parse_release_file(fh.read())
            except (FileNotFoundError, PermissionError, OSError):
                continue
        return {}

    @staticmethod
    def _parse_release_file(text: str) -> Dict[str, str]:
        """Parse key=value pairs from an os-release style file."""
        result: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().lower()
            value = value.strip()
            # Remove surrounding quotes (single or double)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            result[key] = value
        return result

    @staticmethod
    def _get_kernel_version() -> str:
        """Return the kernel version string."""
        try:
            return platform.release()
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Container detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_container() -> tuple:  # type: ignore[type-arg]
        """Detect whether we are running inside a container.

        Returns ``(is_container: bool, runtime: ContainerRuntime)``.
        """
        # Check /.dockerenv -- Docker creates this file
        if os.path.exists("/.dockerenv"):
            return (True, ContainerRuntime.DOCKER)

        # Check for Podman-specific environment variable
        if os.environ.get("container") == "podman":
            return (True, ContainerRuntime.PODMAN)

        # Check environment hints
        if os.environ.get("container") == "lxc":
            return (True, ContainerRuntime.LXC)

        # Check /run/.containerenv (Podman)
        if os.path.exists("/run/.containerenv"):
            return (True, ContainerRuntime.PODMAN)

        # Check /proc/1/cgroup for container signals
        runtime = OsProbe._detect_container_from_cgroup()
        if runtime is not None:
            return (True, runtime)

        # Check /proc/1/environ for container_runtime (requires root)
        runtime = OsProbe._detect_container_from_mountinfo()
        if runtime is not None:
            return (True, runtime)

        return (False, ContainerRuntime.NONE)

    @staticmethod
    def _detect_container_from_cgroup() -> Optional[ContainerRuntime]:
        """Inspect ``/proc/1/cgroup`` for container runtime indicators.

        Docker cgroups contain ``/docker/``, containerd uses
        ``/cri-containerd-``, and LXC uses ``/lxc/``.
        """
        try:
            with open("/proc/1/cgroup", "r") as fh:
                content = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        if "/docker/" in content or "/docker-" in content:
            return ContainerRuntime.DOCKER
        if "/lxc/" in content:
            return ContainerRuntime.LXC
        if "containerd" in content or "/cri-containerd-" in content:
            return ContainerRuntime.CONTAINERD
        if "/kubepods" in content:
            # Kubernetes pods -- usually containerd or Docker underneath
            return ContainerRuntime.CONTAINERD

        return None

    @staticmethod
    def _detect_container_from_mountinfo() -> Optional[ContainerRuntime]:
        """Inspect ``/proc/1/mountinfo`` for overlay filesystems.

        Container runtimes typically mount an overlay root filesystem.
        """
        try:
            with open("/proc/1/mountinfo", "r") as fh:
                content = fh.read()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        if "overlay" in content and ("docker" in content or "containerd" in content):
            if "docker" in content:
                return ContainerRuntime.DOCKER
            return ContainerRuntime.CONTAINERD

        return None

    # ------------------------------------------------------------------
    # VM detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_vm() -> bool:
        """Detect whether the system is running inside a virtual machine.

        Reads DMI/SMBIOS information from ``/sys/class/dmi/id/`` and
        checks for known hypervisor product names.
        """
        vm_indicators = {
            "vmware",
            "virtualbox",
            "vbox",
            "kvm",
            "qemu",
            "xen",
            "hyper-v",
            "microsoft corporation",
            "parallels",
            "bochs",
            "bhyve",
            "amazon ec2",
            "google compute engine",
            "openstack",
            "nutanix",
        }

        dmi_files = [
            "/sys/class/dmi/id/product_name",
            "/sys/class/dmi/id/sys_vendor",
            "/sys/class/dmi/id/board_vendor",
            "/sys/class/dmi/id/bios_vendor",
            "/sys/class/dmi/id/chassis_vendor",
        ]

        for dmi_path in dmi_files:
            try:
                with open(dmi_path, "r") as fh:
                    value = fh.read().strip().lower()
                for indicator in vm_indicators:
                    if indicator in value:
                        return True
            except (FileNotFoundError, PermissionError, OSError):
                continue

        # Fallback: check /proc/cpuinfo for hypervisor flag
        try:
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    if line.strip().lower().startswith("flags"):
                        if "hypervisor" in line.lower():
                            return True
                        break
        except (FileNotFoundError, PermissionError, OSError):
            pass

        return False
