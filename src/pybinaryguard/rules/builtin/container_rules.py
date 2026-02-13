"""Container environment compatibility rules.

These rules detect GPU-related misconfiguration when running inside a
container (Docker, Podman, etc.).  Containers do not automatically get
access to the host GPU unless explicitly configured.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule
from pybinaryguard.rules.builtin.cuda_rules import (
    _driver_major,
    _is_cuda_package,
    _max_cuda_for_driver,
    _fmt_ver,
)


class ContainerNoGPUMountRule(Rule):
    """Warns when a GPU package runs inside a container without GPU access.

    Running ``docker run`` without ``--gpus all`` (or the NVIDIA Container
    Toolkit) means the GPU is invisible to the container.  The package
    will silently fall back to CPU or raise an error.
    """

    rule_id = "CONTAINER_NO_GPU_MOUNT"
    description = (
        "Warn when GPU packages run inside a container that lacks GPU "
        "device mounts."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.is_container and not profile.gpu_available

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        for pkg in packages:
            if not _is_cuda_package(pkg):
                continue
            findings.append(
                Finding(
                    rule_id=self.rule_id,
                    severity=Severity.WARNING,
                    title=(
                        f"{pkg.package_name} needs a GPU but none is "
                        f"visible in this container"
                    ),
                    explanation=(
                        f"You are running inside a container "
                        f"({profile.container_runtime.value}) and package "
                        f"{pkg.package_name} {pkg.package_version} is a "
                        f"GPU-enabled build, but no GPU device was "
                        f"detected inside this container.  This usually "
                        f"means the container was started without GPU "
                        f"passthrough."
                    ),
                    technical_detail=(
                        f"Container runtime: "
                        f"{profile.container_runtime.value}, "
                        f"GPU visible: False"
                    ),
                    suggestion=(
                        f"Run the container with GPU access:\n"
                        f"  docker run --gpus all ...   "
                        f"# Docker with NVIDIA Container Toolkit\n"
                        f"  podman run --device nvidia.com/gpu=all ...  "
                        f"# Podman with CDI\n\n"
                        f"Make sure the NVIDIA Container Toolkit is "
                        f"installed on the host:\n"
                        f"  sudo apt install nvidia-container-toolkit  "
                        f"# Debian/Ubuntu\n"
                        f"  sudo systemctl restart docker"
                    ),
                    package=pkg.package_name,
                    package_version=pkg.package_version,
                )
            )
        return findings


class ContainerDriverMismatchRule(Rule):
    """Warns when the container's CUDA may exceed the host driver capability.

    Inside a container the CUDA toolkit version can be newer than what
    the host GPU driver supports.  The container shares the host's
    kernel-mode driver, so if the driver is too old, CUDA calls will
    fail.
    """

    rule_id = "CONTAINER_DRIVER_MISMATCH"
    description = (
        "Check that the container's CUDA version does not exceed the "
        "host driver's capability."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        return (
            profile.is_container
            and profile.gpu_available
            and profile.gpu_driver_version is not None
            and profile.cuda_runtime_version is not None
        )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []
        if (
            profile.gpu_driver_version is None
            or profile.cuda_runtime_version is None
        ):
            return findings

        drv_major = _driver_major(profile.gpu_driver_version)
        if drv_major is None:
            return findings

        max_cuda = _max_cuda_for_driver(drv_major)
        if max_cuda is None:
            return findings

        cuda_rt = profile.cuda_runtime_version
        if cuda_rt <= max_cuda:
            return findings

        findings.append(
            Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title=(
                    "Container CUDA version exceeds host driver capability"
                ),
                explanation=(
                    f"This container has CUDA {_fmt_ver(cuda_rt)} "
                    f"installed, but the host's NVIDIA driver "
                    f"({profile.gpu_driver_version}) only supports CUDA "
                    f"up to {_fmt_ver(max_cuda)}.  Because the container "
                    f"shares the host's kernel-mode GPU driver, CUDA "
                    f"calls will fail even though the CUDA toolkit is "
                    f"installed inside the container."
                ),
                technical_detail=(
                    f"Container CUDA: {_fmt_ver(cuda_rt)}, "
                    f"Host driver: {profile.gpu_driver_version}, "
                    f"Max CUDA for driver: {_fmt_ver(max_cuda)}"
                ),
                suggestion=(
                    f"Option 1 -- upgrade the GPU driver on the host to "
                    f"one that supports CUDA {_fmt_ver(cuda_rt)}.\n\n"
                    f"Option 2 -- use a container image with CUDA <= "
                    f"{_fmt_ver(max_cuda)}:\n"
                    f"  docker run --gpus all nvidia/cuda:"
                    f"{_fmt_ver(max_cuda)}-runtime-ubuntu22.04"
                ),
            )
        )
        return findings
