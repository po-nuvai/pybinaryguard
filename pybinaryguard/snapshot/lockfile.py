"""Lockfile format for environment snapshots.

JSON-based lockfile that captures binary hashes, GPU stack, and system state
beyond what requirements.txt provides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PackageSnapshot:
    """Snapshot of a single package's binary state."""

    name: str
    version: str
    cuda_version: Optional[str] = None
    binary_hashes: Dict[str, str] = field(default_factory=dict)  # path -> SHA256
    manylinux_tag: Optional[str] = None
    wheel_tags: List[str] = field(default_factory=list)


@dataclass
class SystemSnapshot:
    """Snapshot of system configuration."""

    python_version: str
    platform: str
    architecture: str
    glibc_version: Optional[str] = None
    cuda_runtime: Optional[str] = None
    cuda_driver: Optional[str] = None
    cuda_compute_capability: Optional[str] = None
    tensorrt_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    detected_board: Optional[str] = None
    cpu_flags: List[str] = field(default_factory=list)
    container_runtime: Optional[str] = None


@dataclass
class Lockfile:
    """Complete environment lockfile.

    Captures everything needed to reproduce a binary-compatible environment,
    including information that pip freeze and conda export don't provide.
    """

    version: str = "1.0"
    timestamp: str = ""
    system: Optional[SystemSnapshot] = None
    packages: List[PackageSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert lockfile to dictionary."""
        return {
            "pybinaryguard_lockfile": self.version,
            "timestamp": self.timestamp,
            "system": {
                "python_version": self.system.python_version if self.system else "",
                "platform": self.system.platform if self.system else "",
                "architecture": self.system.architecture if self.system else "",
                "glibc_version": self.system.glibc_version if self.system else None,
                "cuda_runtime": self.system.cuda_runtime if self.system else None,
                "cuda_driver": self.system.cuda_driver if self.system else None,
                "cuda_compute_capability": self.system.cuda_compute_capability if self.system else None,
                "tensorrt_version": self.system.tensorrt_version if self.system else None,
                "cudnn_version": self.system.cudnn_version if self.system else None,
                "detected_board": self.system.detected_board if self.system else None,
                "cpu_flags": self.system.cpu_flags if self.system else [],
                "container_runtime": self.system.container_runtime if self.system else None,
            },
            "packages": [
                {
                    "name": pkg.name,
                    "version": pkg.version,
                    "cuda_version": pkg.cuda_version,
                    "binary_hashes": pkg.binary_hashes,
                    "manylinux_tag": pkg.manylinux_tag,
                    "wheel_tags": pkg.wheel_tags,
                }
                for pkg in self.packages
            ],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert lockfile to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save lockfile to disk."""
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Lockfile:
        """Load lockfile from dictionary."""
        system_data = data.get("system", {})
        system = SystemSnapshot(
            python_version=system_data.get("python_version", ""),
            platform=system_data.get("platform", ""),
            architecture=system_data.get("architecture", ""),
            glibc_version=system_data.get("glibc_version"),
            cuda_runtime=system_data.get("cuda_runtime"),
            cuda_driver=system_data.get("cuda_driver"),
            cuda_compute_capability=system_data.get("cuda_compute_capability"),
            tensorrt_version=system_data.get("tensorrt_version"),
            cudnn_version=system_data.get("cudnn_version"),
            detected_board=system_data.get("detected_board"),
            cpu_flags=system_data.get("cpu_flags", []),
            container_runtime=system_data.get("container_runtime"),
        )

        packages = [
            PackageSnapshot(
                name=pkg["name"],
                version=pkg["version"],
                cuda_version=pkg.get("cuda_version"),
                binary_hashes=pkg.get("binary_hashes", {}),
                manylinux_tag=pkg.get("manylinux_tag"),
                wheel_tags=pkg.get("wheel_tags", []),
            )
            for pkg in data.get("packages", [])
        ]

        return cls(
            version=data.get("pybinaryguard_lockfile", "1.0"),
            timestamp=data.get("timestamp", ""),
            system=system,
            packages=packages,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Lockfile:
        """Load lockfile from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str) -> Lockfile:
        """Load lockfile from disk."""
        return cls.from_json(Path(path).read_text())


def load_lockfile(path: str) -> Lockfile:
    """Convenience function to load a lockfile."""
    return Lockfile.load(path)
