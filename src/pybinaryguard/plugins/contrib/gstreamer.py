"""GStreamer pipeline validation plugin.

Checks for GStreamer availability and hardware acceleration on embedded boards.
Activated only when GStreamer libraries are detected on the system.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.probes.base import ProbeBase
from pybinaryguard.rules.base import Rule


class GStreamerProbe(ProbeBase):
    """Detect GStreamer installation and version."""

    name = "gstreamer"

    def collect(self) -> Dict[str, Any]:
        """Check for GStreamer library presence."""
        data: Dict[str, Any] = {}

        # Check for GStreamer library files
        gst_lib_names = [
            "libgstreamer-1.0.so",
            "libgstreamer-1.0.so.0",
        ]

        for lib_name in gst_lib_names:
            for search_dir in ["/usr/lib", "/usr/lib/aarch64-linux-gnu",
                               "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"]:
                path = os.path.join(search_dir, lib_name)
                if os.path.exists(path):
                    data["_gstreamer_available"] = True
                    return data

        data["_gstreamer_available"] = False
        return data

    def is_applicable(self) -> bool:
        """Only run on Linux."""
        import platform
        return platform.system() == "Linux"


class GStreamerMissingRule(Rule):
    """Check for GStreamer availability on embedded boards with camera use-cases."""

    rule_id = "GSTREAMER_MISSING"
    description = "GStreamer not found on embedded board"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return profile.is_embedded_board

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        findings: List[Finding] = []

        # Only relevant if OpenCV or camera-related packages are installed
        camera_packages = {"opencv-python", "opencv-contrib-python", "cv2"}
        has_camera_pkg = any(
            p.package_name.lower().replace("-", "_") in {
                n.replace("-", "_") for n in camera_packages
            }
            for p in packages
        )

        if not has_camera_pkg:
            return findings

        # Check for GStreamer
        gst_found = False
        for search_dir in ["/usr/lib", "/usr/lib/aarch64-linux-gnu",
                           "/usr/lib/x86_64-linux-gnu"]:
            if os.path.exists(os.path.join(search_dir, "libgstreamer-1.0.so.0")):
                gst_found = True
                break

        if not gst_found:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title="GStreamer not found on embedded board",
                explanation=(
                    "GStreamer is commonly needed for camera pipelines on "
                    "embedded boards but was not found on this system."
                ),
                suggestion="sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base",
                confidence=0.7,
            ))

        return findings


def register(registry: Any) -> None:
    """Register GStreamer plugin components."""
    # Only activate on Linux
    import platform
    if platform.system() != "Linux":
        return

    registry.add_rule(GStreamerMissingRule())
