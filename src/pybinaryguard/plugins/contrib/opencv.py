"""OpenCV plugin for PyBinaryGuard.

Activates when the ``cv2`` package is importable.  Parses
``cv2.getBuildInformation()`` to detect build flags and checks for
common misconfiguration issues on embedded and GPU-equipped systems.

Provides
--------
- **OpenCVBuildProbe** -- Parses the OpenCV build information string to
  extract enabled/disabled modules and backend support.
- **OpenCVNoCUDARule** -- Reports when OpenCV lacks CUDA support on a
  GPU-equipped board (INFO severity).
- **OpenCVNoGStreamerRule** -- Reports when OpenCV lacks GStreamer support
  on an embedded board (WARNING severity).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, TYPE_CHECKING

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.probes.base import ProbeBase
from pybinaryguard.rules.base import Rule

if TYPE_CHECKING:
    from pybinaryguard.plugins.hooks import HookRegistry

logger = logging.getLogger(__name__)


# -- Build-info parser ------------------------------------------------------


def _parse_build_information(build_info: str) -> Dict[str, Any]:
    """Parse the string returned by ``cv2.getBuildInformation()``.

    The output is a multi-section text block.  We extract key boolean
    flags (CUDA, GStreamer, FFmpeg, etc.) and the OpenCV version.

    Args:
        build_info: The raw string from ``cv2.getBuildInformation()``.

    Returns:
        A dict with parsed flags. Keys include ``"cuda"``, ``"gstreamer"``,
        ``"ffmpeg"``, ``"opencl"``, ``"version"``, etc.  Values are booleans
        for feature flags and strings for version info.
    """
    result: Dict[str, Any] = {}

    # Extract version
    version_match = re.search(r"OpenCV\s+([\d.]+)", build_info)
    if version_match:
        result["version"] = version_match.group(1)

    # Common yes/no feature flags.  The build info uses patterns like:
    #   NVIDIA CUDA:                   YES (ver 11.4, ...)
    #   GStreamer:                      NO
    #   FFMPEG:                         YES
    _flag_patterns = {
        "cuda": r"NVIDIA CUDA:\s+(YES|NO)",
        "cudnn": r"cuDNN:\s+(YES|NO)",
        "gstreamer": r"GStreamer:\s+(YES|NO)",
        "ffmpeg": r"FFMPEG:\s+(YES|NO)",
        "opencl": r"OpenCL:\s+(YES|NO)",
        "v4l2": r"v4l/v4l2:\s+(YES|NO)",
        "gtk": r"GTK\+:\s+(YES|NO)",
        "qt": r"QT:\s+(YES|NO)",
        "tbb": r"TBB:\s+(YES|NO)",
        "openmp": r"OpenMP:\s+(YES|NO)",
        "vulkan": r"Vulkan:\s+(YES|NO)",
    }

    for key, pattern in _flag_patterns.items():
        match = re.search(pattern, build_info, re.IGNORECASE)
        if match:
            result[key] = match.group(1).upper() == "YES"
        else:
            result[key] = False

    # Extract CUDA version if present
    cuda_ver_match = re.search(r"NVIDIA CUDA:\s+YES\s+\(ver\s+([\d.]+)", build_info)
    if cuda_ver_match:
        result["cuda_version"] = cuda_ver_match.group(1)

    # Extract cuDNN version if present
    cudnn_ver_match = re.search(r"cuDNN:\s+YES\s+\(ver\s+([\d.]+)", build_info)
    if cudnn_ver_match:
        result["cudnn_version"] = cudnn_ver_match.group(1)

    return result


# -- Probe ------------------------------------------------------------------


class OpenCVBuildProbe(ProbeBase):
    """Probe that extracts OpenCV build configuration.

    This probe imports ``cv2`` and parses the build information string
    to determine which backends and accelerators are available.  It is
    read-only and does not modify any system state.
    """

    name = "opencv_build"

    def is_applicable(self) -> bool:
        """Only run when ``cv2`` is importable."""
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    def collect(self) -> Dict[str, Any]:
        """Parse ``cv2.getBuildInformation()`` and return feature flags.

        The returned dict does not map directly to ``SystemProfile``
        fields; instead it provides supplementary data that framework
        checkers and rules can access.
        """
        try:
            import cv2
            build_info = cv2.getBuildInformation()
        except Exception:
            logger.debug("Failed to call cv2.getBuildInformation()")
            return {}

        parsed = _parse_build_information(build_info)
        logger.debug("OpenCV build flags: %s", parsed)
        return parsed


# -- Rules ------------------------------------------------------------------


class OpenCVNoCUDARule(Rule):
    """Report when OpenCV lacks CUDA support on a GPU-equipped board.

    On systems with a GPU, having an OpenCV build without CUDA support
    means GPU-accelerated image processing is unavailable.  This is
    informational -- it may be intentional.
    """

    rule_id = "OPENCV_NO_CUDA"
    description = (
        "Check whether OpenCV was built with CUDA support on a "
        "GPU-equipped system."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applies when a GPU is available."""
        return profile.gpu_available

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Check OpenCV's CUDA build flag."""
        findings: List[Finding] = []

        build_flags = self._get_opencv_build_flags()
        if build_flags is None:
            # OpenCV not installed or not importable -- nothing to check.
            return findings

        has_cuda = build_flags.get("cuda", False)
        if not has_cuda:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.INFO,
                title="OpenCV built without CUDA support",
                explanation=(
                    "This system has a GPU, but the installed OpenCV was "
                    "built without CUDA support.  GPU-accelerated image "
                    "processing via cv2.cuda will not be available."
                ),
                suggestion=(
                    "Install opencv-contrib-python or build OpenCV from "
                    "source with -D WITH_CUDA=ON to enable GPU acceleration."
                ),
                package="opencv-python",
                confidence=0.9,
            ))
        else:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.PASSED,
                title="OpenCV has CUDA support",
                explanation="OpenCV was built with CUDA support enabled.",
                package="opencv-python",
            ))

        return findings

    @staticmethod
    def _get_opencv_build_flags() -> Optional[Dict[str, Any]]:
        """Import cv2 and parse build flags, returning None on failure."""
        try:
            import cv2
            return _parse_build_information(cv2.getBuildInformation())
        except ImportError:
            return None
        except Exception:
            logger.debug("Failed to parse OpenCV build information", exc_info=True)
            return None


class OpenCVNoGStreamerRule(Rule):
    """Report when OpenCV lacks GStreamer support on an embedded board.

    On embedded systems (Jetson, Raspberry Pi, etc.), GStreamer is the
    primary multimedia pipeline.  An OpenCV build without GStreamer
    support limits video capture and processing capabilities.
    """

    rule_id = "OPENCV_NO_GSTREAMER"
    description = (
        "Check whether OpenCV was built with GStreamer support on an "
        "embedded board."
    )

    def is_applicable(self, profile: SystemProfile) -> bool:
        """Only applies on embedded boards."""
        return profile.is_embedded_board

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Check OpenCV's GStreamer build flag."""
        findings: List[Finding] = []

        build_flags = self._get_opencv_build_flags()
        if build_flags is None:
            return findings

        has_gstreamer = build_flags.get("gstreamer", False)
        if not has_gstreamer:
            board_desc = profile.board_name or "embedded board"
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                title="OpenCV built without GStreamer support",
                explanation=(
                    f"This {board_desc} relies on GStreamer for hardware-"
                    f"accelerated video pipelines, but the installed OpenCV "
                    f"was built without GStreamer support.  Video capture "
                    f"with cv2.VideoCapture using GStreamer pipelines will fail."
                ),
                suggestion=(
                    "Build OpenCV from source with -D WITH_GSTREAMER=ON, or "
                    "install a distribution-provided OpenCV package that "
                    "includes GStreamer support."
                ),
                package="opencv-python",
            ))
        else:
            findings.append(Finding(
                rule_id=self.rule_id,
                severity=Severity.PASSED,
                title="OpenCV has GStreamer support",
                explanation="OpenCV was built with GStreamer support enabled.",
                package="opencv-python",
            ))

        return findings

    @staticmethod
    def _get_opencv_build_flags() -> Optional[Dict[str, Any]]:
        """Import cv2 and parse build flags, returning None on failure."""
        try:
            import cv2
            return _parse_build_information(cv2.getBuildInformation())
        except ImportError:
            return None
        except Exception:
            logger.debug("Failed to parse OpenCV build information", exc_info=True)
            return None


# -- Plugin entry point -----------------------------------------------------


def register(registry: HookRegistry) -> None:
    """Register OpenCV extensions if cv2 is importable.

    This function is called by the plugin loader.  It only activates
    when ``cv2`` can be imported, indicating that OpenCV is installed.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        logger.debug("OpenCV plugin: cv2 not importable; not activating")
        return

    registry.add_probe(OpenCVBuildProbe())
    registry.add_rule(OpenCVNoCUDARule())
    registry.add_rule(OpenCVNoGStreamerRule())
    logger.info("OpenCV plugin activated")
