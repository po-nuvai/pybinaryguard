"""Probe for embedded / single-board computer detection."""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional

from .base import ProbeBase


class BoardProbe(ProbeBase):
    """Detects common embedded boards and single-board computers.

    Supported boards:
    - NVIDIA Jetson (Nano, TX2, Xavier, Orin)
    - Raspberry Pi (all models)
    - BeagleBone (Black, AI, etc.)
    - Google Coral (Dev Board)
    - Generic ARM SBCs (aarch64/armv7l without a desktop GPU)

    Detection relies on ``/proc/device-tree/model``,
    ``/etc/nv_tegra_release``, and device-node presence.
    """

    name = "board"

    def collect(self) -> Dict[str, Any]:
        """Return board detection results."""
        data: Dict[str, Any] = {}

        dt_model = self._read_device_tree_model()

        # Try each board family in order of specificity
        board = self._detect_jetson(dt_model)
        if board is not None:
            data["is_embedded_board"] = True
            data["board_name"] = board["name"]
            if board.get("jetpack_version"):
                data["jetpack_version"] = board["jetpack_version"]
            return data

        board_name = self._detect_raspberry_pi(dt_model)
        if board_name is not None:
            data["is_embedded_board"] = True
            data["board_name"] = board_name
            return data

        board_name = self._detect_beaglebone(dt_model)
        if board_name is not None:
            data["is_embedded_board"] = True
            data["board_name"] = board_name
            return data

        board_name = self._detect_coral(dt_model)
        if board_name is not None:
            data["is_embedded_board"] = True
            data["board_name"] = board_name
            return data

        board_name = self._detect_generic_arm_sbc(dt_model)
        if board_name is not None:
            data["is_embedded_board"] = True
            data["board_name"] = board_name
            return data

        data["is_embedded_board"] = False
        return data

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_device_tree_model() -> str:
        """Read ``/proc/device-tree/model``.

        The file is NUL-terminated on many kernels, so we strip trailing
        NUL bytes and whitespace.
        """
        try:
            with open("/proc/device-tree/model", "rb") as fh:
                raw = fh.read()
            return raw.decode("utf-8", errors="replace").strip("\x00").strip()
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    @staticmethod
    def _read_text_file(path: str) -> str:
        """Read a text file, returning empty string on failure."""
        try:
            with open(path, "r") as fh:
                return fh.read().strip()
        except (FileNotFoundError, PermissionError, OSError):
            return ""

    # ------------------------------------------------------------------
    # Jetson
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_jetson(dt_model: str) -> Optional[Dict[str, str]]:
        """Detect NVIDIA Jetson boards.

        Primary indicator: ``/etc/nv_tegra_release`` exists.
        Secondary: device-tree model contains "Jetson" or "NVIDIA".
        """
        tegra_release = BoardProbe._read_text_file("/etc/nv_tegra_release")
        is_jetson = bool(tegra_release)

        if not is_jetson:
            # Fallback: check device-tree model
            model_lower = dt_model.lower()
            if "jetson" in model_lower or ("nvidia" in model_lower and "tegra" in model_lower):
                is_jetson = True

        if not is_jetson:
            return None

        name = dt_model if dt_model else "NVIDIA Jetson (unknown model)"

        # Try to determine JetPack version from L4T version
        jetpack_version = BoardProbe._jetpack_from_l4t(tegra_release)

        result: Dict[str, str] = {"name": name}
        if jetpack_version:
            result["jetpack_version"] = jetpack_version

        return result

    @staticmethod
    def _jetpack_from_l4t(tegra_release: str) -> str:
        """Map L4T release string to JetPack version.

        ``/etc/nv_tegra_release`` contains a line like::

            # R35 (release), REVISION: 4.1, ...

        Known L4T-to-JetPack mappings (approximate):
        - R36.x -> JetPack 6.x
        - R35.x -> JetPack 5.x
        - R32.x -> JetPack 4.x
        """
        if not tegra_release:
            return ""

        import re

        match = re.search(r"R(\d+)\s.*?REVISION:\s*(\d+(?:\.\d+)?)", tegra_release)
        if not match:
            return ""

        major = int(match.group(1))
        revision = match.group(2)
        l4t_tag = f"L4T R{major} rev {revision}"

        # Coarse mapping
        if major >= 36:
            return f"6.x ({l4t_tag})"
        elif major >= 35:
            return f"5.x ({l4t_tag})"
        elif major >= 32:
            return f"4.x ({l4t_tag})"
        else:
            return l4t_tag

    # ------------------------------------------------------------------
    # Raspberry Pi
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_raspberry_pi(dt_model: str) -> Optional[str]:
        """Detect Raspberry Pi boards via the device-tree model string."""
        if "raspberry pi" in dt_model.lower():
            return dt_model
        return None

    # ------------------------------------------------------------------
    # BeagleBone
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_beaglebone(dt_model: str) -> Optional[str]:
        """Detect BeagleBone boards via the device-tree model string."""
        if "beaglebone" in dt_model.lower() or "beagle bone" in dt_model.lower():
            return dt_model
        return None

    # ------------------------------------------------------------------
    # Google Coral
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_coral(dt_model: str) -> Optional[str]:
        """Detect Google Coral dev board.

        Checks the device-tree model and also looks for the Edge TPU
        USB device node at ``/dev/apex_0``.
        """
        model_lower = dt_model.lower()
        if "coral" in model_lower:
            return dt_model

        # Some Coral boards identify as "Freescale" in the DT model;
        # check for the TPU device node as a secondary signal
        if os.path.exists("/dev/apex_0"):
            if dt_model:
                return f"{dt_model} (Coral TPU detected)"
            return "Google Coral (TPU detected)"

        return None

    # ------------------------------------------------------------------
    # Generic ARM SBC
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_generic_arm_sbc(dt_model: str) -> Optional[str]:
        """Detect a generic ARM single-board computer.

        Criteria:
        - Architecture is aarch64 or armv7l.
        - No desktop GPU is detected (no ``/dev/nvidia*`` or ``/dev/dri/card*``).
        - A device-tree model string is available (bare-metal servers
          typically do not populate ``/proc/device-tree/model``).
        """
        machine = ""
        try:
            machine = platform.machine()
        except Exception:
            pass

        if machine not in ("aarch64", "arm64", "armv7l", "armv6l"):
            return None

        # Must have a device-tree model to differentiate from cloud ARM servers
        if not dt_model:
            return None

        # Exclude systems with a desktop GPU
        has_desktop_gpu = (
            os.path.exists("/dev/nvidia0")
            or os.path.exists("/dev/dri/card0")
        )
        if has_desktop_gpu:
            return None

        return dt_model
