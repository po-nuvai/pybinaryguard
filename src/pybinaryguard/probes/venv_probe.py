"""Virtual environment probe — detects venv type and configuration."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict


from .base import ProbeBase


class VenvProbe(ProbeBase):
    """Detect virtual environment type and configuration.

    Identifies whether running in venv, conda, virtualenv, poetry, pdm,
    pipenv, or system Python. Also detects common misconfigurations.
    """

    name = "venv"

    def collect(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        venv_type = self._detect_venv_type()
        data["venv_type"] = venv_type
        data["is_system_python"] = venv_type == "system"
        data["is_virtual_env"] = venv_type != "system"

        # Base prefix vs prefix (tells us if we're in a venv)
        data["base_prefix"] = getattr(sys, "base_prefix", sys.prefix)
        data["prefix"] = sys.prefix

        # Detect conda-specific info
        if venv_type == "conda":
            data["conda_env_name"] = os.environ.get("CONDA_DEFAULT_ENV", "")
            data["conda_prefix"] = os.environ.get("CONDA_PREFIX", "")

        # Check for common misconfigurations
        data["pip_user_site_enabled"] = self._pip_user_site_active()
        data["mixed_env_risk"] = self._detect_mixed_env()

        return data

    @staticmethod
    def _detect_venv_type() -> str:
        """Identify the virtual environment manager in use."""
        # Conda
        if os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX"):
            return "conda"

        # Poetry
        if os.environ.get("POETRY_ACTIVE") == "1":
            return "poetry"

        # Pipenv
        if os.environ.get("PIPENV_ACTIVE") == "1":
            return "pipenv"

        # PDM
        if os.environ.get("PDM_IN_VENV"):
            return "pdm"

        # Standard venv / virtualenv
        real_prefix = getattr(sys, "real_prefix", None)  # virtualenv sets this
        base_prefix = getattr(sys, "base_prefix", sys.prefix)

        if real_prefix is not None:
            return "virtualenv"

        if base_prefix != sys.prefix:
            # Check for pyvenv.cfg to distinguish venv from virtualenv
            cfg = os.path.join(sys.prefix, "pyvenv.cfg")
            if os.path.isfile(cfg):
                return "venv"
            return "virtualenv"

        return "system"

    @staticmethod
    def _pip_user_site_active() -> bool:
        """Check if pip user-site installs are leaking into the env."""
        try:
            import site
            user_site = getattr(site, "getusersitepackages", lambda: None)()
            if user_site and os.path.isdir(user_site):
                # Check if user site is in sys.path
                return user_site in sys.path
        except Exception:
            pass
        return False

    @staticmethod
    def _detect_mixed_env() -> bool:
        """Detect if packages from multiple environments are on sys.path.

        This happens when system packages leak into a venv or vice versa.
        """
        site_packages_dirs = [
            p for p in sys.path if "site-packages" in p or "dist-packages" in p
        ]
        # If there are multiple distinct site-packages roots, it's mixed
        prefixes = set()
        for sp in site_packages_dirs:
            # Get the env root (two levels up from site-packages)
            parts = sp.split(os.sep)
            for i, part in enumerate(parts):
                if part in ("site-packages", "dist-packages") and i >= 2:
                    prefixes.add(os.sep.join(parts[:i - 1]))
                    break
        return len(prefixes) > 1
