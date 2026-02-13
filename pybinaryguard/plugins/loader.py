"""Plugin discovery and loading via ``importlib.metadata`` entry points.

Plugins declare themselves through the ``pybinaryguard.plugins`` entry-point
group in their ``pyproject.toml`` (or ``setup.cfg`` / ``setup.py``).  Each
entry point must resolve to a module that exposes a ``register(registry)``
function.

Example ``pyproject.toml`` snippet::

    [project.entry-points."pybinaryguard.plugins"]
    my_plugin = "my_package.pybinaryguard_plugin"

The module ``my_package.pybinaryguard_plugin`` must define::

    def register(registry: HookRegistry) -> None:
        ...

This module also discovers the built-in contrib plugins that ship with
PyBinaryGuard itself (Jetson, OpenCV, TensorRT, GStreamer).
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import List

from pybinaryguard.plugins.hooks import HookRegistry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pybinaryguard.plugins"

# Built-in contrib plugin module paths, loaded after external entry points.
_BUILTIN_CONTRIB_MODULES: List[str] = [
    "pybinaryguard.plugins.contrib.jetson",
    "pybinaryguard.plugins.contrib.opencv",
    "pybinaryguard.plugins.contrib.tensorrt",
    "pybinaryguard.plugins.contrib.gstreamer",
]


def _load_entry_point_plugins(registry: HookRegistry) -> None:
    """Discover and load external plugins via ``importlib.metadata`` entry points.

    Handles both the Python 3.9 API (``entry_points()`` returns a dict)
    and the Python 3.10+ API (``entry_points(group=...)``).
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        # Fallback for very old environments that lack importlib.metadata
        # (should not happen on 3.9+ but guard defensively).
        logger.debug("importlib.metadata not available; skipping entry-point discovery")
        return

    eps: list = []
    try:
        # Python 3.12+ and 3.10+ style: keyword argument filtering
        selected = entry_points(group=ENTRY_POINT_GROUP)
        # On 3.10-3.11 this returns a SelectableGroups or list
        if isinstance(selected, dict):
            # 3.9 fallback: entry_points() returns a dict of lists
            eps = list(selected.get(ENTRY_POINT_GROUP, []))
        else:
            eps = list(selected)
    except TypeError:
        # Python 3.9: entry_points() does not accept keyword arguments.
        all_eps = entry_points()
        if isinstance(all_eps, dict):
            eps = list(all_eps.get(ENTRY_POINT_GROUP, []))
        else:
            # Should not reach here, but handle gracefully
            eps = []

    for ep in eps:
        try:
            plugin_module = ep.load()
            register_fn = getattr(plugin_module, "register", None)
            if register_fn is None:
                logger.warning(
                    "Plugin entry point %r resolved to %r but it has no "
                    "'register' function; skipping",
                    ep.name,
                    plugin_module,
                )
                continue
            if not callable(register_fn):
                logger.warning(
                    "Plugin %r has a 'register' attribute but it is not "
                    "callable; skipping",
                    ep.name,
                )
                continue
            register_fn(registry)
            logger.info("Loaded external plugin: %s", ep.name)
        except Exception:
            logger.warning(
                "Failed to load plugin entry point %r; skipping",
                ep.name,
                exc_info=True,
            )


def _load_builtin_contrib_plugins(registry: HookRegistry) -> None:
    """Load the contrib plugins that ship with PyBinaryGuard.

    Each contrib module is imported and its ``register(registry)`` function
    is called.  Errors in individual contrib plugins are logged and do not
    prevent the rest from loading.
    """
    for module_path in _BUILTIN_CONTRIB_MODULES:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.debug(
                "Could not import contrib plugin %s; skipping",
                module_path,
            )
            continue

        register_fn = getattr(mod, "register", None)
        if register_fn is None:
            logger.debug(
                "Contrib module %s has no 'register' function; skipping",
                module_path,
            )
            continue

        if not callable(register_fn):
            logger.debug(
                "Contrib module %s 'register' attribute is not callable; skipping",
                module_path,
            )
            continue

        try:
            register_fn(registry)
            logger.debug("Loaded contrib plugin: %s", module_path)
        except Exception:
            logger.warning(
                "Contrib plugin %s raised an error during registration; skipping",
                module_path,
                exc_info=True,
            )


def discover_plugins(
    *,
    load_contrib: bool = True,
    load_external: bool = True,
) -> HookRegistry:
    """Discover and load all available plugins.

    This is the main entry point for the plugin subsystem.  It creates a
    fresh :class:`HookRegistry`, populates it from external entry points
    and built-in contrib plugins, and returns it.

    Args:
        load_contrib: Whether to load the built-in contrib plugins that
            ship with PyBinaryGuard (Jetson, OpenCV, TensorRT, GStreamer).
            Defaults to ``True``.
        load_external: Whether to discover and load externally-installed
            plugins via ``importlib.metadata`` entry points.  Defaults
            to ``True``.

    Returns:
        A :class:`HookRegistry` populated with all successfully loaded
        plugin extensions.
    """
    registry = HookRegistry()

    if load_external:
        _load_entry_point_plugins(registry)

    if load_contrib:
        _load_builtin_contrib_plugins(registry)

    total = registry.summary()
    registered = sum(total.values())
    if registered:
        logger.info(
            "Plugin discovery complete: %s",
            ", ".join(f"{k}={v}" for k, v in total.items() if v),
        )
    else:
        logger.debug("Plugin discovery complete: no extensions registered")

    return registry
