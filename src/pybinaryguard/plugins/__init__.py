"""Plugin subsystem for PyBinaryGuard.

The plugin system allows third-party packages (and built-in contrib
modules) to extend PyBinaryGuard with additional probes, rules, board
detectors, framework checkers, reporters, and lifecycle hooks.

Public API
----------
.. autoclass:: HookRegistry
.. autofunction:: discover_plugins
"""

from pybinaryguard.plugins.hooks import HookRegistry
from pybinaryguard.plugins.loader import discover_plugins

__all__ = [
    "HookRegistry",
    "discover_plugins",
]
