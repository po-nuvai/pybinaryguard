"""Hook registry for the plugin subsystem.

Plugins register their extensions (probes, rules, detectors, reporters,
and lifecycle hooks) with a :class:`HookRegistry` instance that is passed
to their ``register()`` function during discovery.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pybinaryguard.probes.base import ProbeBase
    from pybinaryguard.rules.base import Rule
    from pybinaryguard.models.system import SystemProfile
    from pybinaryguard.models.package import PackageBinaryInfo
    from pybinaryguard.models.finding import Finding

logger = logging.getLogger(__name__)

# Type aliases for hook signatures.
BoardDetector = Callable[[Dict[str, Any]], Optional[str]]
FrameworkChecker = Callable[["SystemProfile", List["PackageBinaryInfo"]], List["Finding"]]
Reporter = Callable[[List["Finding"], "SystemProfile"], str]
ScanHook = Callable[["SystemProfile"], None]


class HookRegistry:
    """Central registry that collects all plugin-provided extensions.

    An instance is created by the plugin loader and handed to each plugin's
    ``register()`` entry point.  Plugins call the ``add_*`` methods to
    register their contributions, and the host application reads them back
    via the corresponding read-only properties.

    Example (inside a plugin's ``register`` function)::

        def register(registry: HookRegistry) -> None:
            registry.add_probe(MyProbe())
            registry.add_rule(MyRule())
            registry.pre_scan(lambda profile: print("scan starting"))
    """

    def __init__(self) -> None:
        self._probes: List[ProbeBase] = []
        self._rules: List[Rule] = []
        self._board_detectors: List[BoardDetector] = []
        self._framework_checkers: List[FrameworkChecker] = []
        self._reporters: List[Reporter] = []
        self._pre_scan_hooks: List[ScanHook] = []
        self._post_scan_hooks: List[ScanHook] = []

    # ------------------------------------------------------------------
    # Registration methods
    # ------------------------------------------------------------------

    def add_probe(self, probe: ProbeBase) -> None:
        """Register an additional system probe.

        Args:
            probe: A :class:`~pybinaryguard.probes.base.ProbeBase` subclass
                instance.  Must have a unique ``name`` attribute.

        Raises:
            TypeError: If *probe* is not a :class:`ProbeBase` instance.
            ValueError: If a probe with the same ``name`` is already registered.
        """
        from pybinaryguard.probes.base import ProbeBase as _ProbeBase

        if not isinstance(probe, _ProbeBase):
            raise TypeError(
                f"Expected a ProbeBase instance, got {type(probe).__name__}"
            )
        existing_names = {p.name for p in self._probes}
        if probe.name in existing_names:
            raise ValueError(
                f"A probe named {probe.name!r} is already registered"
            )
        self._probes.append(probe)
        logger.debug("Registered plugin probe: %s", probe.name)

    def add_rule(self, rule: Rule) -> None:
        """Register an additional compatibility rule.

        Args:
            rule: A :class:`~pybinaryguard.rules.base.Rule` subclass instance.
                Must have a unique ``rule_id`` attribute.

        Raises:
            TypeError: If *rule* is not a :class:`Rule` instance.
            ValueError: If a rule with the same ``rule_id`` is already registered.
        """
        from pybinaryguard.rules.base import Rule as _Rule

        if not isinstance(rule, _Rule):
            raise TypeError(
                f"Expected a Rule instance, got {type(rule).__name__}"
            )
        existing_ids = {r.rule_id for r in self._rules}
        if rule.rule_id in existing_ids:
            raise ValueError(
                f"A rule with id {rule.rule_id!r} is already registered"
            )
        self._rules.append(rule)
        logger.debug("Registered plugin rule: %s", rule.rule_id)

    def add_board_detector(self, detector: BoardDetector) -> None:
        """Register a board-detection callback.

        A board detector is a callable that receives a dict of raw probe
        data and returns either a board name string (e.g. ``"Jetson Orin
        Nano"``) or ``None`` if the board was not recognised.

        Args:
            detector: The detection callable.

        Raises:
            TypeError: If *detector* is not callable.
        """
        if not callable(detector):
            raise TypeError(
                f"Expected a callable, got {type(detector).__name__}"
            )
        self._board_detectors.append(detector)
        logger.debug("Registered plugin board detector: %s", detector)

    def add_framework_checker(self, checker: FrameworkChecker) -> None:
        """Register a framework-specific compatibility checker.

        A framework checker receives the system profile and packages list
        and returns additional findings for framework-specific issues
        (e.g. OpenCV build flags, TensorRT version compatibility).

        Args:
            checker: The checker callable.

        Raises:
            TypeError: If *checker* is not callable.
        """
        if not callable(checker):
            raise TypeError(
                f"Expected a callable, got {type(checker).__name__}"
            )
        self._framework_checkers.append(checker)
        logger.debug("Registered plugin framework checker: %s", checker)

    def add_reporter(self, reporter: Reporter) -> None:
        """Register a custom report formatter.

        A reporter is a callable that takes a list of findings and the
        system profile and returns a formatted report string.

        Args:
            reporter: The reporter callable.

        Raises:
            TypeError: If *reporter* is not callable.
        """
        if not callable(reporter):
            raise TypeError(
                f"Expected a callable, got {type(reporter).__name__}"
            )
        self._reporters.append(reporter)
        logger.debug("Registered plugin reporter: %s", reporter)

    def pre_scan(self, hook: ScanHook) -> None:
        """Register a hook that runs before scanning begins.

        The hook receives the :class:`SystemProfile` and may perform
        logging, telemetry, or side-channel checks.  It must not modify
        system state.

        Args:
            hook: The pre-scan callable.

        Raises:
            TypeError: If *hook* is not callable.
        """
        if not callable(hook):
            raise TypeError(
                f"Expected a callable, got {type(hook).__name__}"
            )
        self._pre_scan_hooks.append(hook)
        logger.debug("Registered plugin pre-scan hook: %s", hook)

    def post_scan(self, hook: ScanHook) -> None:
        """Register a hook that runs after scanning completes.

        The hook receives the :class:`SystemProfile`.  It must not modify
        system state.

        Args:
            hook: The post-scan callable.

        Raises:
            TypeError: If *hook* is not callable.
        """
        if not callable(hook):
            raise TypeError(
                f"Expected a callable, got {type(hook).__name__}"
            )
        self._post_scan_hooks.append(hook)
        logger.debug("Registered plugin post-scan hook: %s", hook)

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def probes(self) -> List[ProbeBase]:
        """All registered plugin probes (defensive copy)."""
        return list(self._probes)

    @property
    def rules(self) -> List[Rule]:
        """All registered plugin rules (defensive copy)."""
        return list(self._rules)

    @property
    def board_detectors(self) -> List[BoardDetector]:
        """All registered board-detection callbacks (defensive copy)."""
        return list(self._board_detectors)

    @property
    def framework_checkers(self) -> List[FrameworkChecker]:
        """All registered framework checkers (defensive copy)."""
        return list(self._framework_checkers)

    @property
    def reporters(self) -> List[Reporter]:
        """All registered report formatters (defensive copy)."""
        return list(self._reporters)

    @property
    def pre_scan_hooks(self) -> List[ScanHook]:
        """All registered pre-scan hooks (defensive copy)."""
        return list(self._pre_scan_hooks)

    @property
    def post_scan_hooks(self) -> List[ScanHook]:
        """All registered post-scan hooks (defensive copy)."""
        return list(self._post_scan_hooks)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def probe_count(self) -> int:
        """Number of registered probes."""
        return len(self._probes)

    @property
    def rule_count(self) -> int:
        """Number of registered rules."""
        return len(self._rules)

    def summary(self) -> Dict[str, int]:
        """Return a dict summarising the number of registered extensions."""
        return {
            "probes": len(self._probes),
            "rules": len(self._rules),
            "board_detectors": len(self._board_detectors),
            "framework_checkers": len(self._framework_checkers),
            "reporters": len(self._reporters),
            "pre_scan_hooks": len(self._pre_scan_hooks),
            "post_scan_hooks": len(self._post_scan_hooks),
        }

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.summary().items() if v)
        return f"<HookRegistry({parts})>"
