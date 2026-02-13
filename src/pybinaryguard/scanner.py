"""Scanner orchestrator -- runs the full probe -> analyze -> evaluate -> report pipeline.

This module is the heart of PyBinaryGuard.  The :class:`Scanner` class ties
together probes, analyzers, rules, and plugins into a single cohesive scan
pipeline.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from pybinaryguard.models import (
    Finding,
    PackageBinaryInfo,
    ScanMode,
    ScanReport,
    Severity,
    SystemProfile,
    WheelTag,
)

logger = logging.getLogger(__name__)


class Scanner:
    """Orchestrates the full scan pipeline: probe -> analyze -> evaluate -> report.

    Parameters
    ----------
    probes:
        Override the default list of probes.  When ``None``, all built-in
        probes are used.
    packages:
        When set, restrict analysis to only these package names (case-
        insensitive).  ``None`` means scan every installed package.
    severity_threshold:
        Minimum severity to include in the final report.  Findings below
        this threshold are filtered out.
    ignored_rules:
        Set of rule IDs to skip during evaluation.
    timeout:
        Maximum wall-clock seconds for each probe's ``collect()`` call.
    enable_plugins:
        Whether to discover and load plugins via entry points and the
        built-in contrib modules.
    """

    def __init__(
        self,
        probes: Optional[List[Any]] = None,
        packages: Optional[List[str]] = None,
        severity_threshold: Severity = Severity.INFO,
        ignored_rules: Optional[Set[str]] = None,
        timeout: float = 30.0,
        enable_plugins: bool = True,
        scan_mode: ScanMode = ScanMode.STANDARD,
    ) -> None:
        self._custom_probes = probes
        self._packages = (
            {p.lower() for p in packages} if packages else None
        )
        self._severity_threshold = severity_threshold
        self._ignored_rules: Set[str] = ignored_rules or set()
        self._timeout = timeout
        self._enable_plugins = enable_plugins
        self._scan_mode = scan_mode

        # Lazily populated
        self._profile: Optional[SystemProfile] = None
        self._plugin_registry: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ScanReport:
        """Execute the full scan pipeline.

        1. **PROBE** -- run all probes to build a :class:`SystemProfile`.
        2. **ANALYZE** -- walk site-packages, discover packages, run ELF
           analysis on shared objects.
        3. **EVALUATE** -- apply compatibility rules and plugin framework
           checkers to produce findings.
        4. **REPORT** -- filter findings by severity, sort, and assemble
           the :class:`ScanReport`.

        Returns
        -------
        ScanReport
            A complete scan report with all findings, package counts, and
            timing metadata.
        """
        start = time.monotonic()

        try:
            # 1. PROBE PHASE
            profile = self._run_probes()
            self._profile = profile

            # Fire pre-scan hooks
            self._fire_pre_scan_hooks(profile)

            # 2. ANALYZE PHASE
            all_packages = self._discover_and_analyze_packages(profile)

            # 3. EVALUATE PHASE
            findings = self._evaluate_rules(profile, all_packages)

            # 4. REPORT PHASE
            report = self._build_report(
                findings=findings,
                packages_scanned=len(all_packages),
                total_packages=self._count_total_packages(profile),
                elapsed_ms=(time.monotonic() - start) * 1000.0,
            )

            # Fire post-scan hooks
            self._fire_post_scan_hooks(profile)

            return report
        except Exception as exc:
            logger.error("Scan failed: %s", exc, exc_info=True)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            return ScanReport(
                findings=[
                    Finding(
                        rule_id="SCANNER_ERROR",
                        severity=Severity.CRITICAL,
                        title="Scanner encountered an error",
                        explanation=str(exc),
                    )
                ],
                scan_duration_ms=elapsed_ms,
            )

    def check_package(self, package_name: str) -> List[Finding]:
        """Check a single package and return its findings.

        Parameters
        ----------
        package_name:
            The distribution name of the package to check.

        Returns
        -------
        List[Finding]
            All findings for the specified package, sorted by severity.
        """
        profile = self._run_probes()
        self._profile = profile

        pkg_info = self._find_and_analyze_package(profile, package_name)
        if pkg_info is None:
            return [
                Finding(
                    rule_id="PACKAGE_NOT_FOUND",
                    severity=Severity.INFO,
                    title=f"Package '{package_name}' not found",
                    explanation=(
                        f"Could not find '{package_name}' in any "
                        f"site-packages directory."
                    ),
                    package=package_name,
                )
            ]

        findings = self._evaluate_rules(profile, [pkg_info])
        return findings

    def get_profile(self) -> SystemProfile:
        """Collect the system profile without running analysis or rules.

        Returns
        -------
        SystemProfile
            The current system's binary compatibility surface.
        """
        profile = self._run_probes()
        self._profile = profile
        return profile

    def inspect_file(self, file_path: str) -> List[Finding]:
        """Inspect a ``.whl`` or ``.so`` file against the current system.

        Parameters
        ----------
        file_path:
            Absolute or relative path to a ``.whl`` or ``.so`` file.

        Returns
        -------
        List[Finding]
            Findings produced by inspecting the file.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If *file_path* is not a ``.whl`` or ``.so`` file.
        """
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        profile = self._run_probes()
        self._profile = profile

        if abs_path.endswith(".whl"):
            return self._inspect_wheel(abs_path, profile)
        elif abs_path.endswith(".so") or ".so." in os.path.basename(abs_path):
            return self._inspect_shared_object(abs_path, profile)
        else:
            raise ValueError(
                f"Unsupported file type: {abs_path!r}. "
                f"Expected a .whl or .so file."
            )

    # ------------------------------------------------------------------
    # PROBE PHASE
    # ------------------------------------------------------------------

    def _get_all_probes(self) -> list:
        """Return the list of probes to execute."""
        if self._custom_probes is not None:
            return list(self._custom_probes)

        from pybinaryguard.probes.board_probe import BoardProbe
        from pybinaryguard.probes.cpu_probe import CpuProbe
        from pybinaryguard.probes.glibc_probe import GlibcProbe
        from pybinaryguard.probes.library_probe import LibraryProbe
        from pybinaryguard.probes.os_probe import OsProbe
        from pybinaryguard.probes.python_probe import PythonProbe

        probes: list = [
            PythonProbe(),
            OsProbe(),
            CpuProbe(),
            GlibcProbe(),
            LibraryProbe(),
            BoardProbe(),
        ]

        # Add plugin probes
        registry = self._get_plugin_registry()
        if registry is not None:
            probes.extend(registry.probes)

        return probes

    def _run_probes(self) -> SystemProfile:
        """Execute all probes in parallel and merge results into a SystemProfile."""
        if self._profile is not None:
            return self._profile

        probes = self._get_all_probes()
        merged: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=min(len(probes), 8)) as executor:
            future_to_probe = {}
            for probe in probes:
                if not probe.is_applicable():
                    logger.debug("Skipping probe %s (not applicable)", probe.name)
                    continue
                future = executor.submit(self._safe_collect, probe)
                future_to_probe[future] = probe

            for future in as_completed(future_to_probe):
                probe = future_to_probe[future]
                try:
                    result = future.result(timeout=self._timeout)
                    if result:
                        merged.update(result)
                except Exception as exc:
                    logger.debug(
                        "Probe %s failed: %s", probe.name, exc
                    )

        return self._build_profile(merged)

    @staticmethod
    def _safe_collect(probe: Any) -> Dict[str, Any]:
        """Run a probe's collect() method, catching all exceptions."""
        try:
            return probe.collect()
        except Exception as exc:
            logger.debug("Probe %s raised: %s", probe.name, exc)
            return {}

    @staticmethod
    def _build_profile(data: Dict[str, Any]) -> SystemProfile:
        """Construct a SystemProfile from the merged probe data.

        Only fields that are valid SystemProfile attributes are passed through.
        Unknown keys are silently ignored.
        """
        valid_fields = {f.name for f in dataclasses.fields(SystemProfile)}
        filtered = {}
        for key, value in data.items():
            if key in valid_fields and value is not None:
                filtered[key] = value
        return SystemProfile(**filtered)

    # ------------------------------------------------------------------
    # ANALYZE PHASE
    # ------------------------------------------------------------------

    def _discover_and_analyze_packages(
        self, profile: SystemProfile
    ) -> List[PackageBinaryInfo]:
        """Walk site-packages directories and analyze each package's binaries.

        The depth of analysis depends on ``self._scan_mode``:

        - **FAST**: Only read WHEEL metadata and tags, skip ELF parsing.
        - **STANDARD**: Parse ELF headers for arch, GLIBC, DT_NEEDED.
        - **DEEP**: Full ELF analysis plus SHA256 hashing of all .so files.
        """
        packages: List[PackageBinaryInfo] = []
        seen_packages: Set[str] = set()

        # Only import the ELF analyzer for STANDARD and DEEP modes
        analyzer = None
        if self._scan_mode != ScanMode.FAST:
            from pybinaryguard.analyzers.elf_analyzer import ELFAnalyzer
            analyzer = ELFAnalyzer()

        for sp_path in profile.site_packages_paths:
            if not os.path.isdir(sp_path):
                continue

            try:
                entries = os.listdir(sp_path)
            except OSError as exc:
                logger.debug("Cannot list %s: %s", sp_path, exc)
                continue

            for entry in entries:
                if not entry.endswith(".dist-info"):
                    continue

                dist_info_path = os.path.join(sp_path, entry)
                if not os.path.isdir(dist_info_path):
                    continue

                pkg_name, pkg_version = self._parse_dist_info_name(entry)
                if not pkg_name:
                    continue

                # Deduplicate
                pkg_key = pkg_name.lower()
                if pkg_key in seen_packages:
                    continue
                seen_packages.add(pkg_key)

                # Filter if requested
                if self._packages is not None and pkg_key not in self._packages:
                    continue

                # Determine the package's install path
                install_path = self._find_package_install_path(
                    sp_path, dist_info_path, pkg_name
                )

                pkg_info = PackageBinaryInfo(
                    package_name=pkg_name,
                    package_version=pkg_version,
                    install_path=install_path,
                )

                # Parse WHEEL file for tags (all modes)
                wheel_tags = self._parse_wheel_file(dist_info_path)
                if wheel_tags:
                    pkg_info.wheel_tags = wheel_tags

                # FAST mode: detect .so presence without full parsing
                if self._scan_mode == ScanMode.FAST:
                    has_so = self._has_shared_objects(install_path)
                    if has_so:
                        pkg_info.is_pure_python = False
                    elif self._packages is None:
                        continue
                else:
                    # STANDARD and DEEP: run ELF analysis
                    pkg_info = analyzer.analyze(pkg_info)

                    # Skip pure-Python packages unless explicitly requested
                    if not pkg_info.has_binaries and self._packages is None:
                        continue

                    # DEEP mode: compute SHA256 hashes for all .so files
                    if self._scan_mode == ScanMode.DEEP and pkg_info.has_binaries:
                        self._compute_binary_hashes(pkg_info)

                packages.append(pkg_info)

        return packages

    @staticmethod
    def _has_shared_objects(install_path: str) -> bool:
        """Quick check for .so files without parsing them (FAST mode)."""
        if not os.path.isdir(install_path):
            return False
        try:
            for root, _dirs, files in os.walk(install_path):
                for fname in files:
                    if fname.endswith(".so") or ".so." in fname:
                        return True
        except OSError:
            pass
        return False

    @staticmethod
    def _compute_binary_hashes(pkg_info: PackageBinaryInfo) -> None:
        """Compute SHA256 hashes for all shared objects (DEEP mode)."""
        import hashlib

        for so in pkg_info.shared_objects:
            if so.path and os.path.isfile(so.path):
                sha256 = hashlib.sha256()
                try:
                    with open(so.path, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            sha256.update(chunk)
                    so.sha256 = sha256.hexdigest()
                except OSError:
                    pass

    def _find_and_analyze_package(
        self, profile: SystemProfile, package_name: str
    ) -> Optional[PackageBinaryInfo]:
        """Find and analyze a single named package."""
        from pybinaryguard.analyzers.elf_analyzer import ELFAnalyzer

        target = package_name.lower().replace("-", "_")
        analyzer = ELFAnalyzer()

        for sp_path in profile.site_packages_paths:
            if not os.path.isdir(sp_path):
                continue

            try:
                entries = os.listdir(sp_path)
            except OSError:
                continue

            for entry in entries:
                if not entry.endswith(".dist-info"):
                    continue

                pkg_name, pkg_version = self._parse_dist_info_name(entry)
                if not pkg_name:
                    continue

                if pkg_name.lower().replace("-", "_") != target:
                    continue

                dist_info_path = os.path.join(sp_path, entry)
                install_path = self._find_package_install_path(
                    sp_path, dist_info_path, pkg_name
                )

                pkg_info = PackageBinaryInfo(
                    package_name=pkg_name,
                    package_version=pkg_version,
                    install_path=install_path,
                )

                wheel_tags = self._parse_wheel_file(dist_info_path)
                if wheel_tags:
                    pkg_info.wheel_tags = wheel_tags

                pkg_info = analyzer.analyze(pkg_info)
                return pkg_info

        return None

    @staticmethod
    def _parse_dist_info_name(dirname: str) -> Tuple[str, str]:
        """Parse a .dist-info directory name into (name, version).

        Examples:
            ``"numpy-1.26.4.dist-info"`` -> ``("numpy", "1.26.4")``
            ``"Pillow-10.2.0.dist-info"`` -> ``("Pillow", "10.2.0")``

        Returns ``("", "")`` if parsing fails.
        """
        suffix = ".dist-info"
        if not dirname.endswith(suffix):
            return ("", "")
        base = dirname[: -len(suffix)]
        parts = base.split("-", 1)
        if len(parts) != 2:
            return ("", "")
        return (parts[0], parts[1])

    @staticmethod
    def _find_package_install_path(
        site_packages: str, dist_info_path: str, package_name: str
    ) -> str:
        """Determine the install path for a package.

        Tries the RECORD file to find actual installed files, then falls back
        to heuristic name matching.
        """
        # Normalise: packages use underscores in directory names
        normalised = package_name.lower().replace("-", "_")

        # Check RECORD file for top-level package directory
        record_path = os.path.join(dist_info_path, "RECORD")
        if os.path.isfile(record_path):
            try:
                with open(record_path, "r") as fh:
                    for line in fh:
                        parts = line.strip().split(",", 1)
                        if not parts[0]:
                            continue
                        first_component = parts[0].split("/")[0]
                        candidate = os.path.join(site_packages, first_component)
                        if (
                            os.path.isdir(candidate)
                            and first_component.lower().replace("-", "_") == normalised
                        ):
                            return candidate
            except OSError:
                pass

        # Heuristic: look for a directory matching the package name
        candidate = os.path.join(site_packages, normalised)
        if os.path.isdir(candidate):
            return candidate

        # Try the original casing
        candidate = os.path.join(site_packages, package_name)
        if os.path.isdir(candidate):
            return candidate

        # Fall back to the site-packages root itself
        return site_packages

    @staticmethod
    def _parse_wheel_file(dist_info_path: str) -> List[WheelTag]:
        """Parse the WHEEL metadata file for compatibility tags."""
        wheel_path = os.path.join(dist_info_path, "WHEEL")
        if not os.path.isfile(wheel_path):
            return []

        tags: List[WheelTag] = []
        try:
            with open(wheel_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("Tag:"):
                        tag_str = line[4:].strip()
                        parts = tag_str.split("-")
                        if len(parts) == 3:
                            tags.append(WheelTag(
                                interpreter=parts[0],
                                abi=parts[1],
                                platform=parts[2],
                            ))
        except OSError:
            pass

        return tags

    def _count_total_packages(self, profile: SystemProfile) -> int:
        """Count total .dist-info directories across all site-packages."""
        count = 0
        seen: Set[str] = set()
        for sp_path in profile.site_packages_paths:
            if not os.path.isdir(sp_path):
                continue
            try:
                for entry in os.listdir(sp_path):
                    if entry.endswith(".dist-info"):
                        pkg_name, _ = self._parse_dist_info_name(entry)
                        if pkg_name:
                            key = pkg_name.lower()
                            if key not in seen:
                                seen.add(key)
                                count += 1
            except OSError:
                continue
        return count

    # ------------------------------------------------------------------
    # EVALUATE PHASE
    # ------------------------------------------------------------------

    # Rule IDs that require ELF analysis (skip in FAST mode)
    _ELF_DEPENDENT_RULES: FrozenSet[str] = frozenset({
        "GLIBC_VERSION_MISMATCH",
        "GLIBC_SYMBOL_MISSING",
        "MUSL_GLIBC_CONFLICT",
        "MANYLINUX_TAG_VIOLATION",
        "MISSING_SHARED_LIB",
        "LIBSTDCXX_TOO_OLD",
        "NUMPY_ABI_MISMATCH",
        "ILLEGAL_INSTRUCTION_RISK",
        "AVX2_REQUIRED",
        "AVX512_REQUIRED",
        "PREDICTED_IMPORT_ERROR",
        "UNRESOLVED_DEPENDENCY_CHAIN",
    })

    def _evaluate_rules(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        """Run all rules (built-in + plugin) and return findings.

        In FAST mode, rules that depend on ELF analysis data are skipped.
        """
        from pybinaryguard.rules.engine import RuleEngine

        # In FAST mode, add ELF-dependent rules to the ignore list
        ignored = set(self._ignored_rules)
        if self._scan_mode == ScanMode.FAST:
            ignored.update(self._ELF_DEPENDENT_RULES)

        engine = RuleEngine.with_builtin_rules(ignored_rules=ignored)

        # Add plugin rules
        registry = self._get_plugin_registry()
        if registry is not None:
            for rule in registry.rules:
                engine.register(rule)

        findings = engine.evaluate(profile, packages)

        # Run plugin framework checkers (skip in FAST mode)
        if self._scan_mode != ScanMode.FAST and registry is not None:
            for checker in registry.framework_checkers:
                try:
                    extra = checker(profile, packages)
                    if extra:
                        findings.extend(extra)
                except Exception as exc:
                    logger.debug("Framework checker failed: %s", exc)

        return findings

    # ------------------------------------------------------------------
    # REPORT PHASE
    # ------------------------------------------------------------------

    def _build_report(
        self,
        findings: List[Finding],
        packages_scanned: int,
        total_packages: int,
        elapsed_ms: float,
    ) -> ScanReport:
        """Filter, sort, and assemble the final ScanReport."""
        from pybinaryguard.diagnostics.findings import (
            deduplicate_findings,
            filter_findings,
            sort_findings,
        )
        from pybinaryguard.profiles import match_board_profile
        from pybinaryguard.scoring import compute_health_score

        # Deduplicate
        findings = deduplicate_findings(findings)

        # Compute v2 health score (before filtering — uses all findings)
        has_gpu = self._profile.gpu_available if self._profile else False
        is_embedded = self._profile.is_embedded_board if self._profile else False
        score_breakdown = compute_health_score(
            findings, has_gpu=has_gpu, is_embedded=is_embedded
        )

        # Filter by severity threshold
        findings = filter_findings(findings, self._severity_threshold)

        # Sort: CRITICAL first
        findings = sort_findings(findings)

        # Detect board profile if available
        detected_board = None
        if self._profile:
            board_profile = match_board_profile(self._profile)
            if board_profile:
                detected_board = board_profile.display_name

        return ScanReport(
            findings=findings,
            packages_scanned=packages_scanned,
            total_packages=total_packages,
            scan_duration_ms=elapsed_ms,
            detected_board=detected_board,
            score_breakdown=score_breakdown,
        )

    # ------------------------------------------------------------------
    # INSPECT helpers
    # ------------------------------------------------------------------

    def _inspect_wheel(
        self, wheel_path: str, profile: SystemProfile
    ) -> List[Finding]:
        """Inspect a .whl file by extracting and analyzing its contents."""
        import tempfile
        import zipfile

        findings: List[Finding] = []

        if not zipfile.is_zipfile(wheel_path):
            findings.append(Finding(
                rule_id="INVALID_WHEEL",
                severity=Severity.CRITICAL,
                title="Invalid wheel file",
                explanation=f"The file '{wheel_path}' is not a valid ZIP/wheel archive.",
            ))
            return findings

        with tempfile.TemporaryDirectory(prefix="pybinaryguard_") as tmpdir:
            try:
                with zipfile.ZipFile(wheel_path, "r") as zf:
                    zf.extractall(tmpdir)
            except zipfile.BadZipFile as exc:
                findings.append(Finding(
                    rule_id="INVALID_WHEEL",
                    severity=Severity.CRITICAL,
                    title="Corrupt wheel file",
                    explanation=f"Cannot extract '{wheel_path}': {exc}",
                ))
                return findings

            # Parse wheel filename for package metadata
            basename = os.path.basename(wheel_path)
            name, version = self._parse_wheel_filename(basename)

            pkg_info = PackageBinaryInfo(
                package_name=name or basename,
                package_version=version or "unknown",
                install_path=tmpdir,
            )

            # Parse wheel tags from filename
            wheel_tags = self._tags_from_wheel_filename(basename)
            if wheel_tags:
                pkg_info.wheel_tags = wheel_tags

            # Run ELF analysis
            from pybinaryguard.analyzers.elf_analyzer import ELFAnalyzer

            analyzer = ELFAnalyzer()
            pkg_info = analyzer.analyze(pkg_info)

            if not pkg_info.has_binaries:
                findings.append(Finding(
                    rule_id="PURE_PYTHON_WHEEL",
                    severity=Severity.PASSED,
                    title=f"{pkg_info.package_name} is pure Python",
                    explanation="No compiled extensions found in this wheel.",
                    package=pkg_info.package_name,
                    package_version=pkg_info.package_version,
                ))
                return findings

            # Evaluate rules against this package
            findings.extend(self._evaluate_rules(profile, [pkg_info]))

        return findings

    def _inspect_shared_object(
        self, so_path: str, profile: SystemProfile
    ) -> List[Finding]:
        """Inspect a single .so file against the current system."""
        from pybinaryguard.analyzers.elf_analyzer import ELFAnalyzer, ELFParseError

        findings: List[Finding] = []
        basename = os.path.basename(so_path)

        pkg_info = PackageBinaryInfo(
            package_name=basename,
            package_version="",
            install_path=os.path.dirname(so_path),
        )

        analyzer = ELFAnalyzer()
        try:
            so_info = analyzer._parse_so(so_path)
        except Exception as exc:
            findings.append(Finding(
                rule_id="ELF_PARSE_ERROR",
                severity=Severity.CRITICAL,
                title=f"Cannot parse '{basename}'",
                explanation=str(exc),
            ))
            return findings

        if so_info is None:
            findings.append(Finding(
                rule_id="ELF_PARSE_ERROR",
                severity=Severity.CRITICAL,
                title=f"'{basename}' is not a valid ELF binary",
                explanation="The file could not be parsed as an ELF shared object.",
            ))
            return findings

        pkg_info.shared_objects = [so_info]
        pkg_info.is_pure_python = False
        if so_info.required_glibc is not None:
            pkg_info.required_glibc = so_info.required_glibc
        if so_info.architecture is not None:
            pkg_info.target_architecture = so_info.architecture
        pkg_info.required_libraries = set(so_info.dt_needed)

        findings.extend(self._evaluate_rules(profile, [pkg_info]))
        return findings

    @staticmethod
    def _parse_wheel_filename(filename: str) -> Tuple[str, str]:
        """Parse a wheel filename to extract name and version.

        Wheel filenames follow the pattern:
            ``{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl``

        Returns ``("", "")`` if parsing fails.
        """
        if not filename.endswith(".whl"):
            return ("", "")
        base = filename[:-4]
        parts = base.split("-")
        if len(parts) < 5:
            return ("", "")
        return (parts[0], parts[1])

    @staticmethod
    def _tags_from_wheel_filename(filename: str) -> List[WheelTag]:
        """Extract compatibility tags from a wheel filename."""
        if not filename.endswith(".whl"):
            return []
        base = filename[:-4]
        parts = base.split("-")
        if len(parts) < 5:
            return []
        # Last three components are python-abi-platform
        python_tag = parts[-3]
        abi_tag = parts[-2]
        platform_tag = parts[-1]
        # Platform may contain multiple tags separated by "."
        tags: List[WheelTag] = []
        for plat in platform_tag.split("."):
            tags.append(WheelTag(
                interpreter=python_tag,
                abi=abi_tag,
                platform=plat,
            ))
        return tags

    # ------------------------------------------------------------------
    # Plugin helpers
    # ------------------------------------------------------------------

    def _get_plugin_registry(self) -> Optional[Any]:
        """Lazily discover and cache the plugin registry."""
        if not self._enable_plugins:
            return None

        if self._plugin_registry is not None:
            return self._plugin_registry

        try:
            from pybinaryguard.plugins.loader import discover_plugins

            self._plugin_registry = discover_plugins()
        except Exception as exc:
            logger.debug("Plugin discovery failed: %s", exc)
            self._plugin_registry = None

        return self._plugin_registry

    def _fire_pre_scan_hooks(self, profile: SystemProfile) -> None:
        """Execute all registered pre-scan hooks."""
        registry = self._get_plugin_registry()
        if registry is None:
            return
        for hook in registry.pre_scan_hooks:
            try:
                hook(profile)
            except Exception as exc:
                logger.debug("Pre-scan hook failed: %s", exc)

    def _fire_post_scan_hooks(self, profile: SystemProfile) -> None:
        """Execute all registered post-scan hooks."""
        registry = self._get_plugin_registry()
        if registry is None:
            return
        for hook in registry.post_scan_hooks:
            try:
                hook(profile)
            except Exception as exc:
                logger.debug("Post-scan hook failed: %s", exc)
