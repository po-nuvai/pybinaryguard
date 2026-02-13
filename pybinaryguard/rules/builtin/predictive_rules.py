"""Predictive rules that simulate runtime failures before they occur.

These rules use the predictive failure engine to forecast ImportErrors
and symbol resolution failures.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


class PREDICTED_IMPORT_ERROR(Rule):
    """Predicts ImportError before runtime using linker simulation."""

    rule_id = "PREDICTED_IMPORT_ERROR"
    description = "Predicts import failures via linker simulation"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True  # Checked per-package inside evaluate

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.predictor import predict_import_failures

        findings: List[Finding] = []

        for package in packages:
            if not package.has_binaries or len(package.shared_objects) == 0:
                continue

            try:
                predicted_failures = predict_import_failures(package, profile)

                for failure in predicted_failures:
                    if failure.error_type == "SymbolError":
                        explanation = (
                            f"Predicted ImportError in {package.package_name} "
                            f"due to missing symbol: {failure.missing_symbol}."
                        )
                        suggestion = (
                            f"Install missing library providing symbol "
                            f"{failure.missing_symbol}, or use a different "
                            f"version of {package.package_name}."
                        )
                    elif failure.error_type == "LibraryMissing":
                        explanation = (
                            f"Predicted ImportError in {package.package_name} "
                            f"due to missing library: {failure.missing_library}."
                        )
                        suggestion = (
                            f"Install {failure.missing_library} using your "
                            f"package manager, or rebuild {package.package_name}."
                        )
                    else:
                        explanation = failure.error_message
                        suggestion = (
                            f"Review {package.package_name} installation "
                            f"and dependencies"
                        )

                    findings.append(
                        Finding(
                            rule_id=self.rule_id,
                            severity=Severity.CRITICAL,
                            title=f"ImportError predicted: {failure.error_type}",
                            explanation=explanation,
                            package=package.package_name,
                            technical_detail=f"Module: {failure.module_path}",
                            suggestion=suggestion,
                            confidence=failure.confidence,
                        )
                    )
            except Exception:
                pass

        return findings


class UNRESOLVED_DEPENDENCY_CHAIN(Rule):
    """Detects unresolved dependencies in the dependency graph."""

    rule_id = "UNRESOLVED_DEPENDENCY_CHAIN"
    description = "Checks for unresolved libraries in dependency chain"

    def is_applicable(self, profile: SystemProfile) -> bool:
        return True

    def evaluate(
        self, profile: SystemProfile, packages: List[PackageBinaryInfo]
    ) -> List[Finding]:
        from pybinaryguard.predictor import build_dependency_graph

        findings: List[Finding] = []

        for package in packages:
            if not package.has_binaries or len(package.shared_objects) == 0:
                continue

            for so in package.shared_objects:
                if not so.path:
                    continue

                try:
                    graph = build_dependency_graph(so.path, None)
                    unresolved = graph.get_unresolved_dependencies()

                    if unresolved:
                        findings.append(
                            Finding(
                                rule_id=self.rule_id,
                                severity=Severity.WARNING,
                                title=(
                                    f"Unresolved dependencies in "
                                    f"{package.package_name}"
                                ),
                                explanation=(
                                    f"Required but not found: "
                                    f"{', '.join(unresolved)}."
                                ),
                                package=package.package_name,
                                technical_detail=f"Module: {so.path}",
                                suggestion=(
                                    f"Install missing libraries or use a "
                                    f"different build of {package.package_name}"
                                ),
                                confidence=0.85,
                            )
                        )

                    if graph.has_circular_dependencies():
                        findings.append(
                            Finding(
                                rule_id=self.rule_id,
                                severity=Severity.INFO,
                                title=(
                                    f"Circular dependencies in "
                                    f"{package.package_name}"
                                ),
                                explanation=(
                                    "Circular dependencies detected in the "
                                    "library dependency chain."
                                ),
                                package=package.package_name,
                                technical_detail=f"Module: {so.path}",
                                suggestion="Monitor for runtime issues",
                                confidence=0.7,
                            )
                        )
                except Exception:
                    pass

        return findings


__all__ = [
    "PREDICTED_IMPORT_ERROR",
    "UNRESOLVED_DEPENDENCY_CHAIN",
]
