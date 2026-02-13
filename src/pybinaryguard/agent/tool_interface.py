"""Agent-native tool interface for PyBinaryGuard.

Every function returns a structured, JSON-serializable dataclass that agents
can parse without interpreting human-formatted text.  Each result includes
recommended actions classified by safety level.

Usage::

    from pybinaryguard.agent import scan, check, simulate_install, doctor

    report = scan()                        # Full environment scan
    result = check("torch")                # Single package
    sim    = simulate_install("torch==2.4.0+cu124")  # Pre-install prediction
    dx     = doctor("GLIBC_2.34 not found")          # Error diagnosis
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pybinaryguard.agent.recommender import ActionRecommender, RecommendedAction
from pybinaryguard.models.enums import ScanMode, Severity
from pybinaryguard.models.finding import Finding


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ActionableReport:
    """Structured scan result for agent consumption.

    Every field is JSON-serializable.  Agents should check ``risk_level``
    and iterate ``safe_actions`` for auto-executable fixes.
    """

    health_score: int
    risk_level: str          # "low", "medium", "high", "critical"
    total_packages: int
    packages_scanned: int
    scan_duration_ms: float
    issues: List[Dict[str, object]] = field(default_factory=list)
    safe_actions: List[Dict[str, object]] = field(default_factory=list)
    review_actions: List[Dict[str, object]] = field(default_factory=list)
    dangerous_actions: List[Dict[str, object]] = field(default_factory=list)
    score_breakdown: Optional[Dict[str, object]] = None
    detected_board: Optional[str] = None
    profile_summary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "health_score": self.health_score,
            "risk_level": self.risk_level,
            "total_packages": self.total_packages,
            "packages_scanned": self.packages_scanned,
            "scan_duration_ms": round(self.scan_duration_ms, 1),
            "issue_count": len(self.issues),
            "issues": self.issues,
            "safe_actions": self.safe_actions,
            "review_actions": self.review_actions,
            "dangerous_actions": self.dangerous_actions,
        }
        if self.score_breakdown:
            result["score_breakdown"] = self.score_breakdown
        if self.detected_board:
            result["detected_board"] = self.detected_board
        if self.profile_summary:
            result["profile"] = self.profile_summary
        return result

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class AgentCheckResult:
    """Structured result from checking a single package."""

    package: str
    compatible: bool
    risk_level: str
    issues: List[Dict[str, object]] = field(default_factory=list)
    safe_actions: List[Dict[str, object]] = field(default_factory=list)
    review_actions: List[Dict[str, object]] = field(default_factory=list)
    dangerous_actions: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "package": self.package,
            "compatible": self.compatible,
            "risk_level": self.risk_level,
            "issue_count": len(self.issues),
            "issues": self.issues,
            "safe_actions": self.safe_actions,
            "review_actions": self.review_actions,
            "dangerous_actions": self.dangerous_actions,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class AgentSimulateResult:
    """Structured result from simulating a package installation."""

    package_spec: str
    predicted_compatible: bool
    confidence: float
    risk_level: str
    warnings: List[Dict[str, object]] = field(default_factory=list)
    blockers: List[Dict[str, object]] = field(default_factory=list)
    parsed_tags: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "package_spec": self.package_spec,
            "predicted_compatible": self.predicted_compatible,
            "confidence": round(self.confidence, 2),
            "risk_level": self.risk_level,
            "blocker_count": len(self.blockers),
            "blockers": self.blockers,
            "warning_count": len(self.warnings),
            "warnings": self.warnings,
        }
        if self.parsed_tags:
            result["parsed_tags"] = self.parsed_tags
        return result

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class AgentDoctorResult:
    """Structured result from diagnosing an error."""

    error_message: str
    diagnosis: str
    fix_plan: List[Dict[str, object]] = field(default_factory=list)
    auto_fix_safe: bool = False
    related_findings: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "error_message": self.error_message,
            "diagnosis": self.diagnosis,
            "fix_plan": self.fix_plan,
            "auto_fix_safe": self.auto_fix_safe,
            "related_findings": self.related_findings,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _risk_level(findings: List[Finding]) -> str:
    """Compute risk level from findings."""
    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
    warning = sum(1 for f in findings if f.severity == Severity.WARNING)
    if critical >= 3:
        return "critical"
    if critical >= 1:
        return "high"
    if warning >= 3:
        return "medium"
    if warning >= 1:
        return "low"
    return "none"


def _findings_to_issues(findings: List[Finding]) -> List[Dict[str, object]]:
    """Convert findings to agent-friendly issue dicts."""
    issues: List[Dict[str, object]] = []
    for f in findings:
        if f.severity == Severity.PASSED:
            continue
        issue: Dict[str, object] = {
            "type": f.rule_id,
            "severity": f.severity.value,
            "title": f.title,
            "explanation": f.explanation,
        }
        if f.package:
            issue["package"] = f.package
        if f.suggestion:
            issue["fix_hint"] = f.suggestion
        if f.confidence < 1.0:
            issue["confidence"] = round(f.confidence, 2)
        issues.append(issue)
    return issues


def _classify_actions(
    actions: List[RecommendedAction],
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Split actions into safe, review, dangerous buckets."""
    safe: List[Dict[str, object]] = []
    review: List[Dict[str, object]] = []
    dangerous: List[Dict[str, object]] = []
    for action in actions:
        d = action.to_dict()
        if action.safety == "safe":
            safe.append(d)
        elif action.safety == "dangerous":
            dangerous.append(d)
        else:
            review.append(d)
    return safe, review, dangerous


# ---------------------------------------------------------------------------
# Core agent API functions
# ---------------------------------------------------------------------------

_recommender = ActionRecommender()


def scan(
    scan_mode: str = "standard",
    packages: Optional[List[str]] = None,
    severity_threshold: str = "all",
    timeout: float = 30.0,
) -> ActionableReport:
    """Full environment scan returning structured, agent-consumable results.

    Parameters
    ----------
    scan_mode:
        "fast", "standard", or "deep".
    packages:
        Restrict scan to these package names.  ``None`` = scan all.
    severity_threshold:
        Minimum severity: "critical", "warning", "info", "all".
    timeout:
        Max probe time in seconds.

    Returns
    -------
    ActionableReport
        Structured report with health score, issues, and classified actions.
    """
    from pybinaryguard.scanner import Scanner

    mode = ScanMode(scan_mode)
    severity_map = {
        "critical": Severity.CRITICAL,
        "warning": Severity.WARNING,
        "info": Severity.INFO,
        "all": Severity.INFO,
    }

    scanner = Scanner(
        packages=packages,
        severity_threshold=severity_map.get(severity_threshold, Severity.INFO),
        timeout=timeout,
        scan_mode=mode,
    )

    report = scanner.run()
    profile = scanner.get_profile()

    # Generate actions
    actions = _recommender.recommend(report.findings)
    safe, review, dangerous = _classify_actions(actions)

    # Score breakdown
    breakdown_dict = None
    if report.score_breakdown is not None:
        breakdown_dict = report.score_breakdown.as_dict()

    return ActionableReport(
        health_score=report.health_score,
        risk_level=_risk_level(report.findings),
        total_packages=report.total_packages,
        packages_scanned=report.packages_scanned,
        scan_duration_ms=report.scan_duration_ms,
        issues=_findings_to_issues(report.findings),
        safe_actions=safe,
        review_actions=review,
        dangerous_actions=dangerous,
        score_breakdown=breakdown_dict,
        detected_board=report.detected_board,
        profile_summary=profile.summary(),
    )


def check(package: str, timeout: float = 30.0) -> AgentCheckResult:
    """Check a single installed package.

    Parameters
    ----------
    package:
        Package name (e.g. "torch", "numpy").
    timeout:
        Max probe time in seconds.

    Returns
    -------
    AgentCheckResult
        Structured compatibility result with actions.
    """
    from pybinaryguard.scanner import Scanner

    scanner = Scanner(packages=[package], timeout=timeout)
    findings = scanner.check_package(package)

    actions = _recommender.recommend(findings)
    safe, review, dangerous = _classify_actions(actions)

    critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)

    return AgentCheckResult(
        package=package,
        compatible=critical == 0,
        risk_level=_risk_level(findings),
        issues=_findings_to_issues(findings),
        safe_actions=safe,
        review_actions=review,
        dangerous_actions=dangerous,
    )


def simulate_install(package_spec: str) -> AgentSimulateResult:
    """Predict compatibility BEFORE installing a package.

    Analyses the package specifier (name, version pin, or wheel filename)
    against the current system profile to predict whether installation
    will produce a working binary.

    Parameters
    ----------
    package_spec:
        Package specifier.  Examples:
        - ``"torch"`` — name only, limited prediction
        - ``"torch==2.4.0+cu124"`` — version with CUDA variant
        - ``"torch-2.4.0+cu124-cp312-cp312-manylinux_2_17_x86_64.whl"``

    Returns
    -------
    AgentSimulateResult
        Prediction with blockers, warnings, and confidence.
    """
    from pybinaryguard.agent.simulator import simulate
    return simulate(package_spec)


def doctor(
    error_message: str,
    package: Optional[str] = None,
    timeout: float = 30.0,
) -> AgentDoctorResult:
    """Diagnose an error message and produce a structured fix plan.

    Parameters
    ----------
    error_message:
        The error string or traceback to diagnose.
    package:
        Optional package name related to the error.
    timeout:
        Max probe time in seconds.

    Returns
    -------
    AgentDoctorResult
        Diagnosis with fix plan and auto-fix safety indicator.
    """
    from pybinaryguard.diagnostics.explainer import diagnose_error

    # Diagnose the error text
    diagnosis_text = ""
    fix_plan: List[Dict[str, object]] = []
    related_findings: List[Dict[str, object]] = []

    result = diagnose_error(error_message)
    if result:
        diagnosis_text = result.get("explanation", "Unknown error")
        if result.get("fix_hint"):
            fix_plan.append({
                "step": 1,
                "action": result["fix_hint"],
                "safety": "review",
            })

    # If package specified, also run a check
    if package:
        from pybinaryguard.scanner import Scanner

        scanner = Scanner(packages=[package], timeout=timeout)
        findings = scanner.check_package(package)

        related_findings = _findings_to_issues(findings)

        # Generate fix actions from findings
        actions = _recommender.recommend(findings)
        for i, action in enumerate(actions):
            fix_plan.append({
                "step": len(fix_plan) + 1,
                "action": action.command,
                "target": action.target,
                "reason": action.reason,
                "safety": action.safety,
            })

    if not diagnosis_text:
        diagnosis_text = "Could not match this error to a known pattern."

    # Auto-fix is safe only if ALL fix plan items are "safe"
    auto_fix_safe = (
        len(fix_plan) > 0
        and all(step.get("safety") == "safe" for step in fix_plan)
    )

    return AgentDoctorResult(
        error_message=error_message,
        diagnosis=diagnosis_text,
        fix_plan=fix_plan,
        auto_fix_safe=auto_fix_safe,
        related_findings=related_findings,
    )


# ---------------------------------------------------------------------------
# Framework integration helper
# ---------------------------------------------------------------------------

def as_agent_tool() -> Dict[str, Any]:
    """Return a tool descriptor with bound handler functions.

    Returns a dict that agent frameworks can register directly::

        tool = pybinaryguard.agent.as_agent_tool()
        # tool["scan"]["handler"](scan_mode="fast")

    Returns
    -------
    Dict[str, Any]
        Tool descriptors with ``schema`` and ``handler`` keys.
    """
    from pybinaryguard.agent.schema import get_tool_descriptors

    schemas = get_tool_descriptors(format="json_schema")
    handlers = {
        "pybinaryguard_scan": scan,
        "pybinaryguard_check": check,
        "pybinaryguard_simulate_install": simulate_install,
        "pybinaryguard_doctor": doctor,
    }

    tools: Dict[str, Any] = {}
    for schema in schemas:
        name = schema["name"]
        tools[name] = {
            "schema": schema,
            "handler": handlers.get(name),
        }
    return tools
