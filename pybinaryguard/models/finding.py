"""Finding and report data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from .enums import Severity


@dataclass
class Finding:
    """A single diagnostic result from the compatibility check."""

    rule_id: str
    severity: Severity
    title: str
    explanation: str
    technical_detail: str = ""
    suggestion: str = ""
    package: Optional[str] = None
    package_version: Optional[str] = None
    confidence: float = 1.0
    related_error: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        """Convert to a JSON-serializable dict."""
        result: Dict[str, object] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "title": self.title,
            "explanation": self.explanation,
        }
        if self.technical_detail:
            result["technical_detail"] = self.technical_detail
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.package:
            result["package"] = self.package
        if self.package_version:
            result["package_version"] = self.package_version
        if self.confidence < 1.0:
            result["confidence"] = self.confidence
        if self.related_error:
            result["related_error"] = self.related_error
        return result


@dataclass
class ScanReport:
    """Complete scan report containing all findings and metadata."""

    findings: List[Finding] = field(default_factory=list)
    packages_scanned: int = 0
    total_packages: int = 0
    scan_duration_ms: float = 0.0
    detected_board: Optional[str] = None  # Display name of detected embedded board
    score_breakdown: Optional[object] = None  # ScoreBreakdown from scoring.engine

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    @property
    def passed_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.PASSED)

    @property
    def health_score(self) -> int:
        """Return the v2 weighted health score if available, else v1 linear."""
        if self.score_breakdown is not None:
            return round(self.score_breakdown.overall_score)
        score = 100 - (self.critical_count * 25) - (self.warning_count * 5) - (self.info_count * 1)
        return max(0, min(100, score))

    @property
    def health_label(self) -> str:
        if self.score_breakdown is not None:
            return self.score_breakdown.overall_label
        score = self.health_score
        if score >= 90:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Needs Attention"
        else:
            return "Critical"

    def summary_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "total_packages": self.total_packages,
            "packages_scanned": self.packages_scanned,
            "critical": self.critical_count,
            "warning": self.warning_count,
            "info": self.info_count,
            "passed": self.passed_count,
        }
        if self.detected_board:
            result["detected_board"] = self.detected_board
        return result
