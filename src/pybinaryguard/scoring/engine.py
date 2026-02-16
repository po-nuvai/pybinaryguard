"""Advanced Health Scoring v2 engine.

Multi-dimensional weighted scoring that evaluates environment health
across four categories rather than a simple linear penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

# Map rule_id prefixes/patterns to scoring categories
_CATEGORY_MAP: Dict[str, str] = {
    # Binary Stability
    "GLIBC_": "binary_stability",
    "MUSL_": "binary_stability",
    "MANYLINUX_": "binary_stability",
    "ARCH_": "binary_stability",
    "PYTHON_ABI_": "binary_stability",
    "PYTHON_VERSION_": "binary_stability",
    "ABI3_": "binary_stability",
    "DEBUG_RELEASE_": "binary_stability",
    "ELF_": "binary_stability",
    "LIBSTDCXX_": "binary_stability",
    # GPU Compatibility
    "CUDA_": "gpu_compat",
    "CUDNN_": "gpu_compat",
    "COMPUTE_CAPABILITY_": "gpu_compat",
    "PYTORCH_CUDA_": "gpu_compat",
    "TENSORFLOW_COMPUTE_": "gpu_compat",
    "TENSORFLOW_CUDA_": "gpu_compat",
    "TENSORRT_": "gpu_compat",
    "ONNX_RUNTIME_": "gpu_compat",
    "PYTORCH_TORCHVISION_": "gpu_compat",
    # Dependency Health
    "MISSING_SHARED_LIB": "dependency_health",
    "NUMPY_ABI_": "dependency_health",
    "PREDICTED_IMPORT_": "dependency_health",
    "UNRESOLVED_DEPENDENCY_": "dependency_health",
    "PACKAGE_NOT_FOUND": "dependency_health",
    # Platform Risk
    "CONTAINER_": "platform_risk",
    "JETPACK_": "platform_risk",
    "JETSON_": "platform_risk",
    "OPENCV_": "platform_risk",
    "BOARD_": "platform_risk",
    "KNOWN_BROKEN_": "platform_risk",
    "AVX": "binary_stability",
    "ILLEGAL_INSTRUCTION_": "binary_stability",
    "CPU_": "binary_stability",
    # Source build
    "SOURCE_BUILD_": "binary_stability",
    # Dependency health
    "DEPENDENCY_": "dependency_health",
    # Virtual environment
    "VENV_": "platform_risk",
}

# Category display metadata
_CATEGORY_INFO: Dict[str, Tuple[str, str]] = {
    "binary_stability": ("Binary Stability", "GLIBC, architecture, ABI, and ELF integrity"),
    "gpu_compat": ("GPU Compatibility", "CUDA stack, cuDNN, compute capability, framework builds"),
    "dependency_health": ("Dependency Health", "Shared library resolution and symbol availability"),
    "platform_risk": ("Platform Risk", "Container, embedded board, and platform-specific issues"),
}

# Default weights — sum to 1.0
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "binary_stability": 0.35,
    "gpu_compat": 0.30,
    "dependency_health": 0.25,
    "platform_risk": 0.10,
}

# Severity penalties (applied within each category, out of 100)
_SEVERITY_PENALTY: Dict[Severity, float] = {
    Severity.CRITICAL: 30.0,
    Severity.WARNING: 10.0,
    Severity.INFO: 2.0,
    Severity.PASSED: 0.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryScore:
    """Score for a single health category."""

    name: str
    display_name: str
    description: str
    score: float  # 0-100
    weight: float
    weighted_score: float  # score * weight
    finding_count: int = 0
    critical_count: int = 0
    warning_count: int = 0
    top_issues: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score >= 90:
            return "Excellent"
        elif self.score >= 70:
            return "Good"
        elif self.score >= 50:
            return "Fair"
        elif self.score >= 30:
            return "Poor"
        else:
            return "Critical"

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "score": round(self.score, 1),
            "weight": self.weight,
            "weighted_score": round(self.weighted_score, 1),
            "label": self.label,
            "finding_count": self.finding_count,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "top_issues": self.top_issues,
        }


@dataclass
class ScoreBreakdown:
    """Complete health score breakdown across all categories."""

    overall_score: float  # 0-100
    overall_label: str
    categories: Dict[str, CategoryScore] = field(default_factory=dict)
    total_findings: int = 0
    total_critical: int = 0
    total_warnings: int = 0

    @property
    def weakest_category(self) -> Optional[CategoryScore]:
        if not self.categories:
            return None
        return min(self.categories.values(), key=lambda c: c.score)

    @property
    def strongest_category(self) -> Optional[CategoryScore]:
        if not self.categories:
            return None
        return max(self.categories.values(), key=lambda c: c.score)

    def as_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "overall_score": round(self.overall_score, 1),
            "overall_label": self.overall_label,
            "total_findings": self.total_findings,
            "total_critical": self.total_critical,
            "total_warnings": self.total_warnings,
            "categories": {
                name: cat.as_dict() for name, cat in self.categories.items()
            },
        }
        weakest = self.weakest_category
        if weakest:
            result["weakest_category"] = weakest.name
        return result


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

class HealthScoreV2:
    """Multi-dimensional health scoring engine.

    Evaluates findings across four categories with configurable weights,
    producing a breakdown that shows exactly where environment health
    is strong and where it needs attention.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        has_gpu: bool = False,
        is_embedded: bool = False,
    ) -> None:
        self._weights = dict(weights or _DEFAULT_WEIGHTS)
        self._has_gpu = has_gpu
        self._is_embedded = is_embedded

        # Adjust weights based on environment
        self._adjust_weights()

    def _adjust_weights(self) -> None:
        """Redistribute weights based on environment context.

        If no GPU is present, GPU weight is redistributed.
        If not on embedded hardware, platform risk weight is redistributed.
        """
        redistribute = 0.0

        if not self._has_gpu and self._weights.get("gpu_compat", 0) > 0:
            redistribute += self._weights["gpu_compat"]
            self._weights["gpu_compat"] = 0.0

        if not self._is_embedded and self._weights.get("platform_risk", 0) > 0:
            extra = self._weights["platform_risk"] * 0.5
            redistribute += extra
            self._weights["platform_risk"] -= extra

        if redistribute > 0:
            # Distribute to non-zero categories
            active = [k for k, v in self._weights.items() if v > 0]
            if active:
                per_category = redistribute / len(active)
                for key in active:
                    self._weights[key] += per_category

        # Normalize to sum to 1.0
        total = sum(self._weights.values())
        if total > 0:
            for key in self._weights:
                self._weights[key] /= total

    def score(self, findings: List[Finding]) -> ScoreBreakdown:
        """Compute health score breakdown from findings.

        Parameters
        ----------
        findings:
            All findings from the scan.

        Returns
        -------
        ScoreBreakdown
            Detailed breakdown with per-category scores and overall score.
        """
        # Categorize findings
        categorized: Dict[str, List[Finding]] = {
            "binary_stability": [],
            "gpu_compat": [],
            "dependency_health": [],
            "platform_risk": [],
        }

        uncategorized: List[Finding] = []

        for finding in findings:
            category = self._classify_finding(finding)
            if category and category in categorized:
                categorized[category].append(finding)
            else:
                uncategorized.append(finding)

        # Assign uncategorized to binary_stability (safest default)
        categorized["binary_stability"].extend(uncategorized)

        # Score each category
        categories: Dict[str, CategoryScore] = {}
        for cat_name, cat_findings in categorized.items():
            display_name, description = _CATEGORY_INFO.get(
                cat_name, (cat_name, "")
            )
            weight = self._weights.get(cat_name, 0.0)
            cat_score = self._score_category(
                cat_name, display_name, description, cat_findings, weight
            )
            categories[cat_name] = cat_score

        # Compute overall weighted score
        overall = sum(cat.weighted_score for cat in categories.values())
        overall = max(0.0, min(100.0, overall))

        # Determine overall label
        overall_label = self._score_label(overall)

        total_findings = sum(
            1 for f in findings if f.severity != Severity.PASSED
        )
        total_critical = sum(
            1 for f in findings if f.severity == Severity.CRITICAL
        )
        total_warnings = sum(
            1 for f in findings if f.severity == Severity.WARNING
        )

        return ScoreBreakdown(
            overall_score=overall,
            overall_label=overall_label,
            categories=categories,
            total_findings=total_findings,
            total_critical=total_critical,
            total_warnings=total_warnings,
        )

    def _classify_finding(self, finding: Finding) -> Optional[str]:
        """Classify a finding into a scoring category."""
        rule_id = finding.rule_id
        for prefix, category in _CATEGORY_MAP.items():
            if rule_id.startswith(prefix):
                return category
        return None

    @staticmethod
    def _score_category(
        name: str,
        display_name: str,
        description: str,
        findings: List[Finding],
        weight: float,
    ) -> CategoryScore:
        """Score a single category from its findings."""
        # Start at 100, subtract penalties
        score = 100.0

        critical_count = 0
        warning_count = 0
        top_issues: List[str] = []

        for finding in findings:
            if finding.severity == Severity.PASSED:
                continue

            penalty = _SEVERITY_PENALTY.get(finding.severity, 0.0)

            # Scale penalty by confidence
            penalty *= finding.confidence

            score -= penalty

            if finding.severity == Severity.CRITICAL:
                critical_count += 1
                if len(top_issues) < 3:
                    top_issues.append(finding.title)
            elif finding.severity == Severity.WARNING:
                warning_count += 1
                if len(top_issues) < 3 and critical_count == 0:
                    top_issues.append(finding.title)

        score = max(0.0, min(100.0, score))

        finding_count = sum(
            1 for f in findings if f.severity != Severity.PASSED
        )

        return CategoryScore(
            name=name,
            display_name=display_name,
            description=description,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            finding_count=finding_count,
            critical_count=critical_count,
            warning_count=warning_count,
            top_issues=top_issues,
        )

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 90:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Needs Attention"
        elif score >= 30:
            return "Poor"
        else:
            return "Critical"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_health_score(
    findings: List[Finding],
    has_gpu: bool = False,
    is_embedded: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> ScoreBreakdown:
    """Compute the v2 health score breakdown.

    Parameters
    ----------
    findings:
        All findings from the scan.
    has_gpu:
        Whether the system has a GPU (affects weight distribution).
    is_embedded:
        Whether running on an embedded board (affects weight distribution).
    weights:
        Optional custom category weights (must sum to 1.0).

    Returns
    -------
    ScoreBreakdown
        Complete health score breakdown.
    """
    engine = HealthScoreV2(
        weights=weights,
        has_gpu=has_gpu,
        is_embedded=is_embedded,
    )
    return engine.score(findings)
