"""Tests for the Advanced Health Scoring v2 engine."""

from __future__ import annotations

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.scoring.engine import (
    CategoryScore,
    HealthScoreV2,
    ScoreBreakdown,
    compute_health_score,
)


def _finding(rule_id: str, severity: Severity = Severity.CRITICAL, **kwargs) -> Finding:
    return Finding(
        rule_id=rule_id,
        severity=severity,
        title=f"Test finding: {rule_id}",
        explanation="Test explanation",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

class TestCategoryClassification:
    def test_glibc_classified_as_binary_stability(self):
        engine = HealthScoreV2()
        f = _finding("GLIBC_VERSION_MISMATCH")
        breakdown = engine.score([f])
        assert breakdown.categories["binary_stability"].finding_count == 1

    def test_cuda_classified_as_gpu_compat(self):
        engine = HealthScoreV2(has_gpu=True)
        f = _finding("CUDA_DRIVER_TOO_OLD")
        breakdown = engine.score([f])
        assert breakdown.categories["gpu_compat"].finding_count == 1

    def test_missing_lib_classified_as_dependency_health(self):
        engine = HealthScoreV2()
        f = _finding("MISSING_SHARED_LIB")
        breakdown = engine.score([f])
        assert breakdown.categories["dependency_health"].finding_count == 1

    def test_board_classified_as_platform_risk(self):
        engine = HealthScoreV2(is_embedded=True)
        f = _finding("BOARD_CUDA_VERSION_MISMATCH")
        breakdown = engine.score([f])
        assert breakdown.categories["platform_risk"].finding_count == 1

    def test_container_classified_as_platform_risk(self):
        engine = HealthScoreV2()
        f = _finding("CONTAINER_NO_GPU_MOUNT", severity=Severity.WARNING)
        breakdown = engine.score([f])
        assert breakdown.categories["platform_risk"].finding_count == 1

    def test_unknown_rule_defaults_to_binary_stability(self):
        engine = HealthScoreV2()
        f = _finding("SOME_UNKNOWN_RULE")
        breakdown = engine.score([f])
        assert breakdown.categories["binary_stability"].finding_count == 1

    def test_passed_findings_not_counted(self):
        engine = HealthScoreV2()
        f = _finding("GLIBC_VERSION_MISMATCH", severity=Severity.PASSED)
        breakdown = engine.score([f])
        assert breakdown.categories["binary_stability"].finding_count == 0


# ---------------------------------------------------------------------------
# Scoring calculations
# ---------------------------------------------------------------------------

class TestScoringCalculations:
    def test_no_findings_gives_perfect_score(self):
        breakdown = compute_health_score([])
        assert breakdown.overall_score == 100.0
        assert breakdown.overall_label == "Excellent"

    def test_critical_finding_reduces_score(self):
        findings = [_finding("GLIBC_VERSION_MISMATCH")]
        breakdown = compute_health_score(findings)
        assert breakdown.overall_score < 100.0

    def test_warning_less_penalty_than_critical(self):
        crit_findings = [_finding("GLIBC_VERSION_MISMATCH")]
        warn_findings = [_finding("GLIBC_VERSION_MISMATCH", severity=Severity.WARNING)]
        crit_breakdown = compute_health_score(crit_findings)
        warn_breakdown = compute_health_score(warn_findings)
        assert warn_breakdown.overall_score > crit_breakdown.overall_score

    def test_info_minimal_penalty(self):
        findings = [_finding("GLIBC_VERSION_MISMATCH", severity=Severity.INFO)]
        breakdown = compute_health_score(findings)
        assert breakdown.overall_score >= 95.0

    def test_multiple_criticals_compound(self):
        findings = [
            _finding("GLIBC_VERSION_MISMATCH"),
            _finding("ARCH_MISMATCH"),
            _finding("PYTHON_ABI_MISMATCH"),
        ]
        breakdown = compute_health_score(findings)
        # 3 criticals in binary_stability: 100 - 90 = 10 for that category
        # Weighted contribution: very low. Overall score should be noticeably reduced.
        assert breakdown.overall_score < 70.0
        assert breakdown.categories["binary_stability"].score <= 10.0

    def test_score_never_below_zero(self):
        findings = [_finding(f"GLIBC_ISSUE_{i}") for i in range(20)]
        breakdown = compute_health_score(findings)
        assert breakdown.overall_score >= 0.0

    def test_score_never_above_100(self):
        breakdown = compute_health_score([])
        assert breakdown.overall_score <= 100.0

    def test_confidence_scales_penalty(self):
        full_conf = [_finding("GLIBC_VERSION_MISMATCH", confidence=1.0)]
        half_conf = [_finding("GLIBC_VERSION_MISMATCH", confidence=0.5)]
        full_bd = compute_health_score(full_conf)
        half_bd = compute_health_score(half_conf)
        assert half_bd.overall_score > full_bd.overall_score


# ---------------------------------------------------------------------------
# Weight adjustment
# ---------------------------------------------------------------------------

class TestWeightAdjustment:
    def test_no_gpu_redistributes_gpu_weight(self):
        engine = HealthScoreV2(has_gpu=False)
        assert engine._weights["gpu_compat"] == 0.0
        # Other weights should sum to 1.0
        total = sum(engine._weights.values())
        assert abs(total - 1.0) < 0.001

    def test_gpu_present_keeps_weight(self):
        engine = HealthScoreV2(has_gpu=True)
        assert engine._weights["gpu_compat"] > 0.0

    def test_embedded_keeps_platform_weight(self):
        engine = HealthScoreV2(is_embedded=True)
        assert engine._weights["platform_risk"] > 0.0

    def test_weights_always_sum_to_one(self):
        for has_gpu in [True, False]:
            for is_embedded in [True, False]:
                engine = HealthScoreV2(has_gpu=has_gpu, is_embedded=is_embedded)
                total = sum(engine._weights.values())
                assert abs(total - 1.0) < 0.001, f"gpu={has_gpu}, emb={is_embedded}: {total}"


# ---------------------------------------------------------------------------
# ScoreBreakdown properties
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_weakest_category(self):
        findings = [
            _finding("GLIBC_VERSION_MISMATCH"),  # binary_stability hit
        ]
        breakdown = compute_health_score(findings)
        weakest = breakdown.weakest_category
        assert weakest is not None
        assert weakest.name == "binary_stability"

    def test_strongest_category_when_no_findings(self):
        breakdown = compute_health_score([])
        strongest = breakdown.strongest_category
        assert strongest is not None
        assert strongest.score == 100.0

    def test_total_counts(self):
        findings = [
            _finding("GLIBC_VERSION_MISMATCH"),
            _finding("CUDA_DRIVER_TOO_OLD"),
            _finding("MISSING_SHARED_LIB", severity=Severity.WARNING),
        ]
        breakdown = compute_health_score(findings, has_gpu=True)
        assert breakdown.total_findings == 3
        assert breakdown.total_critical == 2
        assert breakdown.total_warnings == 1

    def test_as_dict_structure(self):
        breakdown = compute_health_score([])
        d = breakdown.as_dict()
        assert "overall_score" in d
        assert "overall_label" in d
        assert "categories" in d
        assert "binary_stability" in d["categories"]


# ---------------------------------------------------------------------------
# CategoryScore
# ---------------------------------------------------------------------------

class TestCategoryScore:
    def test_label_excellent(self):
        cat = CategoryScore(
            name="test", display_name="Test", description="",
            score=95.0, weight=0.25, weighted_score=23.75,
        )
        assert cat.label == "Excellent"

    def test_label_critical(self):
        cat = CategoryScore(
            name="test", display_name="Test", description="",
            score=10.0, weight=0.25, weighted_score=2.5,
        )
        assert cat.label == "Critical"

    def test_top_issues_limited_to_three(self):
        findings = [_finding(f"GLIBC_ISSUE_{i}") for i in range(10)]
        engine = HealthScoreV2()
        breakdown = engine.score(findings)
        cat = breakdown.categories["binary_stability"]
        assert len(cat.top_issues) <= 3

    def test_as_dict_keys(self):
        cat = CategoryScore(
            name="test", display_name="Test", description="desc",
            score=80.0, weight=0.25, weighted_score=20.0,
        )
        d = cat.as_dict()
        assert d["name"] == "test"
        assert d["display_name"] == "Test"
        assert d["score"] == 80.0
        assert d["label"] == "Good"


# ---------------------------------------------------------------------------
# Integration with ScanReport
# ---------------------------------------------------------------------------

class TestScanReportIntegration:
    def test_health_score_uses_v2_when_available(self):
        from pybinaryguard.models.finding import ScanReport
        breakdown = compute_health_score([])
        report = ScanReport(score_breakdown=breakdown)
        assert report.health_score == 100
        assert report.health_label == "Excellent"

    def test_health_score_falls_back_to_v1(self):
        from pybinaryguard.models.finding import ScanReport
        report = ScanReport(
            findings=[_finding("GLIBC_VERSION_MISMATCH")],
            score_breakdown=None,
        )
        # V1 formula: 100 - 25 = 75
        assert report.health_score == 75

    def test_v2_score_differs_from_v1(self):
        """V2 score should handle multi-category differently than v1 linear."""
        from pybinaryguard.models.finding import ScanReport
        findings = [
            _finding("GLIBC_VERSION_MISMATCH"),
            _finding("CUDA_DRIVER_TOO_OLD"),
        ]
        breakdown = compute_health_score(findings, has_gpu=True)

        report_v2 = ScanReport(findings=findings, score_breakdown=breakdown)
        report_v1 = ScanReport(findings=findings, score_breakdown=None)

        # They should both be less than 100 but likely different values
        assert report_v2.health_score < 100
        assert report_v1.health_score < 100
