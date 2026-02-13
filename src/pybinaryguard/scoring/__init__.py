"""Advanced Health Scoring v2 — multi-dimensional weighted environment scoring.

Provides category-based scoring across four dimensions:
- Binary Stability (GLIBC, architecture, ELF integrity)
- GPU Compatibility (CUDA stack, cuDNN, compute capability)
- Dependency Health (missing libraries, symbol resolution)
- Platform Risk (embedded thermal, container issues, board compat)
"""

from __future__ import annotations

from .engine import HealthScoreV2, ScoreBreakdown, CategoryScore, compute_health_score

__all__ = [
    "HealthScoreV2",
    "ScoreBreakdown",
    "CategoryScore",
    "compute_health_score",
]
