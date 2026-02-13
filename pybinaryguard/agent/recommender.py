"""Action recommendation engine for agent workflows.

Transforms raw diagnostic findings into structured, executable actions
classified by safety level so agents know what they can auto-execute
versus what requires human approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding


@dataclass
class RecommendedAction:
    """A single recommended action an agent can take.

    Attributes:
        action_type: Category — "reinstall", "install", "uninstall",
            "downgrade", "upgrade", "configure", "ignore".
        target: The package or system component to act on.
        command: Ready-to-execute shell command (usually pip).
        reason: One-sentence explanation of why this action is needed.
        safety: "safe" (agent can auto-execute), "review" (agent should
            confirm with user), or "dangerous" (requires human approval).
        confidence: 0.0-1.0 confidence that this action will fix the issue.
        finding_id: The rule_id of the finding that triggered this action.
        priority: Lower number = higher priority. 1 = fix first.
    """

    action_type: str
    target: str
    command: str
    reason: str
    safety: str  # "safe", "review", "dangerous"
    confidence: float = 0.8
    finding_id: str = ""
    priority: int = 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "command": self.command,
            "reason": self.reason,
            "safety": self.safety,
            "confidence": round(self.confidence, 2),
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Action templates keyed by rule_id patterns
# ---------------------------------------------------------------------------

_ACTION_TEMPLATES: List[Dict[str, object]] = [
    # CUDA mismatches
    {
        "pattern": r"CUDA_RUNTIME_MISMATCH|PYTORCH_CUDA_ABI_MISMATCH",
        "action_type": "reinstall",
        "command_template": "pip install {package} --index-url https://download.pytorch.org/whl/cu{cuda_short}",
        "reason": "Package built for wrong CUDA version",
        "safety": "safe",
        "confidence": 0.9,
    },
    {
        "pattern": r"CUDA_DRIVER_TOO_OLD",
        "action_type": "configure",
        "command_template": "# Upgrade GPU driver or install package for older CUDA: pip install {package}+cu{driver_cuda}",
        "reason": "GPU driver too old for installed CUDA runtime",
        "safety": "dangerous",
        "confidence": 0.7,
    },
    # GLIBC
    {
        "pattern": r"GLIBC_VERSION_MISMATCH",
        "action_type": "downgrade",
        "command_template": "pip install {package}<={safe_version}",
        "reason": "Package requires newer GLIBC than system provides",
        "safety": "safe",
        "confidence": 0.8,
    },
    {
        "pattern": r"MUSL_GLIBC_CONFLICT",
        "action_type": "reinstall",
        "command_template": "pip install --no-binary :all: {package}",
        "reason": "Binary wheel is glibc-linked but system uses musl (Alpine)",
        "safety": "review",
        "confidence": 0.7,
    },
    # Architecture
    {
        "pattern": r"ARCH_MISMATCH|JETSON_X86_WHEEL",
        "action_type": "reinstall",
        "command_template": "pip install --no-binary :all: {package}",
        "reason": "Binary compiled for wrong CPU architecture",
        "safety": "review",
        "confidence": 0.6,
    },
    # Python ABI
    {
        "pattern": r"PYTHON_ABI_MISMATCH|PYTHON_VERSION_MISMATCH",
        "action_type": "reinstall",
        "command_template": "pip install --force-reinstall {package}",
        "reason": "Package built for different Python version/ABI",
        "safety": "safe",
        "confidence": 0.9,
    },
    # Missing libraries
    {
        "pattern": r"MISSING_SHARED_LIB|CUDA_LIB_MISSING",
        "action_type": "install",
        "command_template": "# Install missing system library or: pip install {package}",
        "reason": "Required shared library not found on system",
        "safety": "review",
        "confidence": 0.6,
    },
    # NumPy ABI
    {
        "pattern": r"NUMPY_ABI_MISMATCH",
        "action_type": "reinstall",
        "command_template": "pip install --force-reinstall {package}",
        "reason": "NumPy C API version mismatch",
        "safety": "safe",
        "confidence": 0.9,
    },
    # CPU instruction set
    {
        "pattern": r"AVX2_REQUIRED|ILLEGAL_INSTRUCTION_RISK",
        "action_type": "downgrade",
        "command_template": "pip install {package} --prefer-binary",
        "reason": "Binary requires CPU instructions not available on this system",
        "safety": "review",
        "confidence": 0.5,
    },
    # Board/embedded
    {
        "pattern": r"KNOWN_BROKEN_WHEEL|BOARD_INCOMPATIBLE_PACKAGE",
        "action_type": "uninstall",
        "command_template": "pip uninstall -y {package}",
        "reason": "Package known incompatible with detected board",
        "safety": "review",
        "confidence": 0.8,
    },
    # Predictive failures
    {
        "pattern": r"PREDICTED_IMPORT_ERROR",
        "action_type": "reinstall",
        "command_template": "pip install --force-reinstall {package}",
        "reason": "Predicted import failure based on dependency analysis",
        "safety": "safe",
        "confidence": 0.7,
    },
    # Container
    {
        "pattern": r"CONTAINER_NO_GPU_MOUNT",
        "action_type": "configure",
        "command_template": "# Re-run container with: --gpus all",
        "reason": "GPU devices not mounted in container",
        "safety": "dangerous",
        "confidence": 0.9,
    },
    # TensorRT
    {
        "pattern": r"TENSORRT_INCOMPATIBLE",
        "action_type": "reinstall",
        "command_template": "pip install tensorrt=={compatible_version}",
        "reason": "TensorRT version incompatible with current GPU/CUDA",
        "safety": "review",
        "confidence": 0.7,
    },
]


class ActionRecommender:
    """Transforms findings into executable recommended actions.

    The recommender matches findings against known action templates and
    produces structured actions that agents can auto-execute (safe),
    present for review, or flag as dangerous.
    """

    def recommend(self, findings: List[Finding]) -> List[RecommendedAction]:
        """Generate recommended actions from findings.

        Parameters
        ----------
        findings:
            Diagnostic findings from a scan or check.

        Returns
        -------
        List[RecommendedAction]
            Actions sorted by priority (highest priority first).
        """
        actions: List[RecommendedAction] = []
        seen_targets: set = set()

        for finding in findings:
            if finding.severity == Severity.PASSED:
                continue

            action = self._match_action(finding)
            if action is None:
                continue

            # Deduplicate: one action per target package
            target_key = (action.action_type, action.target)
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)

            actions.append(action)

        # Sort by priority: critical first, then by confidence descending
        actions.sort(key=lambda a: (a.priority, -a.confidence))
        return actions

    def _match_action(self, finding: Finding) -> Optional[RecommendedAction]:
        """Match a finding to an action template."""
        for template in _ACTION_TEMPLATES:
            pattern = template["pattern"]
            if re.match(pattern, finding.rule_id):
                return self._instantiate_action(template, finding)

        # Fallback: use the finding's own suggestion if available
        if finding.suggestion and finding.package:
            return RecommendedAction(
                action_type="fix",
                target=finding.package,
                command=finding.suggestion,
                reason=finding.title,
                safety="review",
                confidence=finding.confidence * 0.6,
                finding_id=finding.rule_id,
                priority=self._severity_to_priority(finding.severity),
            )

        return None

    def _instantiate_action(
        self, template: Dict[str, object], finding: Finding
    ) -> RecommendedAction:
        """Create a concrete action from a template and finding."""
        package = finding.package or "unknown"
        command_template = str(template["command_template"])

        # Fill template variables
        command = command_template.format(
            package=package,
            cuda_short="121",  # sensible default
            safe_version="",
            driver_cuda="118",
            compatible_version="",
        )

        # If the finding has a specific suggestion, prefer it
        if finding.suggestion:
            command = finding.suggestion

        return RecommendedAction(
            action_type=str(template["action_type"]),
            target=package,
            command=command,
            reason=str(template["reason"]),
            safety=str(template["safety"]),
            confidence=float(template.get("confidence", 0.7)),
            finding_id=finding.rule_id,
            priority=self._severity_to_priority(finding.severity),
        )

    @staticmethod
    def _severity_to_priority(severity: Severity) -> int:
        return {
            Severity.CRITICAL: 1,
            Severity.WARNING: 2,
            Severity.INFO: 3,
            Severity.PASSED: 4,
        }.get(severity, 3)
