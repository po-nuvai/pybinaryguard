"""PyBinaryGuard Agent SDK — agent-native binary compatibility intelligence.

Provides structured, machine-readable diagnostics that AI agents can call,
interpret, and act on without parsing human-formatted output.

Quick start for agents::

    from pybinaryguard.agent import scan, check, simulate_install, doctor

    # Full environment scan — returns structured ActionableReport
    report = scan()
    print(report.to_dict())     # JSON-serializable
    print(report.safe_actions)  # Auto-executable pip commands

    # Check one package
    result = check("torch")

    # Predict compatibility BEFORE installing
    sim = simulate_install("torch==2.4.0+cu124")

    # Diagnose an error
    dx = doctor("GLIBC_2.34 not found")

    # Export tool schema for agent framework registration
    from pybinaryguard.agent import export_tool_schema
    schema = export_tool_schema(format="openai")

    # Register as agent tool (one-liner)
    tool = as_agent_tool()
"""

from __future__ import annotations

from pybinaryguard.agent.tool_interface import (
    ActionableReport,
    AgentCheckResult,
    AgentDoctorResult,
    AgentSimulateResult,
    RecommendedAction,
    as_agent_tool,
    check,
    doctor,
    scan,
    simulate_install,
)
from pybinaryguard.agent.schema import export_tool_schema, get_tool_descriptors
from pybinaryguard.agent.recommender import ActionRecommender

__all__ = [
    "ActionRecommender",
    "ActionableReport",
    "AgentCheckResult",
    "AgentDoctorResult",
    "AgentSimulateResult",
    "RecommendedAction",
    "as_agent_tool",
    "check",
    "doctor",
    "export_tool_schema",
    "get_tool_descriptors",
    "scan",
    "simulate_install",
]
