"""Tool schema export for agent framework registration.

Generates tool descriptors in formats compatible with:
- OpenAI function calling (Chat Completions API)
- MCP (Model Context Protocol) tool format
- Generic JSON Schema
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "pybinaryguard_scan",
        "description": (
            "Scan the current Python environment for binary compatibility "
            "issues. Returns a structured report with health score, findings, "
            "and recommended fix actions. Use this before deploying or after "
            "installing new packages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scan_mode": {
                    "type": "string",
                    "enum": ["fast", "standard", "deep"],
                    "description": (
                        "Scan depth. 'fast' checks metadata only (<1s). "
                        "'standard' parses binaries. 'deep' adds hash "
                        "verification and full symbol resolution."
                    ),
                    "default": "standard",
                },
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict scan to these package names. Empty means scan all.",
                },
                "severity_threshold": {
                    "type": "string",
                    "enum": ["critical", "warning", "info", "all"],
                    "description": "Minimum severity to include in results.",
                    "default": "all",
                },
            },
            "required": [],
        },
    },
    {
        "name": "pybinaryguard_check",
        "description": (
            "Check a specific installed Python package for binary "
            "compatibility issues with the current system."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package name to check (e.g. 'torch', 'numpy').",
                },
            },
            "required": ["package"],
        },
    },
    {
        "name": "pybinaryguard_simulate_install",
        "description": (
            "Predict whether a package will be compatible with the current "
            "system BEFORE installing it. Checks architecture, GLIBC version, "
            "CUDA compatibility, and Python ABI from the wheel filename."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package_spec": {
                    "type": "string",
                    "description": (
                        "Package specifier. Can be a package name ('torch'), "
                        "a version pin ('torch==2.4.0'), or a wheel filename "
                        "('torch-2.4.0+cu124-cp312-cp312-manylinux_2_17_x86_64.whl')."
                    ),
                },
            },
            "required": ["package_spec"],
        },
    },
    {
        "name": "pybinaryguard_doctor",
        "description": (
            "Diagnose a specific error message. Paste a Python traceback or "
            "error string and get a structured diagnosis with fix plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "error_message": {
                    "type": "string",
                    "description": "The error message or traceback to diagnose.",
                },
                "package": {
                    "type": "string",
                    "description": "Optional package name related to the error.",
                },
            },
            "required": ["error_message"],
        },
    },
    {
        "name": "pybinaryguard_profile",
        "description": (
            "Get the current system's binary compatibility profile: Python "
            "version, architecture, GLIBC, CUDA stack, GPU info, and board "
            "detection. Use this to understand the environment."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def get_tool_descriptors(format: str = "openai") -> List[Dict[str, Any]]:
    """Get tool descriptors in the specified format.

    Parameters
    ----------
    format:
        Output format: "openai", "mcp", or "json_schema".

    Returns
    -------
    List[Dict[str, Any]]
        Tool descriptors ready for agent framework registration.
    """
    if format == "openai":
        return _to_openai_format()
    elif format == "mcp":
        return _to_mcp_format()
    elif format == "json_schema":
        return _TOOLS
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'openai', 'mcp', or 'json_schema'.")


def export_tool_schema(format: str = "openai") -> str:
    """Export tool schemas as a JSON string.

    Parameters
    ----------
    format:
        Output format: "openai", "mcp", or "json_schema".

    Returns
    -------
    str
        JSON string of tool descriptors.
    """
    descriptors = get_tool_descriptors(format)
    return json.dumps(descriptors, indent=2)


def _to_openai_format() -> List[Dict[str, Any]]:
    """Convert to OpenAI function calling format."""
    tools = []
    for tool in _TOOLS:
        tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return tools


def _to_mcp_format() -> List[Dict[str, Any]]:
    """Convert to MCP (Model Context Protocol) tool format."""
    tools = []
    for tool in _TOOLS:
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["parameters"],
        })
    return tools
