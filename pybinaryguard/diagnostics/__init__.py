"""Diagnostics subsystem -- error diagnosis, explanation, and fix suggestions."""

from pybinaryguard.diagnostics.explainer import diagnose_error, explain_finding
from pybinaryguard.diagnostics.findings import filter_findings, sort_findings
from pybinaryguard.diagnostics.suggestions import suggest_fix

__all__ = [
    "diagnose_error",
    "explain_finding",
    "filter_findings",
    "sort_findings",
    "suggest_fix",
]
