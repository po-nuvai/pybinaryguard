"""Helper functions for filtering, sorting, grouping, and deduplicating findings."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Sequence, Tuple

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding

# Canonical ordering: CRITICAL (most severe) first, PASSED last.
_SEVERITY_ORDER: Dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.INFO: 2,
    Severity.PASSED: 3,
}


def filter_findings(
    findings: Sequence[Finding],
    min_severity: Severity,
) -> List[Finding]:
    """Return findings whose severity is at or above *min_severity*.

    Severity is ordered CRITICAL > WARNING > INFO > PASSED, so calling with
    ``min_severity=Severity.WARNING`` keeps CRITICAL and WARNING, dropping
    INFO and PASSED.

    Parameters
    ----------
    findings:
        The collection of findings to filter.
    min_severity:
        The lowest severity to include.  Findings with a severity that
        compares *less than or equal to* ``min_severity`` (i.e. more severe
        or equally severe) are included.

    Returns
    -------
    List[Finding]
        A new list containing only the findings that pass the threshold.
    """
    if not isinstance(min_severity, Severity):
        raise TypeError(
            f"min_severity must be a Severity enum member, got {type(min_severity).__name__}"
        )

    threshold = _SEVERITY_ORDER[min_severity]
    return [f for f in findings if _SEVERITY_ORDER.get(f.severity, 99) <= threshold]


def sort_findings(findings: Sequence[Finding]) -> List[Finding]:
    """Sort findings by severity (CRITICAL first), then alphabetically by package name.

    Findings without a package name sort after those with one.

    Parameters
    ----------
    findings:
        The collection of findings to sort.

    Returns
    -------
    List[Finding]
        A new list sorted by severity (descending) then package name (ascending).
    """
    return sorted(
        findings,
        key=lambda f: (
            _SEVERITY_ORDER.get(f.severity, 99),
            f.package or "\uffff",  # None sorts last
        ),
    )


def group_by_package(findings: Sequence[Finding]) -> Dict[str, List[Finding]]:
    """Group findings by their package name.

    Findings with ``package=None`` are grouped under the key
    ``"<unknown>"`` so that every key is a usable string.

    Parameters
    ----------
    findings:
        The collection of findings to group.

    Returns
    -------
    Dict[str, List[Finding]]
        A mapping from package name to the list of findings for that package.
    """
    groups: Dict[str, List[Finding]] = {}
    for finding in findings:
        key = finding.package or "<unknown>"
        groups.setdefault(key, []).append(finding)
    return groups


def group_by_severity(findings: Sequence[Finding]) -> Dict[Severity, List[Finding]]:
    """Group findings by their severity level.

    The returned dict only contains keys for severity levels that have at
    least one finding.

    Parameters
    ----------
    findings:
        The collection of findings to group.

    Returns
    -------
    Dict[Severity, List[Finding]]
        A mapping from severity level to the list of findings at that level.
    """
    groups: Dict[Severity, List[Finding]] = {}
    for finding in findings:
        groups.setdefault(finding.severity, []).append(finding)
    return groups


def deduplicate_findings(findings: Sequence[Finding]) -> List[Finding]:
    """Remove duplicate findings that share the same rule_id and package.

    When duplicates exist, the first occurrence (in iteration order) is
    kept.  Two findings are considered duplicates when both their
    ``rule_id`` and ``package`` fields match exactly (including ``None``).

    Parameters
    ----------
    findings:
        The collection of findings to deduplicate.

    Returns
    -------
    List[Finding]
        A new list with duplicates removed, preserving the order of first
        occurrences.
    """
    seen: Set[Tuple[str, Optional[str]]] = set()
    result: List[Finding] = []
    for finding in findings:
        key = (finding.rule_id, finding.package)
        if key not in seen:
            seen.add(key)
            result.append(finding)
    return result
