"""CI/CD integration example — use as a pre-deploy gate."""

import sys

import pybinaryguard
from pybinaryguard import ScanMode


def ci_check(strict: bool = True) -> int:
    """Run binary compatibility check as CI gate.

    Args:
        strict: If True, fail on any warnings. If False, only fail on critical.

    Returns:
        Exit code: 0 = pass, 1 = fail.
    """
    # Fast scan for CI speed
    report = pybinaryguard.scan(scan_mode=ScanMode.FAST)

    print(f"PyBinaryGuard CI Check")
    print(f"  Health Score: {report.health_score}/100")
    print(f"  Critical: {report.critical_count}")
    print(f"  Warnings: {report.warning_count}")

    if report.critical_count > 0:
        print(f"\nFAILED: {report.critical_count} critical issues found")
        for f in report.findings:
            if f.severity.value == "critical":
                print(f"  - {f.rule_id}: {f.message}")
        return 1

    if strict and report.warning_count > 0:
        print(f"\nFAILED (strict mode): {report.warning_count} warnings found")
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    sys.exit(ci_check(strict=strict))
