"""Basic usage example — full environment scan."""

import pybinaryguard


def main():
    # Run a full environment scan
    report = pybinaryguard.scan()

    # Print summary
    print(f"Health Score: {report.health_score}/100 ({report.health_label})")
    print(f"Packages Scanned: {report.packages_scanned}")
    print(f"Total Issues: {report.total_findings}")
    print(f"  Critical: {report.critical_count}")
    print(f"  Warnings: {report.warning_count}")
    print()

    # Print each finding
    for finding in report.findings:
        icon = {"critical": "[!]", "warning": "[~]", "info": "[i]"}
        severity = finding.severity.value
        print(f"{icon.get(severity, '[ ]')} [{severity.upper()}] {finding.rule_id}")
        print(f"    Package: {finding.package}")
        print(f"    Message: {finding.message}")
        if finding.suggestion:
            print(f"    Fix: {finding.suggestion}")
        print()


if __name__ == "__main__":
    main()
