"""Agent SDK usage example — structured output for AI agents."""

from pybinaryguard.agent import scan, check, simulate_install, doctor, export_tool_schema


def demo_scan():
    """Full scan with action recommendations."""
    print("=" * 60)
    print("AGENT SCAN")
    print("=" * 60)

    report = scan()

    print(f"Health: {report.health_score}/100")
    print(f"Risk Level: {report.risk_level}")
    print(f"Issues: {len(report.issues)}")

    if report.safe_actions:
        print(f"\nSafe Actions (auto-executable):")
        for action in report.safe_actions:
            print(f"  $ {action.command}")

    if report.review_actions:
        print(f"\nReview Actions (needs confirmation):")
        for action in report.review_actions:
            print(f"  $ {action.command}  # {action.reason}")

    if report.dangerous_actions:
        print(f"\nDangerous Actions (human only):")
        for action in report.dangerous_actions:
            print(f"  $ {action.command}  # {action.reason}")

    # JSON-serializable dict for API responses
    data = report.to_dict()
    print(f"\nJSON keys: {list(data.keys())}")


def demo_simulate():
    """Pre-install compatibility check."""
    print("\n" + "=" * 60)
    print("SIMULATE INSTALL")
    print("=" * 60)

    specs = [
        "torch==2.4.0+cu124",
        "numpy-1.26.0-cp311-cp311-manylinux1_x86_64.whl",
        "tensorflow==2.15.0",
    ]

    for spec in specs:
        result = simulate_install(spec)
        status = "COMPATIBLE" if result.predicted_compatible else "INCOMPATIBLE"
        print(f"\n  {spec}")
        print(f"    Status: {status}")
        print(f"    Confidence: {result.confidence:.0%}")
        if result.blockers:
            for b in result.blockers:
                print(f"    Blocker: {b['message']}")


def demo_doctor():
    """Error diagnosis."""
    print("\n" + "=" * 60)
    print("DOCTOR")
    print("=" * 60)

    errors = [
        "GLIBC_2.34 not found",
        "Illegal instruction (core dumped)",
        "libcudart.so.12: cannot open shared object file",
    ]

    for error in errors:
        dx = doctor(error)
        print(f"\n  Error: {error}")
        print(f"    Diagnosis: {dx.diagnosis}")
        if dx.fix_plan:
            print(f"    Fix: {dx.fix_plan}")
        print(f"    Auto-fixable: {dx.auto_fix_safe}")


def demo_schema():
    """Tool schema export."""
    print("\n" + "=" * 60)
    print("TOOL SCHEMA (OpenAI format)")
    print("=" * 60)

    schema = export_tool_schema(format="openai")
    # Truncate for display
    print(schema[:500] + "...")


if __name__ == "__main__":
    demo_scan()
    demo_simulate()
    demo_doctor()
    demo_schema()
