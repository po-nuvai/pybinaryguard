# PyBinaryGuard Usage Guide

## Installation

```bash
pip install pybinaryguard

# With full ELF analysis support
pip install pybinaryguard[full]

# For development
pip install pybinaryguard[dev]
```

## CLI Quick Start

### Full Environment Scan

Scan all installed Python packages for binary compatibility issues:

```bash
pybinaryguard scan
```

Output includes:
- Health score (0-100) with category breakdown
- Critical issues (must fix)
- Warnings (should fix)
- Info (good to know)

### Check a Specific Package

```bash
pybinaryguard check torch
pybinaryguard check numpy
pybinaryguard check opencv-python
```

### System Profile

View your system's binary compatibility profile:

```bash
pybinaryguard profile
```

Shows: Python version, CPU architecture, GLIBC version, CUDA toolkit, GPU compute capability, OS details.

### Doctor Mode

Diagnose a specific error message:

```bash
pybinaryguard doctor --error "GLIBC_2.34 not found"
pybinaryguard doctor --error "Illegal instruction"
pybinaryguard doctor --package torch
```

### Inspect a Wheel File

Check a `.whl` file before installing:

```bash
pybinaryguard inspect torch-2.4.0-cp311-cp311-manylinux1_x86_64.whl
```

### Scan Modes

```bash
# Fast: metadata only, skips binary analysis (<1 second)
pybinaryguard scan --fast

# Standard: full binary analysis (default, ~3 seconds)
pybinaryguard scan

# Deep: symbol resolution + SHA256 hash verification (~10 seconds)
pybinaryguard scan --deep
```

### Output Formats

```bash
# Human-readable table (default)
pybinaryguard scan --format table

# Machine-readable JSON
pybinaryguard scan --format json

# Minimal one-line-per-finding
pybinaryguard scan --format minimal
```

### Filtering

```bash
# Only show critical issues
pybinaryguard scan --severity critical

# Ignore specific rules
pybinaryguard scan --ignore CUDA_MINOR_MISMATCH NUMPY_ABI_WARNING

# Quiet mode (critical only)
pybinaryguard scan -q

# Verbose mode (all technical details)
pybinaryguard scan -v
```

### CI/CD Integration

```bash
# CI mode: minimal output, strict exit codes, no color
pybinaryguard scan --ci
```

Exit codes:
- `0` — All clear
- `1` — Warnings found
- `2` — Critical issues found
- `3` — Scanner error

## Python API

### Basic Scanning

```python
import pybinaryguard

# Full environment scan
report = pybinaryguard.scan()
print(f"Health Score: {report.health_score}/100")
print(f"Health Label: {report.health_label}")
print(f"Total Issues: {report.total_findings}")
print(f"Critical: {report.critical_count}")
print(f"Warnings: {report.warning_count}")

# Iterate findings
for finding in report.findings:
    print(f"[{finding.severity.value}] {finding.rule_id}")
    print(f"  Package: {finding.package}")
    print(f"  Message: {finding.message}")
    if finding.suggestion:
        print(f"  Fix: {finding.suggestion}")
```

### Check a Single Package

```python
findings = pybinaryguard.check("torch")
for f in findings:
    print(f"{f.severity.value}: {f.message}")
```

### System Profile

```python
profile = pybinaryguard.profile()
print(f"Python: {profile.python_version}")
print(f"Architecture: {profile.architecture}")
print(f"GLIBC: {profile.glibc_version}")
print(f"CUDA: {profile.cuda_version}")
print(f"GPU: {profile.gpu_name}")
print(f"Compute Capability: {profile.compute_capability}")
```

### Inspect a File

```python
findings = pybinaryguard.inspect("path/to/wheel.whl")
```

### Scan Modes

```python
from pybinaryguard import ScanMode

# Fast scan
report = pybinaryguard.scan(scan_mode=ScanMode.FAST)

# Deep scan
report = pybinaryguard.scan(scan_mode=ScanMode.DEEP)
```

## Agent SDK

### Structured Scan

```python
from pybinaryguard.agent import scan

report = scan()

# JSON-serializable output
data = report.to_dict()

# Pre-classified actions
for action in report.safe_actions:
    print(f"Auto-fix: {action.command}")  # e.g., pip install --upgrade numpy

for action in report.review_actions:
    print(f"Needs review: {action.command}")

for action in report.dangerous_actions:
    print(f"Human only: {action.command}")
```

### Pre-Install Simulation

```python
from pybinaryguard.agent import simulate_install

result = simulate_install("torch==2.4.0+cu124")

if result.predicted_compatible:
    print("Safe to install!")
else:
    print(f"Will fail: {result.blockers}")
    print(f"Confidence: {result.confidence:.0%}")
```

### Error Diagnosis

```python
from pybinaryguard.agent import doctor

dx = doctor("GLIBC_2.34 not found")
print(dx.diagnosis)
print(dx.fix_plan)
print(f"Auto-fixable: {dx.auto_fix_safe}")
```

### Tool Schema Export

```python
from pybinaryguard.agent import export_tool_schema

# For OpenAI function calling
schema = export_tool_schema(format="openai")

# For MCP (Model Context Protocol)
schema = export_tool_schema(format="mcp")
```

### Runtime Import Guard

```python
from pybinaryguard.agent.guard import guarded_imports

with guarded_imports() as guard:
    import torch

for error in guard.captured_errors:
    print(error["category"])
    print(error["diagnosis"])
```

## Environment Snapshots

### Create a Snapshot

```bash
# Save to file
pybinaryguard snapshot -o env.lock.json

# Print to stdout
pybinaryguard snapshot
```

### Verify Against Snapshot

```bash
# Full verification
pybinaryguard verify env.lock.json

# Skip hash checks (faster)
pybinaryguard verify env.lock.json --no-hashes

# Skip compatibility checks
pybinaryguard verify env.lock.json --no-compat-checks
```

## Plugin Development

Create custom plugins by implementing probes, analyzers, or rules:

```python
# my_plugin.py
from pybinaryguard.rules.base import Rule
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.enums import Severity

class MyCustomRule(Rule):
    rule_id = "MY_CUSTOM_CHECK"
    description = "Check for my custom condition"
    severity = Severity.WARNING

    def is_applicable(self, profile):
        return True

    def evaluate(self, profile, packages):
        findings = []
        for pkg in packages:
            if some_condition(pkg):
                findings.append(Finding(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    package=pkg.name,
                    message="Custom issue detected",
                    suggestion="How to fix it",
                ))
        return findings

def register(registry):
    registry.add_rule(MyCustomRule())
```

Register via `pyproject.toml`:

```toml
[project.entry-points."pybinaryguard.plugins"]
my_plugin = "my_plugin:register"
```
