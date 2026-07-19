<p align="center">
  <a href="https://pypi.org/project/pybinaryguard/"><img src="https://img.shields.io/pypi/v/pybinaryguard?style=for-the-badge&logo=pypi&logoColor=white&color=purple" alt="PyPI"></a>
  <a href="https://pypi.org/project/pybinaryguard/"><img src="https://img.shields.io/pypi/dm/pybinaryguard?style=for-the-badge&color=blue" alt="Downloads"></a>
  <a href="https://github.com/po-nuvai/pybinaryguard/stargazers"><img src="https://img.shields.io/github/stars/po-nuvai/pybinaryguard?style=for-the-badge&logo=github&color=yellow" alt="Stars"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/platform-linux-orange?style=for-the-badge&logo=linux&logoColor=white" alt="Linux">
  <img src="https://img.shields.io/badge/tests-505%20passing-brightgreen?style=for-the-badge" alt="505 Tests">
</p>

<h1 align="center">PyBinaryGuard</h1>

<p align="center">
  <strong>Binary Compatibility Intelligence for Python</strong><br>
  Detect incompatibilities <em>before</em> they crash your program.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#why-pybinaryguard">Why?</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#cli-usage">CLI</a> &bull;
  <a href="#python-api">Python API</a> &bull;
  <a href="#agent-sdk">Agent SDK</a> &bull;
  <a href="#architecture">Architecture</a>
</p>

---

## Origin Story

Back in college, I was working on real-time object detection on an **NVIDIA Jetson TX2** — computer vision, image processing, and ML models for live detection. I spent weeks writing the code. Tested the logic. Verified every layer, every weight, every preprocessing step. The code had **zero errors**.

Then I hit run.

```
Illegal instruction (core dumped)
```

That's it. No traceback. No helpful message. Just — *"instruction unclear, core dumped."*

I stared at the screen. I checked the code again. **Nothing was wrong.** I rewrote parts of it. Same crash. I tried different package versions. Same crash. I searched Stack Overflow for hours. Nothing worked.

This went on for **days**. Every single day, the same cryptic error. I started questioning my own code, my understanding of Python, everything. It was pure rage.

Then one day I finally figured it out — it wasn't my code at all. The pip packages I installed were compiled for a different architecture. The CUDA toolkit version didn't match what PyTorch expected. The GLIBC on the Jetson was too old for the prebuilt wheels. The binaries simply didn't belong on that hardware.

**My code was perfect. The binaries were incompatible.**

And the worst part? There was no tool to tell me this. No `pip check` that catches binary mismatches. No scanner that says *"hey, this .so file needs GLIBC 2.34 but you only have 2.27."* Nothing.

I told myself: if I ever get the chance, I'll build the tool I wish I had during those sleepless nights on the Jetson. Something that looks at your system — your CPU, your GPU, your GLIBC, your CUDA — and tells you **what's actually wrong** before you waste another day blaming your own code.

That tool is **PyBinaryGuard**.

---

## The Problem

You install a Python package. Your code is correct. Your machine is fine. **It still crashes.**

```
ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.34' not found
```

```
Illegal instruction (core dumped)
```

```
OSError: libcudart.so.12: cannot open shared object file
```

These aren't bugs in your code. They're **binary-level incompatibilities** between compiled C/C++ libraries inside Python packages and your system's hardware, OS, or drivers. No existing tool catches them before runtime.

**PyBinaryGuard does.**

---

## Why PyBinaryGuard?

| Tool | What it does | What it misses |
|------|-------------|----------------|
| `pip check` | Version conflicts | Binary/ABI compatibility |
| `ldd` | Shared library links | Python package context |
| `nvidia-smi` | GPU info | Cross-package CUDA conflicts |
| `file` | ELF metadata | Compatibility analysis |
| **PyBinaryGuard** | **All of the above, unified** | -- |

PyBinaryGuard is the **first tool** that correlates your Python version, CPU architecture, GLIBC version, CUDA toolkit, GPU compute capability, and installed package binaries into a single compatibility verdict.

---

## Quick Start

### Install

```bash
# Recommended — isolated CLI install (Ubuntu 24.04+, Debian 12+, etc.)
pipx install pybinaryguard

# Or per-user install
pip install --user pybinaryguard

# Or classic pip (inside a venv)
pip install pybinaryguard
```

> **Note:** On modern Linux distros (Ubuntu 24.04+, Debian 12+), system-wide
> `pip install` is blocked by PEP 668. Use `pipx` or a virtual environment.

### Use

```bash
# Run a full scan
pybinaryguard scan

# Check a specific package
pybinaryguard check torch

# Fast scan (metadata only, <1 second)
pybinaryguard scan --fast

# Deep scan (full symbol resolution + hash verification)
pybinaryguard scan --deep
```

### Install from source (for development)

```bash
git clone https://github.com/po-nuvai/pybinaryguard.git
cd pybinaryguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Features

### Core Scanner
- **System Profiling** -- Detects Python version, CPU architecture (x86_64/aarch64/armv7l), GLIBC version, CUDA toolkit, GPU compute capability, container environment
- **Binary Analysis** -- Inspects `.so` shared objects inside installed packages using ELF header parsing and symbol resolution
- **20+ Built-in Rules** -- Covers GLIBC version requirements, CUDA ABI mismatches, CPU instruction set conflicts, architecture mismatches, NumPy ABI breaks, container-specific issues, and more
- **Plugin System** -- Extensible with custom probes, analyzers, and rules for Jetson, TensorRT, OpenCV, GStreamer

### Health Scoring v2
Multi-dimensional weighted scoring across 4 categories:

| Category | Weight | What it measures |
|----------|--------|------------------|
| Binary Stability | 35% | GLIBC, ELF, ABI issues |
| GPU Compatibility | 30% | CUDA, compute capability, driver mismatches |
| Dependency Health | 25% | Version conflicts, missing libraries |
| Platform Risk | 10% | Architecture, container, OS-specific issues |

Weights auto-adjust based on your system (e.g., no GPU = GPU weight redistributed).

### Scan Modes

| Mode | Speed | Depth | Use case |
|------|-------|-------|----------|
| `--fast` | <1s | Metadata only, skips ELF analysis | CI pipelines, quick checks |
| *(default)* | ~3s | Full binary analysis | Development workflow |
| `--deep` | ~10s | Symbol resolution + SHA256 hashes | Security audits, production deploys |

### Board Profile Engine
Built-in support for embedded/edge platforms:
- **NVIDIA Jetson** (Nano, TX2, Xavier, Orin)
- **Raspberry Pi** (3B, 4B, 5, Zero 2W)
- Custom board profiles via plugin system

### AI Framework Inspection
Specialized checks for ML/AI stacks:
- PyTorch CUDA ABI compatibility
- PyTorch + TorchVision version matrix
- TensorFlow compute capability requirements
- TensorRT version compatibility
- ONNX Runtime execution provider validation

### Predictive Failure Engine
Predicts runtime failures before they happen:
- Import error prediction based on dependency chain analysis
- Unresolved symbol detection
- Cross-package ABI conflict detection

### Environment Snapshots
Lock and verify your binary environment:

```bash
# Create a snapshot
pybinaryguard snapshot -o env.lock.json

# Verify against snapshot on another machine
pybinaryguard verify env.lock.json
```

---

## CLI Usage

```
pybinaryguard <command> [options]

Commands:
  scan                Full environment scan
  check <package>     Check a specific package
  profile             Show system profile
  doctor              Interactive troubleshooting
  inspect <file>      Analyse a .whl or .so file
  snapshot            Create environment snapshot
  verify <lockfile>   Verify against snapshot
  simulate <spec>     Predict compatibility before install
  export-tool-schema  Export agent tool schema

Global Options:
  --format {table,json,minimal}   Output format (default: table)
  --severity {critical,warning,info,all}
  --fast / --deep                 Scan depth
  --ci                            CI mode (minimal + strict exit codes)
  --ignore RULE_ID [...]          Rules to skip
  --timeout SECONDS               Max scan time (default: 30)
  --no-color                      Disable coloured output
  -v, --verbose                   Show technical details
  -q, --quiet                     Critical findings only
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All clear |
| 1 | Warnings found |
| 2 | Critical issues found |
| 3 | Scanner error |

### CI/CD Integration

```yaml
# GitHub Actions
- name: Binary compatibility check
  run: |
    pip install pybinaryguard
    pybinaryguard scan --ci
```

```Dockerfile
# Docker health check
HEALTHCHECK CMD pybinaryguard scan --fast --ci || exit 1
```

---

## Python API

```python
import pybinaryguard

# Full environment scan
report = pybinaryguard.scan()
print(f"Health: {report.health_score}/100")
print(f"Issues: {report.total_findings}")

for finding in report.findings:
    print(f"[{finding.severity}] {finding.rule_id}: {finding.message}")
    if finding.suggestion:
        print(f"  Fix: {finding.suggestion}")

# Check a single package
findings = pybinaryguard.check("torch")

# Get system profile
profile = pybinaryguard.profile()
print(f"Python: {profile.python_version}")
print(f"GLIBC: {profile.glibc_version}")
print(f"Arch: {profile.architecture}")
print(f"CUDA: {profile.cuda_version}")

# Inspect a wheel file before installing
findings = pybinaryguard.inspect("torch-2.4.0-cp311-cp311-manylinux1_x86_64.whl")
```

---

## Agent SDK

PyBinaryGuard is **agent-native** -- designed for AI agents and automation pipelines to consume directly.

### Structured Output

```python
from pybinaryguard.agent import scan, check, simulate_install, doctor

# Returns machine-readable ActionableReport
report = scan()
report.to_dict()  # JSON-serializable

# Classified actions with safety levels
report.safe_actions      # Auto-executable (e.g., pip install --upgrade)
report.review_actions    # Needs human confirmation
report.dangerous_actions # Human-only (e.g., system library changes)

# Pre-install simulation
sim = simulate_install("torch==2.4.0+cu124")
sim.predicted_compatible  # True/False
sim.confidence            # 0.0-1.0
sim.blockers              # List of blocking issues

# Error diagnosis
dx = doctor("GLIBC_2.34 not found")
dx.diagnosis       # What went wrong
dx.fix_plan        # Step-by-step fix
dx.auto_fix_safe   # Can an agent fix this automatically?
```

### Tool Schema Export

Register PyBinaryGuard as a tool in any agent framework:

```python
from pybinaryguard.agent import export_tool_schema

# OpenAI function calling format
schema = export_tool_schema(format="openai")

# MCP (Model Context Protocol) format
schema = export_tool_schema(format="mcp")

# Generic JSON Schema
schema = export_tool_schema(format="json_schema")
```

```bash
# CLI export
pybinaryguard export-tool-schema --schema-format openai
pybinaryguard export-tool-schema --schema-format mcp
```

### One-Liner Agent Registration

```python
from pybinaryguard.agent import as_agent_tool

# Returns {schema: ..., handlers: {scan: fn, check: fn, ...}}
tool = as_agent_tool()
```

### Runtime Import Guard

Capture and diagnose import failures in real-time:

```python
from pybinaryguard.agent.guard import guarded_imports

with guarded_imports() as guard:
    import torch  # If this fails, guard captures structured diagnostics

for error in guard.captured_errors:
    print(error["category"])   # e.g., "glibc_mismatch"
    print(error["diagnosis"])  # Human-readable explanation
```

---

## How It Works (UML Diagrams)

### Core Scan Pipeline

The entire library follows a 4-phase pipeline: **Probe → Analyze → Evaluate → Report**.

```mermaid
flowchart TB
    subgraph INPUT["INPUT"]
        USER["User / Agent / CI"]
    end

    subgraph PHASE1["PHASE 1: PROBE — Collect System Info"]
        direction LR
        PP["PythonProbe<br/>Python version, ABI"]
        CP["CpuProbe<br/>Architecture, ISA"]
        GP["GlibcProbe<br/>GLIBC version"]
        GPU["GpuProbe<br/>CUDA, GPU, compute cap"]
        OP["OsProbe<br/>OS, container detect"]
        LP["LibraryProbe<br/>System shared libs"]
        BP["BoardProbe<br/>Jetson, RPi detect"]
    end

    subgraph PROFILE["SystemProfile"]
        SP["python_version<br/>architecture<br/>glibc_version<br/>cuda_version<br/>gpu_name<br/>compute_capability<br/>os_name<br/>is_container<br/>..."]
    end

    subgraph PHASE2["PHASE 2: ANALYZE — Inspect Package Binaries"]
        direction LR
        WA["WheelAnalyzer<br/>WHEEL metadata, tags"]
        EA["ELF Analyzer<br/>Headers, symbols, DT_NEEDED"]
        SA["SymbolAnalyzer<br/>Unresolved symbols"]
        DA["DependencyAnalyzer<br/>Dependency chain"]
    end

    subgraph PACKAGES["PackageBinaryInfo[]"]
        PKG["package_name<br/>shared_objects[]<br/>wheel_tags[]<br/>required_glibc<br/>target_architecture<br/>..."]
    end

    subgraph PHASE3["PHASE 3: EVALUATE — Run Compatibility Rules"]
        direction LR
        RE["RuleEngine"]
        R1["GLIBC Rules"]
        R2["CUDA Rules"]
        R3["Arch Rules"]
        R4["CPU Rules"]
        R5["NumPy ABI Rules"]
        R6["Container Rules"]
        R7["Framework Rules<br/>PyTorch, TF, TRT, ONNX"]
        R8["Board Rules<br/>Jetson, RPi"]
        R9["Predictive Rules"]
    end

    subgraph FINDINGS["Finding[]"]
        F["rule_id<br/>severity<br/>package<br/>message<br/>suggestion<br/>confidence"]
    end

    subgraph PHASE4["PHASE 4: REPORT — Score & Format"]
        direction LR
        HS["HealthScoreV2<br/>4-category weighted"]
        FMT["Formatter<br/>Table / JSON / Minimal"]
        AGENT["Agent SDK<br/>ActionRecommender"]
    end

    subgraph OUTPUT["OUTPUT"]
        SR["ScanReport<br/>health_score: 73/100<br/>findings: [...]<br/>score_breakdown: {...}"]
    end

    USER -->|"scan() / check() / CLI"| PHASE1
    PP & CP & GP & GPU & OP & LP & BP -->|"parallel threads"| PROFILE
    PROFILE --> PHASE2
    WA & EA & SA & DA -->|"per package"| PACKAGES
    PROFILE --> PHASE3
    PACKAGES --> PHASE3
    RE --> R1 & R2 & R3 & R4 & R5 & R6 & R7 & R8 & R9
    PHASE3 --> FINDINGS
    FINDINGS --> PHASE4
    HS & FMT & AGENT --> OUTPUT
    OUTPUT --> USER
```

### Sequence Diagram — Full Scan Flow

Step-by-step execution when a user runs `pybinaryguard scan`:

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI (main.py)
    participant CMD as Commands (commands.py)
    participant Scanner
    participant Probes as Probes (7 probes)
    participant Analyzers as ELF/Wheel Analyzers
    participant Rules as Rule Engine (20+ rules)
    participant Scoring as HealthScoreV2
    participant Formatter as Table/JSON Formatter

    User->>CLI: pybinaryguard scan
    CLI->>CMD: dispatch(args)
    CMD->>Scanner: Scanner(scan_mode, ...).run()

    Note over Scanner: PHASE 1: PROBE
    Scanner->>Probes: run all probes (parallel threads)
    Probes-->>Scanner: merged dict
    Scanner->>Scanner: build SystemProfile

    Note over Scanner: PHASE 2: ANALYZE
    Scanner->>Scanner: walk site-packages dirs
    loop Each package with .dist-info
        Scanner->>Analyzers: parse WHEEL tags
        alt STANDARD or DEEP mode
            Scanner->>Analyzers: parse ELF headers (.so files)
            Analyzers-->>Scanner: SharedObjectInfo[]
        end
        alt DEEP mode only
            Scanner->>Scanner: compute SHA256 hashes
        end
    end
    Scanner-->>Scanner: PackageBinaryInfo[]

    Note over Scanner: PHASE 3: EVALUATE
    Scanner->>Rules: RuleEngine.with_builtin_rules()
    loop Each rule
        Rules->>Rules: is_applicable(profile)?
        alt applicable
            Rules->>Rules: evaluate(profile, packages)
            Rules-->>Scanner: Finding[]
        end
    end

    Note over Scanner: PHASE 4: REPORT
    Scanner->>Scoring: compute_health_score(findings)
    Scoring-->>Scanner: ScoreBreakdown
    Scanner->>Scanner: deduplicate, filter, sort
    Scanner-->>CMD: ScanReport

    CMD->>Formatter: format_scan(report, profile)
    Formatter-->>CMD: formatted string
    CMD-->>CLI: exit code (0/1/2)
    CLI-->>User: printed output
```

### Health Scoring Model

How the multi-dimensional health score is calculated:

```mermaid
flowchart LR
    subgraph FINDINGS["All Findings"]
        F1["GLIBC_VERSION_MISMATCH<br/>severity: CRITICAL"]
        F2["CUDA_MINOR_MISMATCH<br/>severity: WARNING"]
        F3["NUMPY_ABI_MISMATCH<br/>severity: WARNING"]
        F4["CONTAINER_GLIBC_OLD<br/>severity: INFO"]
    end

    subgraph CLASSIFY["Classify by Rule ID Prefix"]
        C1["binary_stability<br/>GLIBC_, ARCH_, ELF_, ..."]
        C2["gpu_compat<br/>CUDA_, PYTORCH_CUDA_, ..."]
        C3["dependency_health<br/>NUMPY_ABI_, MISSING_, ..."]
        C4["platform_risk<br/>CONTAINER_, BOARD_, ..."]
    end

    subgraph SCORE["Score Each Category"]
        S1["Binary: 70/100<br/>weight: 0.35"]
        S2["GPU: 90/100<br/>weight: 0.30"]
        S3["Deps: 90/100<br/>weight: 0.25"]
        S4["Platform: 98/100<br/>weight: 0.10"]
    end

    subgraph TOTAL["Weighted Total"]
        T["70x0.35 + 90x0.30 + 90x0.25 + 98x0.10<br/>= 24.5 + 27.0 + 22.5 + 9.8<br/>= <strong>83.8 / 100</strong>"]
    end

    F1 --> C1
    F2 --> C2
    F3 --> C3
    F4 --> C4

    C1 --> S1
    C2 --> S2
    C3 --> S3
    C4 --> S4

    S1 & S2 & S3 & S4 --> T
```

### Agent SDK Flow

How AI agents interact with PyBinaryGuard:

```mermaid
flowchart TB
    subgraph AGENT["AI Agent (GPT, LangChain, AutoGen, CrewAI, etc.)"]
        A1["1. Register tool via schema"]
        A2["2. Call scan/check/simulate/doctor"]
        A3["3. Read structured result"]
        A4["4. Execute safe_actions automatically"]
        A5["5. Ask human for review_actions"]
    end

    subgraph SDK["PyBinaryGuard Agent SDK"]
        SCHEMA["export_tool_schema()<br/>OpenAI / MCP / JSON Schema"]
        SCAN["scan() → ActionableReport"]
        CHECK["check('torch') → AgentCheckResult"]
        SIM["simulate_install('torch==2.4.0+cu124')<br/>→ AgentSimulateResult"]
        DOC["doctor('GLIBC_2.34 not found')<br/>→ AgentDoctorResult"]
        REC["ActionRecommender<br/>classify actions by safety"]
    end

    subgraph SAFETY["Action Safety Classification"]
        SAFE["SAFE (auto-execute)<br/>pip install --upgrade numpy<br/>pip install torch==2.3.0"]
        REVIEW["REVIEW (confirm first)<br/>pip install --force-reinstall ...<br/>pip install package[cuda12]"]
        DANGER["DANGEROUS (human only)<br/>apt install libcuda1<br/>System library changes"]
    end

    A1 -->|"schema = export_tool_schema('openai')"| SCHEMA
    A2 --> SCAN & CHECK & SIM & DOC
    SCAN --> REC
    REC --> SAFE & REVIEW & DANGER
    SAFE --> A4
    REVIEW --> A5
    DANGER -->|"flag to human"| A5
    A3 -->|"result.to_dict() → JSON"| A2
```

### Scan Mode Comparison

```mermaid
flowchart LR
    subgraph FAST["--fast  (<1 second)"]
        F1_["Read WHEEL metadata"]
        F2_["Check .so file existence"]
        F3_["Run metadata-only rules"]
        F4_["Skip ELF parsing entirely"]
    end

    subgraph STD["default  (~3 seconds)"]
        S1_["Read WHEEL metadata"]
        S2_["Parse ELF headers"]
        S3_["Extract DT_NEEDED, symbols"]
        S4_["Run ALL rules"]
    end

    subgraph DEEP["--deep  (~10 seconds)"]
        D1_["Read WHEEL metadata"]
        D2_["Parse ELF headers"]
        D3_["Full symbol resolution"]
        D4_["SHA256 hash every .so"]
        D5_["Run ALL rules"]
    end

    F1_ --> F2_ --> F3_ --> F4_
    S1_ --> S2_ --> S3_ --> S4_
    D1_ --> D2_ --> D3_ --> D4_ --> D5_
```

### Class Diagram — Core Data Model

```mermaid
classDiagram
    class SystemProfile {
        +str python_version
        +str architecture
        +str glibc_version
        +str cuda_version
        +str gpu_name
        +float compute_capability
        +str os_name
        +bool is_container
        +bool gpu_available
        +bool is_embedded_board
        +list site_packages_paths
    }

    class PackageBinaryInfo {
        +str package_name
        +str package_version
        +str install_path
        +list~SharedObjectInfo~ shared_objects
        +list~WheelTag~ wheel_tags
        +str required_glibc
        +str target_architecture
        +bool has_binaries
        +bool is_pure_python
    }

    class SharedObjectInfo {
        +str name
        +str path
        +str architecture
        +str required_glibc
        +list~str~ dt_needed
        +str sha256
    }

    class Finding {
        +str rule_id
        +Severity severity
        +str title
        +str explanation
        +str package
        +str suggestion
        +float confidence
    }

    class ScanReport {
        +list~Finding~ findings
        +int packages_scanned
        +int total_packages
        +float scan_duration_ms
        +ScoreBreakdown score_breakdown
        +int health_score
        +str health_label
        +int critical_count
        +int warning_count
    }

    class ScoreBreakdown {
        +float overall_score
        +str overall_label
        +dict categories
        +int total_findings
        +CategoryScore weakest_category
    }

    class CategoryScore {
        +str name
        +float score
        +float weight
        +float weighted_score
        +str label
        +list top_issues
    }

    class Rule {
        <<abstract>>
        +str rule_id
        +Severity severity
        +is_applicable(SystemProfile) bool
        +evaluate(SystemProfile, list) list~Finding~
    }

    class Scanner {
        +run() ScanReport
        +check_package(str) list~Finding~
        +get_profile() SystemProfile
        +inspect_file(str) list~Finding~
    }

    Scanner --> SystemProfile : collects
    Scanner --> PackageBinaryInfo : analyzes
    Scanner --> Rule : evaluates
    Rule --> Finding : produces
    Scanner --> ScanReport : builds
    ScanReport --> Finding : contains
    ScanReport --> ScoreBreakdown : includes
    ScoreBreakdown --> CategoryScore : has 4
    PackageBinaryInfo --> SharedObjectInfo : contains
```

---

## Project Structure

```
pybinaryguard/
|
|-- pyproject.toml               # Packaging config (PEP 621)
|-- README.md
|-- LICENSE
|-- CHANGELOG.md
|-- .gitignore
|
|-- src/
|   +-- pybinaryguard/
|       |-- __init__.py          # Public API (scan, check, profile, inspect)
|       |-- scanner.py           # Core orchestrator
|       |
|       |-- models/              # Data structures
|       |   |-- system.py        # SystemProfile dataclass
|       |   |-- finding.py       # Finding, ScanReport
|       |   |-- package.py       # PackageBinaryInfo, SharedObjectInfo
|       |   +-- enums.py         # Severity, ScanMode
|       |
|       |-- probes/              # System information collectors
|       |   |-- os_probe.py      # OS, GLIBC, container detection
|       |   |-- python_probe.py  # Python version, ABI flags
|       |   |-- cpu_probe.py     # Architecture, instruction sets
|       |   |-- gpu_probe.py     # CUDA, GPU compute capability
|       |   |-- glibc_probe.py   # GLIBC version detection
|       |   |-- library_probe.py # System shared library inventory
|       |   +-- board_probe.py   # Embedded board detection (Jetson, RPi)
|       |
|       |-- analyzers/           # Package binary inspectors
|       |   |-- elf_analyzer.py  # ELF header & symbol table parsing
|       |   |-- wheel_analyzer.py
|       |   |-- symbol_analyzer.py
|       |   +-- dependency_analyzer.py
|       |
|       |-- rules/               # Compatibility rule engine
|       |   |-- engine.py        # Rule evaluation orchestrator
|       |   +-- builtin/         # 20+ built-in rules
|       |       |-- glibc_rules.py
|       |       |-- cuda_rules.py
|       |       |-- arch_rules.py
|       |       |-- cpu_rules.py
|       |       |-- numpy_rules.py
|       |       |-- container_rules.py
|       |       |-- python_abi_rules.py
|       |       |-- board_profile_rules.py
|       |       |-- framework_rules.py
|       |       +-- predictive_rules.py
|       |
|       |-- scoring/             # Health scoring v2
|       |   +-- engine.py        # Multi-dimensional weighted scoring
|       |
|       |-- profiles/            # Board profile engine
|       |   +-- engine.py        # Board detection & matching
|       |
|       |-- diagnostics/         # Error explanation
|       |   |-- findings.py
|       |   |-- explainer.py
|       |   +-- suggestions.py
|       |
|       |-- agent/               # Agent SDK
|       |   |-- tool_interface.py
|       |   |-- recommender.py
|       |   |-- schema.py        # Tool schema export (OpenAI/MCP)
|       |   |-- simulator.py     # Pre-install compatibility prediction
|       |   +-- guard.py         # Runtime import guard
|       |
|       |-- plugins/             # Plugin system
|       |   |-- loader.py
|       |   |-- hooks.py
|       |   +-- contrib/         # Built-in plugins (Jetson, TensorRT, OpenCV, GStreamer)
|       |
|       +-- cli/                 # Command-line interface
|           |-- main.py          # Argument parser
|           |-- commands.py      # Command handlers
|           +-- formatters.py    # Output formatting (table/json/minimal)
|
|-- tests/                       # 505 tests
|-- docs/                        # Documentation
+-- examples/                    # Usage examples
```

### Design Principles

1. **Zero dependencies by default** -- Core functionality works with just the Python standard library. Optional `pyelftools` enables deep ELF analysis.

2. **Read-only** -- Never modifies your system, packages, or files. Safe to run anywhere.

3. **Offline-first** -- No network calls. All analysis runs locally. Works in air-gapped environments.

4. **Agent-native** -- Structured JSON-serializable outputs. Tool schema export. Safety-classified actions.

5. **Extensible** -- Plugin system for custom probes, analyzers, and rules via entry points.

---

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Ubuntu 20.04+ | x86_64 | Fully supported |
| Ubuntu 20.04+ | aarch64 | Fully supported |
| Debian 11+ | x86_64 / aarch64 | Fully supported |
| RHEL / CentOS 8+ | x86_64 | Supported |
| Alpine (musl) | x86_64 | Supported |
| NVIDIA Jetson | aarch64 | Supported (board profiles) |
| Raspberry Pi OS | armv7l / aarch64 | Supported (board profiles) |
| Docker containers | any | Supported (auto-detected) |

---

## Testing

```bash
# Run all tests (505 tests)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=pybinaryguard --cov-report=term-missing

# Run specific test suites
python -m pytest tests/test_scoring.py -v      # Health scoring
python -m pytest tests/test_scan_modes.py -v   # Scan modes
python -m pytest tests/test_agent_sdk.py -v    # Agent SDK
```

---

## Author

**S P Pothihai Selvan** (Po-nuvai)
Applied Research Scientist @ [Nuvai AI Solution Pvt Ltd](https://nuvai.dev)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
