# PyBinaryGuard Architecture

## Overview

PyBinaryGuard follows a **probe → analyze → evaluate → report** pipeline:

```
System Probes    Package Analyzers    Rule Engine    Reporter
┌──────────┐    ┌──────────────┐    ┌───────────┐  ┌──────────┐
│ Python   │    │ ELF Parser   │    │ GLIBC     │  │ Table    │
│ CPU      │───>│ Wheel Meta   │───>│ CUDA      │─>│ JSON     │
│ GLIBC    │    │ Symbol Res.  │    │ Arch      │  │ Minimal  │
│ GPU/CUDA │    │ Dependency   │    │ CPU       │  │ Agent    │
│ OS       │    └──────────────┘    │ NumPy ABI │  └──────────┘
│ Board    │                        │ Container │
└──────────┘                        │ Framework │
                                    │ Predictive│
                                    └───────────┘
```

## Module Responsibilities

### `probes/` — System Information Collectors

Each probe collects one category of system info and populates `SystemProfile`:

| Probe | Collects | Method |
|-------|----------|--------|
| `python_probe` | Python version, ABI flags, implementation | `sys.version_info`, `sysconfig` |
| `cpu_probe` | Architecture, instruction sets (SSE, AVX) | `/proc/cpuinfo`, `platform.machine()` |
| `glibc_probe` | GLIBC version | `ctypes.CDLL`, `ldd --version` |
| `gpu_probe` | CUDA version, GPU name, compute capability | `nvidia-smi`, `nvcc`, `/usr/local/cuda` |
| `os_probe` | OS name/version, container detection | `/etc/os-release`, `/.dockerenv` |
| `library_probe` | System shared library inventory | `ldconfig -p`, `/usr/lib` scan |
| `board_probe` | Embedded board detection | `/proc/device-tree/model`, Tegra release |

### `analyzers/` — Package Binary Inspectors

Analyzers inspect the binary contents of installed Python packages:

| Analyzer | Inspects | Output |
|----------|----------|--------|
| `elf_analyzer` | `.so` files inside packages | ELF class, machine type, needed libs, symbol versions |
| `wheel_analyzer` | Wheel metadata (WHEEL, METADATA) | Platform tags, ABI tags, Python version |
| `symbol_analyzer` | Symbol tables in shared objects | Unresolved symbols, version requirements |
| `dependency_analyzer` | Package dependency chains | Dependency graph, conflict detection |

### `rules/` — Compatibility Rule Engine

Rules evaluate system profile + package data to produce findings:

```python
class Rule:
    def is_applicable(self, profile: SystemProfile) -> bool: ...
    def evaluate(self, profile: SystemProfile, packages: list) -> list[Finding]: ...
```

Categories:
- **GLIBC rules**: Version requirements, manylinux policy compliance
- **CUDA rules**: Toolkit version, driver compatibility, compute capability
- **Architecture rules**: x86_64/aarch64/armv7l mismatches
- **CPU rules**: Instruction set requirements (AVX2, SSE4)
- **NumPy ABI rules**: ABI version compatibility across NumPy versions
- **Container rules**: Docker/container-specific issues
- **Framework rules**: PyTorch, TensorFlow, TensorRT, ONNX Runtime
- **Board rules**: Embedded platform-specific checks
- **Predictive rules**: Import failure prediction, dependency chain analysis

### `scoring/` — Health Score Engine

Multi-dimensional weighted scoring:

```
Overall Score = Σ (category_score × category_weight)

Categories:
  Binary Stability  (35%) — GLIBC, ELF, ABI findings
  GPU Compatibility (30%) — CUDA, compute capability, driver
  Dependency Health (25%) — Version conflicts, missing libs
  Platform Risk     (10%) — Architecture, container, OS
```

Weights auto-adjust based on system (no GPU → redistribute GPU weight).

### `agent/` — Agent SDK

Structured interface for AI agents and automation:

| Module | Purpose |
|--------|---------|
| `tool_interface` | High-level functions returning structured dataclasses |
| `recommender` | Action recommendation with safety classification (safe/review/dangerous) |
| `schema` | Tool schema export (OpenAI function calling, MCP, JSON Schema) |
| `simulator` | Pre-install compatibility prediction by parsing wheel filenames |
| `guard` | Runtime import guard that captures and diagnoses failures |

### `plugins/` — Extension System

Plugin discovery via `setuptools` entry points:

```toml
[project.entry-points."pybinaryguard.plugins"]
my_plugin = "my_package:register"
```

Plugins can add:
- Custom probes (new system info sources)
- Custom analyzers (new binary inspection methods)
- Custom rules (new compatibility checks)

## Data Flow

```
1. Scanner.run()
   ├── Collect SystemProfile (all probes)
   ├── Discover installed packages
   ├── Analyze each package (analyzers)
   ├── Evaluate all rules (rule engine)
   ├── Compute health score (scoring engine)
   └── Return ScanReport

2. ScanReport contains:
   ├── findings: List[Finding]
   ├── health_score: float (0-100)
   ├── score_breakdown: ScoreBreakdown
   ├── packages_scanned: int
   ├── critical_count: int
   └── warning_count: int
```

## Design Decisions

1. **Zero dependencies**: Core works with stdlib only. `pyelftools` is optional for deep ELF analysis.
2. **Lazy imports**: Heavy modules imported inside functions to keep startup fast.
3. **Read-only**: Never modifies system, packages, or files.
4. **Offline**: No network calls. All analysis is local.
5. **Dataclass-based**: All data structures are `@dataclass` for type safety and serialization.
