# PyBinaryGuard — Complete Technical Deep Dive

**How it works. Why it matters. Every layer explained.**

---

## Table of Contents

1. [The Problem It Solves](#1-the-problem-it-solves)
2. [Why Society Needs This](#2-why-society-needs-this)
3. [How It Works — The Big Picture](#3-how-it-works--the-big-picture)
4. [Phase 1: System Probing](#4-phase-1-system-probing--fingerprinting-your-machine)
5. [Phase 2: Binary Analysis](#5-phase-2-binary-analysis--looking-inside-so-files)
6. [Phase 3: Rule Evaluation](#6-phase-3-rule-evaluation--42-rules-that-catch-failures)
7. [Phase 4: Diagnostics & Scoring](#7-phase-4-diagnostics--scoring)
8. [The Import Validator](#8-the-import-validator--the-nuclear-option)
9. [The Predictor Engine](#9-the-predictor-engine--simulating-the-linux-linker)
10. [The Agent SDK](#10-the-agent-sdk--ai-native-binary-intelligence)
11. [The Plugin System](#11-the-plugin-system--community-extensibility)
12. [Snapshot & Lockfile System](#12-snapshot--lockfile-system--environment-reproducibility)
13. [Board Profiles](#13-board-profiles--embedded-hardware-intelligence)
14. [Framework-Specific Deep Checks](#14-framework-specific-deep-checks)
15. [CLI & Output Formats](#15-cli--output-formats)
16. [Performance & Architecture Decisions](#16-performance--architecture-decisions)
17. [Real-World Impact Scenarios](#17-real-world-impact-scenarios)
18. [Complete File Map](#18-complete-file-map)

---

## 1. The Problem It Solves

Python is an interpreted language, but the libraries that make it powerful — NumPy, PyTorch, TensorFlow, OpenCV, SciPy — are not. They contain **compiled C/C++/CUDA code** packaged as `.so` (shared object) files inside `.whl` (wheel) archives.

A single `.so` file is compiled for a specific combination of:

| Factor | Example | What Goes Wrong |
|--------|---------|-----------------|
| **GLIBC version** | Built on Ubuntu 24.04 (GLIBC 2.39) | `GLIBC_2.39 not found` on Ubuntu 22.04 |
| **CPU architecture** | Built for x86_64 | `exec format error` on ARM Jetson |
| **CPU instructions** | Uses AVX2 optimizations | `Illegal instruction (core dumped)` on old CPUs |
| **CUDA version** | Built for CUDA 11.8 | `libcudart.so.11: cannot open` on CUDA 12 systems |
| **Python ABI** | Built for Python 3.10 | `undefined symbol: _PyGen_Send` on Python 3.12 |
| **GPU compute capability** | Requires compute 7.0+ | `cudaErrorInsufficientDriver` on older GPUs |
| **Shared library dependencies** | Needs `libcudnn.so.8` | `OSError: libcudnn.so.8: cannot open` |

**The developer's code is correct.** The binaries underneath don't match the system. And no existing tool catches this:

- `pip install` succeeds (it only checks metadata tags, not binary contents)
- `pip check` passes (it validates dependency versions, not binary compatibility)
- `poetry install` works (it resolves version constraints, not GLIBC/CUDA/CPU requirements)
- The crash happens at `import time` or during first computation

**PyBinaryGuard catches all of these before the first `import`.**

---

## 2. Why Society Needs This

### 2.1 The Human Cost

Every day, thousands of developers worldwide waste hours debugging crashes that aren't bugs:

**Students & beginners** — A student installs PyTorch on their laptop. Their code is correct. They get `Illegal instruction (core dumped)`. They think their code is broken. They spend 3 days debugging a non-bug. Many give up on programming.

**ML researchers** — A research team deploys a model to a GPU server. It crashes with `libcudart.so.11: cannot open shared object file`. The CUDA version doesn't match. They lose a day of compute time on a $10/hour GPU cluster.

**Edge AI engineers** — An engineer deploys a vision model to a Jetson Nano for a factory inspection system. The PyTorch wheel was built for x86. It silently fails. The factory line has no quality control until someone notices.

**Hospital/Medical AI** — A hospital deploys an AI diagnostic model. The container image has a GLIBC mismatch. The model fails silently during a night shift. Patient scans go unprocessed for 8 hours.

**Self-driving/Robotics** — A robotics team updates their ROS stack. The new OpenCV wheel requires AVX2 instructions. Their field robots have older Atom processors without AVX2. The robots lose vision capability mid-mission.

### 2.2 The Scale

- **23 billion** pip downloads per month (2025)
- **~15%** of packages contain compiled binaries
- **~3.5 billion** binary package installs per month
- Even a **0.1% failure rate** = **3.5 million crashes per month** from binary incompatibility

### 2.3 The Economic Impact

| Sector | Problem | Cost |
|--------|---------|------|
| **Education** | Students abandon programming due to incomprehensible errors | Lost talent pipeline |
| **Research** | GPU time wasted on environment issues instead of science | $10-50/hr * millions of hours |
| **Production ML** | Silent model failures, rollbacks, incident response | Millions in downtime |
| **Edge/IoT** | Field failures requiring physical device access | Travel, downtime, liability |
| **Open Source** | Maintainers triaging "bugs" that are env issues | 30-50% of issue trackers |

### 2.4 Why No One Built This Before

1. **Cross-domain expertise** — requires simultaneous knowledge of ELF binary format, CUDA SDK, Python ABI, Linux dynamic linker, ARM architecture, and manylinux policies
2. **Testing is hard** — you need real mismatched environments to test against
3. **Nobody owns the problem** — it falls between pip (packaging), NVIDIA (GPU), and Linux distros (GLIBC)
4. **Moving targets** — CUDA versions, manylinux policies, and Python ABIs evolve every few months

PyBinaryGuard exists because someone finally sat down and connected all the dots.

---

## 3. How It Works — The Big Picture

When you run `pybinaryguard scan`, four phases execute in sequence:

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER RUNS SCAN                           │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: PROBE — Fingerprint Your Machine                      │
│                                                                  │
│  9 probes run in PARALLEL (ThreadPoolExecutor, max 8 workers):   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ PythonProbe │  │  GlibcProbe │  │   CpuProbe  │              │
│  │ version,ABI │  │ 2.35 / musl │  │ x86_64,AVX2 │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   GpuProbe  │  │   OsProbe   │  │ LibraryProbe│              │
│  │ CUDA 12.2   │  │ Ubuntu,Dock │  │ ldconfig    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  BoardProbe │  │ToolchainPrb │  │  VenvProbe  │              │
│  │ Jetson/RPi  │  │ gcc,cmake   │  │ conda/venv  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  Output: SystemProfile (70+ fields describing your machine)      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2: ANALYZE — Look Inside Every Package's Binaries        │
│                                                                  │
│  For each installed package with .so files:                      │
│                                                                  │
│  ┌──────────────────┐  Reads WHEEL file for compatibility tags   │
│  │  WheelAnalyzer   │  Reads METADATA for name/version           │
│  │  (.dist-info)    │  Detects CUDA build version from metadata  │
│  └──────────────────┘                                            │
│  ┌──────────────────┐  Pure-Python ELF parser (no pyelftools)    │
│  │   ELFAnalyzer    │  Extracts: architecture, DT_NEEDED,        │
│  │   (.so files)    │  GLIBC symbol versions, build-id           │
│  └──────────────────┘                                            │
│  ┌──────────────────┐  Finds max GLIBC/GLIBCXX across all .so    │
│  │  SymbolAnalyzer  │  Detects CPython ABI linkage               │
│  └──────────────────┘                                            │
│  ┌──────────────────┐  Resolves DT_NEEDED using ld.so algorithm  │
│  │ DependencyAnalzr │  Identifies missing shared libraries        │
│  └──────────────────┘                                            │
│                                                                  │
│  Output: List[PackageBinaryInfo] with full binary metadata       │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3: EVALUATE — Run 42 Compatibility Rules                 │
│                                                                  │
│  Rules cross-reference Phase 1 (system) with Phase 2 (packages) │
│                                                                  │
│  GLIBC Rules (4):     Package needs GLIBC 2.38, you have 2.35   │
│  CUDA Rules (7):      PyTorch built for CUDA 11.8, you have 12  │
│  Python ABI Rules (3): Wheel for cp310, you run cp312           │
│  CPU Rules (2):       Binary needs AVX2, your CPU lacks it      │
│  Arch Rules (1):      x86_64 binary on aarch64 system           │
│  Framework Rules (5): PyTorch/TF/TensorRT/ONNX version checks   │
│  NumPy Rules (1):     NumPy C API version mismatch              │
│  Container Rules (2): Docker GPU mount issues                   │
│  Board Rules (5):     Known broken wheels for Jetson/RPi        │
│  Predictive Rules (2):Import failure prediction                  │
│  Dependency Rules (2):Version conflict + missing dependency     │
│  Source Build Rules(3):No compiler for source-built packages    │
│  Venv Rules (4):      Mixed environments, user-site leak        │
│  ────────────────────────────────────────────────────────────    │
│  Total: 42 rules producing Finding objects                      │
│                                                                  │
│  Output: List[Finding] with severity, explanation, fix command   │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       v
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 4: REPORT — Score, Format, Present                       │
│                                                                  │
│  ┌──────────────────┐  4-category weighted scoring:              │
│  │  Health Score v2 │  Binary Stability (35%)                    │
│  │  (0-100)         │  GPU Compatibility (30%)                   │
│  │                  │  Dependency Health (25%)                    │
│  │                  │  Platform Risk (10%)                        │
│  └──────────────────┘  Dynamic weight redistribution             │
│  ┌──────────────────┐                                            │
│  │  Diagnostics     │  Error pattern matching (20+ regex)        │
│  │  Engine          │  System-specific fix suggestions            │
│  │                  │  Copy-pasteable remediation commands        │
│  └──────────────────┘                                            │
│  ┌──────────────────┐                                            │
│  │  Formatter       │  Table (human), JSON (machine), Minimal    │
│  └──────────────────┘                                            │
│                                                                  │
│  Output: Formatted report + exit code (0/1/2/3)                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: System Probing — Fingerprinting Your Machine

The probe system collects everything about your environment using **only safe, read-only operations** (no binary execution, no sudo, no network).

### 4.1 PythonProbe — Your Interpreter

| What It Reads | How | What It Produces |
|---------------|-----|-----------------|
| Python version | `sys.version_info` | `(3, 12, 1)` |
| ABI tag | `sysconfig.get_config_var("SOABI")` | `"cp312-cp312-linux_x86_64"` |
| Implementation | `sys.implementation.name` | `"cpython"` |
| Debug build | `sys.flags.debug` + `sysconfig.get_config_var("Py_DEBUG")` | `True/False` |
| Stable ABI | `sys.abiflags` | `True/False` |

### 4.2 GlibcProbe — Your C Library

This probe uses a **3-layer fallback strategy** because GLIBC detection is notoriously unreliable across environments:

```
Layer 1: os.confstr("CS_GNU_LIBC_VERSION")
         → Fast, works on most systems
         → Fails on minimal containers

Layer 2: ctypes.CDLL("libc.so.6").gnu_get_libc_version()
         → Works when confstr fails
         → Fails if libc.so.6 isn't named exactly that

Layer 3: musl detection
         → Scans /proc/self/maps for musl linker path
         → Runs musl linker binary to get version
         → Falls back to ldd --version (musl identifies itself in stderr)
```

**Why this matters:** Alpine Linux uses musl instead of GLIBC. Any wheel built for `manylinux` (which means GLIBC) will fail on Alpine. This probe detects it.

### 4.3 CpuProbe — Your Processor

Reads `/proc/cpuinfo` and extracts:

| Field | x86 Location | ARM Location |
|-------|-------------|--------------|
| Model name | `model name` | `CPU implementer` + lookup |
| Features | `flags` line | `Features` line |
| Core count | `os.cpu_count()` | `os.cpu_count()` |

**Critical instruction set detection:**

| Flag | Why It Matters |
|------|---------------|
| `avx` | Some NumPy/SciPy binaries require AVX |
| `avx2` | Many modern ML libraries require AVX2 |
| `avx512f` | Some optimized builds use AVX-512 |
| `sse4_2` | Baseline for most modern x86 binaries |
| `neon` | ARM SIMD — equivalent of SSE for ARM |

### 4.4 GpuProbe — Your GPU Stack

The most complex probe, using **7 detection layers** because GPU information is fragmented across multiple sources:

```
Layer 1: Device Nodes
         Check /dev/nvidia0, /dev/nvidiactl
         → Confirms physical GPU presence

Layer 2: Kernel Driver
         Parse /proc/driver/nvidia/version
         → Gets driver version string

Layer 3: NVML (Management Library)
         ctypes.CDLL("libnvidia-ml.so.1")
         → nvmlInit()
         → nvmlSystemGetDriverVersion()
         → nvmlDeviceGetCount()
         → For each device: name, memory, compute capability
         → Returns: GPU name, compute cap, driver version, memory

Layer 4: CUDA Runtime
         ctypes.CDLL("libcudart.so.12") (tries 12, 11, generic)
         → cudaRuntimeGetVersion()
         → Decodes: major*1000 + minor*10

Layer 5: CUDA Toolkit
         Check CUDA_HOME / CUDA_PATH environment
         → Find nvcc in PATH
         → Parse nvcc --version output

Layer 6: Jetson-Specific
         Parse /etc/nv_tegra_release
         → Map L4T version to JetPack version
         → Detect unified memory architecture

Layer 7: cuDNN
         ctypes.CDLL("libcudnn.so.9") (tries 9, 8, generic)
         → cudnnGetVersion()
         OR parse cudnn_version.h header file
```

**Why so many layers:** On a Jetson, `nvidia-smi` doesn't exist. In a Docker container, device nodes might be mounted but NVML might not be accessible. On a headless server, the CUDA runtime might be installed without the toolkit. Each layer compensates for the others.

### 4.5 OsProbe — Your Operating System

Detects distribution, kernel, containers, and VMs:

**Container detection** (5 methods checked in order):
1. `/.dockerenv` file exists → Docker
2. `/proc/1/cgroup` contains "docker"/"lxc"/"containerd" → respective runtime
3. `/proc/1/mountinfo` contains overlay filesystem → Docker/containerd
4. `os.environ["container"]` is set → systemd-nspawn/podman
5. `/run/.containerenv` exists → Podman

**VM detection** reads `/sys/class/dmi/id/` looking for:
- `product_name`: "VirtualBox", "VMware", "QEMU", etc.
- `sys_vendor`: "Xen", "Microsoft Corporation" (Hyper-V), etc.
- Also checks `/proc/cpuinfo` for `hypervisor` flag

### 4.6 LibraryProbe — Your Shared Libraries

Maps every shared library available on the system:

1. Parses `$LD_LIBRARY_PATH` environment variable
2. Runs `ldconfig -p` to get the system linker cache
3. Parses output into `Dict[str, str]` mapping (e.g., `"libz.so.1" → "/lib/x86_64-linux-gnu/libz.so.1"`)
4. Collects all Python `site-packages` directories

This data is critical for Phase 2 — when checking if a package's `DT_NEEDED` libraries actually exist on the system.

### 4.7 BoardProbe — Your Hardware Platform

Detects embedded boards through device tree and platform-specific files:

| Board | Detection Method |
|-------|-----------------|
| NVIDIA Jetson | `/etc/nv_tegra_release` + `/proc/device-tree/model` contains "NVIDIA" |
| Raspberry Pi | `/proc/device-tree/model` contains "Raspberry Pi" |
| BeagleBone | `/proc/device-tree/model` contains "BeagleBone" |
| Google Coral | `/proc/device-tree/model` contains "coral"/"Freescale" + `/dev/apex_0` |
| Generic ARM SBC | `aarch64`/`armv7l` architecture + no desktop GPU |

When a board is detected, it activates board-specific rules and profiles (Section 13).

### 4.8 ToolchainProbe — Your Compiler Stack

Checks availability of build tools:

| Tool | Version Regex | Why It Matters |
|------|--------------|---------------|
| `gcc` | `(\d+\.\d+\.\d+)` | C compiler for building from source |
| `g++` | `(\d+\.\d+\.\d+)` | C++ compiler |
| `clang` | `version (\d+\.\d+\.\d+)` | Alternative compiler |
| `cmake` | `version (\d+\.\d+\.\d+)` | Build system |
| `make` | `(\d+\.\d+)` | Build automation |
| `rustc` | `(\d+\.\d+\.\d+)` | Rust compiler (for Rust extensions) |

Also checks `sysconfig.get_path("include")` for `Python.h` — required to compile C extensions.

### 4.9 VenvProbe — Your Environment Isolation

Detects virtual environment type and misconfigurations:

**Detection priority:**
1. `CONDA_DEFAULT_ENV` set → conda
2. `POETRY_ACTIVE` set → poetry
3. `PIPENV_ACTIVE` set → pipenv
4. `PDM_IN_VENV` set → pdm
5. `sys.real_prefix` exists → virtualenv (legacy)
6. `sys.base_prefix != sys.prefix` → venv
7. None of above → system

**Hygiene checks:**
- `mixed_env_risk`: Multiple site-packages roots on `sys.path` (system packages leaking into venv)
- `pip_user_site_enabled`: `~/.local/lib/python3.x/site-packages` on `sys.path` inside a venv

### All Probes Combined → SystemProfile

The output of all 9 probes is merged into a single `SystemProfile` frozen dataclass with **70+ fields**. This is the complete fingerprint of the machine.

---

## 5. Phase 2: Binary Analysis — Looking Inside .so Files

### 5.1 Package Discovery

The scanner walks every `site-packages` directory looking for `.dist-info` folders:

```
site-packages/
├── torch-2.4.0+cu124.dist-info/
│   ├── WHEEL          ← Compatibility tags
│   ├── METADATA       ← Package name, version
│   ├── RECORD         ← File manifest
│   └── top_level.txt  ← Importable module name
├── torch/
│   ├── lib/
│   │   ├── libtorch_cuda.so    ← Binary to analyze
│   │   ├── libc10.so           ← Binary to analyze
│   │   └── ...
│   └── ...
└── ...
```

### 5.2 WheelAnalyzer — Reading Package Metadata

From the `WHEEL` file, extracts compatibility tags like:
```
Tag: cp312-cp312-manylinux_2_17_x86_64
     ^^^^  ^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
     │     │     └─ Platform: GLIBC 2.17 on x86_64
     │     └─ ABI: CPython 3.12
     └─ Interpreter: CPython 3.12
```

From the `METADATA` file, extracts package name and version. For packages like `torch==2.4.0+cu124`, it extracts the CUDA version `(12, 4)` from the version string.

### 5.3 ELFAnalyzer — Pure Python Binary Parser

**This is the core innovation.** A complete ELF (Executable and Linkable Format) parser written in pure Python, reading binary files as raw data:

```
ELF File Structure:
┌────────────────────┐
│   ELF Header       │  → Architecture (x86_64? ARM?), 32/64-bit, endianness
│   (16 + 48 bytes)  │
├────────────────────┤
│   Program Headers  │  → Memory segments (for loading, not analyzed deeply)
├────────────────────┤
│   Section Headers  │  → Points to all data sections
├────────────────────┤
│   .dynamic          │  → DT_NEEDED (required libraries)
│                     │     DT_SONAME (library's own name)
│                     │     DT_RPATH/RUNPATH (embedded search paths)
├────────────────────┤
│   .gnu.version_r   │  → Required symbol versions:
│                     │     GLIBC_2.17, GLIBC_2.35, GLIBCXX_3.4.29
├────────────────────┤
│   .note.gnu.build-id│ → Unique build identifier (SHA1 hash)
├────────────────────┤
│   .dynstr          │  → String table for dynamic symbols
└────────────────────┘
```

**What we extract from each .so:**

| Data | Source | Why |
|------|--------|-----|
| Target architecture | `e_machine` field | Detect x86 binary on ARM system |
| Bit width | `ei_class` | 32 vs 64-bit mismatch |
| Required libraries | `.dynamic` DT_NEEDED | Find missing libcudart.so, libcudnn.so |
| Embedded search paths | `.dynamic` DT_RPATH/RUNPATH | Library resolution |
| GLIBC requirements | `.gnu.version_r` section | Max GLIBC version needed |
| GLIBCXX requirements | `.gnu.version_r` section | C++ stdlib version needed |
| Build ID | `.note.gnu.build-id` | Binary integrity tracking |

**Example:** PyTorch's `libtorch_cuda.so` might have:
```
DT_NEEDED: libcudart.so.12, libcublas.so.12, libcudnn.so.9, libc.so.6
GLIBC versions required: GLIBC_2.4, GLIBC_2.17
→ Max GLIBC needed: 2.17
```

### 5.4 SymbolAnalyzer — GLIBC Version Extraction

For every `.so` file, finds the **maximum required GLIBC version** across all version requirements:

```
File: scipy/_lib/_ccallback_c.cpython-312-x86_64-linux-gnu.so
  Version requirements:
    GLIBC_2.2.5    (basic memory functions)
    GLIBC_2.4      (stack protector)
    GLIBC_2.17     (clock_gettime)
    GLIBC_2.38     (strlcpy — new in glibc 2.38!)
                    ^^^^^^^^ This is the one that will crash your program
  → Required GLIBC: 2.38
  → Your system GLIBC: 2.35
  → VERDICT: WILL FAIL with "version 'GLIBC_2.38' not found"
```

### 5.5 DependencyAnalyzer — Shared Library Resolution

For each `DT_NEEDED` library, searches using the **same algorithm as the Linux dynamic linker** (`ld.so`):

```
Search order for "libcudart.so.12":

1. DT_RPATH from the binary itself
   → Check /opt/pytorch/lib/libcudart.so.12

2. LD_LIBRARY_PATH
   → Check /usr/local/cuda/lib64/libcudart.so.12

3. DT_RUNPATH from the binary
   → Check $ORIGIN/../lib/libcudart.so.12

4. ldconfig cache (from ldconfig -p)
   → libcudart.so.12 → /usr/local/cuda-12.2/lib64/libcudart.so.12

5. Standard library paths
   → /lib/x86_64-linux-gnu/libcudart.so.12
   → /usr/lib/x86_64-linux-gnu/libcudart.so.12
   → /lib64/libcudart.so.12
   → /usr/lib64/libcudart.so.12

Not found anywhere? → MISSING_SHARED_LIB finding
```

**Safety:** This is pure path checking. We never call `ldd` (which actually loads the binary and executes constructor functions in untrusted code).

---

## 6. Phase 3: Rule Evaluation — 42 Rules That Catch Failures

Each rule takes `(SystemProfile, List[PackageBinaryInfo])` and returns `List[Finding]`.

### 6.1 GLIBC Rules (4 rules)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `GLIBC_VERSION_MISMATCH` | CRITICAL | Package .so needs GLIBC 2.38, system has 2.35 |
| `MUSL_GLIBC_CONFLICT` | CRITICAL | Any manylinux wheel on Alpine Linux (musl) |
| `MANYLINUX_TAG_VIOLATION` | WARNING | Wheel says manylinux_2_17 but binary actually needs 2.28 |
| `LIBSTDCXX_TOO_OLD` | CRITICAL | Package needs GLIBCXX_3.4.30, system has 3.4.29 |

**GLIBC_VERSION_MISMATCH** is the most common real-world finding. Example output:
```
CRITICAL  scipy 1.14.0
          GLIBC version incompatible
          Package requires GLIBC >= 2.38 but your system has GLIBC 2.35.
          This will cause "version 'GLIBC_2.38' not found" on import.
          Fix: pip install scipy==1.11.4  (last version supporting GLIBC 2.35)
```

### 6.2 CUDA Rules (7 rules)

Cross-validates the entire GPU stack:

```
                GPU Driver (535.129.03)
                    │
                    │ supports up to CUDA 12.2?
                    v
              CUDA Runtime (12.2)
                    │
                    │ matches framework build?
        ┌───────────┼───────────┐
        v           v           v
   PyTorch      TensorFlow   TensorRT
   cu124        cuda 12.3    8.6
   MISMATCH!    OK           OK
```

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `CUDA_DRIVER_TOO_OLD` | CRITICAL | CUDA 12.4 runtime but driver only supports up to 12.2 |
| `CUDA_RUNTIME_MISMATCH` | CRITICAL | PyTorch built for CUDA 11.8, system has CUDA 12.x |
| `CUDA_MINOR_MISMATCH` | WARNING | CUDA 12.1 vs 12.4 (usually works but worth noting) |
| `CUDNN_VERSION_MISMATCH` | CRITICAL | cuDNN 8 vs cuDNN 9 mismatch |
| `CUDA_NOT_FOUND` | WARNING | GPU package installed but no CUDA runtime detected |
| `COMPUTE_CAPABILITY_LOW` | CRITICAL | GPU compute 3.0 but PyTorch needs 3.7+ |
| `CUDA_LIB_MISSING` | CRITICAL | `libcudart.so.12` not found anywhere on system |

Uses compatibility data from JSON files:
- `cuda_compat_matrix.json` — driver version → max supported CUDA
- `pytorch_cuda_matrix.json` — PyTorch version → supported CUDA variants
- `tensorflow_cuda_matrix.json` — TF version → required CUDA/cuDNN

### 6.3 Architecture Rules (1 rule)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `ARCH_MISMATCH` | CRITICAL | ELF e_machine is EM_X86_64 (62) but system is EM_AARCH64 (183) |

This catches the common Jetson mistake: `pip install torch` downloads the x86 wheel because PyPI's default is x86.

### 6.4 CPU Rules (2 rules)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `AVX2_REQUIRED` | CRITICAL | Binary uses AVX2 instructions, CPU doesn't have `avx2` flag |
| `ILLEGAL_INSTRUCTION_RISK` | CRITICAL | Heuristic: known packages that compile with advanced instructions |

The `Illegal instruction (core dumped)` error is one of the most confusing crashes in all of computing. It happens when a CPU encounters an instruction it doesn't support. PyBinaryGuard catches this **before** it happens.

### 6.5 Python ABI Rules (3 rules)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `PYTHON_ABI_MISMATCH` | CRITICAL | Wheel for `cp310` loaded into Python 3.12 |
| `PYTHON_VERSION_MISMATCH` | CRITICAL | Interpreter tag `cp310` vs system `cp312` |
| `DEBUG_RELEASE_MIX` | WARNING | Debug-built Python loading release extensions |

### 6.6 AI Framework Rules (5 rules)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `PYTORCH_CUDA_ABI_MISMATCH` | CRITICAL | PyTorch CUDA build version vs system CUDA |
| `PYTORCH_TORCHVISION_INCOMPATIBLE` | WARNING | torchvision version incompatible with torch |
| `TENSORFLOW_COMPUTE_CAPABILITY_LOW` | CRITICAL | GPU compute capability below TF minimum |
| `TENSORRT_INCOMPATIBLE` | CRITICAL | TensorRT version vs CUDA version conflict |
| `ONNX_RUNTIME_PROVIDER_MISMATCH` | WARNING | ONNX Runtime CUDA provider vs system CUDA |

### 6.7 Board-Specific Rules (5 rules)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `KNOWN_BROKEN_WHEEL` | CRITICAL | Package version known to fail on this board |
| `BOARD_INCOMPATIBLE_PACKAGE` | CRITICAL | Package fundamentally incompatible (nvidia-* on RPi) |
| `BOARD_CUDA_VERSION_MISMATCH` | CRITICAL | CUDA version doesn't match JetPack expectations |
| `BOARD_GLIBC_MISMATCH` | WARNING | GLIBC newer than board's recommended version |
| `BOARD_PYTHON_VERSION_UNSUPPORTED` | WARNING | Python version not validated for this board |

### 6.8 Dependency & Environment Rules (9 rules)

| Rule ID | Severity | What It Catches |
|---------|----------|-----------------|
| `DEPENDENCY_VERSION_CONFLICT` | WARNING | Package A needs B>=2.0 but B==1.9 is installed |
| `DEPENDENCY_MISSING` | WARNING | Required dependency not installed at all |
| `SOURCE_BUILD_DETECTED` | INFO | Package was built from source (sdist) |
| `SOURCE_BUILD_NO_COMPILER` | WARNING | Source-built packages but no gcc/clang available |
| `SOURCE_BUILD_NO_PYTHON_HEADERS` | WARNING | No Python.h for building C extensions |
| `VENV_SYSTEM_PYTHON` | INFO | Running on system Python with many packages |
| `VENV_MIXED_ENVIRONMENT` | WARNING | Multiple environment roots on sys.path |
| `VENV_USER_SITE_LEAK` | WARNING | User site-packages leaking into venv |
| `VENV_CONDA_PIP_MIXING` | INFO | Many pip binary packages inside conda env |

---

## 7. Phase 4: Diagnostics & Scoring

### 7.1 Health Score v2 — Multi-Dimensional

Unlike a simple penalty system, PyBinaryGuard scores across **4 independent categories**:

```
┌─────────────────────────────────────────────────────┐
│                  HEALTH SCORE: 72                    │
│                                                     │
│  Binary Stability    ████████░░  80/100  (×0.35)    │
│  GPU Compatibility   ██████░░░░  60/100  (×0.30)    │
│  Dependency Health   █████████░  90/100  (×0.25)    │
│  Platform Risk       ██████████  100/100 (×0.10)    │
│                                                     │
│  Overall = 80×0.35 + 60×0.30 + 90×0.25 + 100×0.10  │
│         = 28 + 18 + 22.5 + 10 = 78.5               │
└─────────────────────────────────────────────────────┘
```

**Dynamic weight adjustment:**
- No GPU detected → GPU weight redistributed to other categories
- Not on embedded board → Platform risk weight partially redistributed
- Weights always normalized to sum to 1.0

**Per-finding penalty:**
- CRITICAL: -30 points × confidence
- WARNING: -10 points × confidence
- INFO: -2 points × confidence
- PASSED: 0

### 7.2 Error Pattern Database

The diagnostics engine includes 20+ compiled regex patterns that map cryptic errors to root causes:

```python
# When you paste this error:
"version `GLIBC_2.34' not found (required by ./mylib.so)"

# PyBinaryGuard matches:
Pattern: r"version .GLIBC_(\d+\.\d+). not found"
→ Root cause: "GLIBC version too old"
→ Rule ID: GLIBC_VERSION_MISMATCH
→ Fix hint: "Upgrade system or find an older wheel build"
```

### 7.3 Context-Aware Suggestions

Fix suggestions detect your OS and give the right package manager command:

| Detected OS | Suggestion Format |
|-------------|-------------------|
| Debian/Ubuntu | `apt install python3.12-dev` |
| RHEL/CentOS | `yum install python3-devel` |
| SUSE | `zypper install python3-devel` |
| Arch Linux | `pacman -S python` |
| Unknown | Generic instructions |

---

## 8. The Import Validator — The Nuclear Option

Beyond static analysis, PyBinaryGuard can actually **test imports** in isolated subprocesses:

```
pybinaryguard validate
```

### How It Works

For each package with `.so` files:

```
Main Process                    Subprocess (isolated)
     │                               │
     ├─ Spawn subprocess ─────────► │
     │   python -c "                 │
     │     import sys                │
     │     try:                      │
     │       import torch            │
     │       print('OK')             │
     │     except ImportError as e:  │
     │       print(e, file=stderr)   │
     │       sys.exit(10)            │
     │     except OSError as e:      │
     │       print(e, file=stderr)   │
     │       sys.exit(11)            │
     │   "                           │
     │                               │
     ◄─ Capture result ──────────── │
     │                               │
     ├─ exit code 0 → SUCCESS        │
     ├─ exit code 10 → ImportError   │
     ├─ exit code 11 → OSError       │
     ├─ exit code 12 → Other Error   │
     ├─ exit code 132 → SIGILL       │  (Illegal instruction!)
     ├─ exit code 139 → SIGSEGV      │  (Segmentation fault!)
     ├─ exit code 134 → SIGABRT      │  (Abort!)
     └─ timeout → hung import        │
```

### Signal Classification

When a process is killed by a signal, Linux sets the exit code to `128 + signal_number`:

| Exit Code | Signal | Category | Human Meaning |
|-----------|--------|----------|---------------|
| 132 or -4 | SIGILL | `illegal_instruction` | Binary compiled for incompatible CPU |
| 139 or -11 | SIGSEGV | `segfault` | Memory access violation in binary |
| 134 or -6 | SIGABRT | `abort` | Binary assertion failure |
| 137 or -9 | SIGKILL | `killed` | Process killed (out of memory?) |

### Error Categorization from stderr

| stderr Contains | Category | Likely Cause |
|----------------|----------|--------------|
| `glibc` or `version` | `glibc_mismatch` | GLIBC too old |
| `libcuda` or `cuda` | `cuda_missing` | CUDA library not found |
| `cannot open shared object` | `missing_shared_library` | Missing .so dependency |
| `undefined symbol` | `undefined_symbol` | ABI mismatch |

---

## 9. The Predictor Engine — Simulating the Linux Linker

Without actually executing anything, PyBinaryGuard can **predict** whether an import will fail:

### 9.1 Dependency Graph Builder

Recursively builds the full tree of shared library dependencies:

```
torch/_C.cpython-312-x86_64-linux-gnu.so
├── libtorch_python.so
│   ├── libtorch_cuda.so
│   │   ├── libcudart.so.12     ← Found in /usr/local/cuda/lib64/
│   │   ├── libcublas.so.12     ← Found in /usr/local/cuda/lib64/
│   │   ├── libcudnn.so.9       ← NOT FOUND! ← PREDICTED FAILURE
│   │   └── libnccl.so.2        ← NOT FOUND! ← PREDICTED FAILURE
│   ├── libc10.so
│   │   └── libgomp.so.1        ← Found in /usr/lib/
│   └── libc10_cuda.so
│       └── libcuda.so.1        ← Found in /usr/lib/
├── libpython3.12.so.1.0        ← Found in /usr/lib/
└── libc.so.6                   ← Found in /lib/
```

### 9.2 Linker Simulator

Mimics the behavior of `ld.so` without executing any code:

1. Parse ELF headers of the target binary
2. Build `DT_NEEDED` dependency tree
3. For each library, search: RPATH → LD_LIBRARY_PATH → RUNPATH → ldconfig → standard paths
4. Check GLIBC symbol versions against system GLIBC
5. Detect circular dependencies
6. Report all predicted failures with confidence scores

### 9.3 Pre-Install Simulation

The `simulate` command can predict compatibility from a wheel filename alone:

```bash
$ pybinaryguard simulate torch-2.4.0+cu124-cp312-cp312-manylinux_2_17_x86_64.whl

[+] torch-2.4.0+cu124: COMPATIBLE
    Confidence: 85%
    Risk Level: low
```

Parses the wheel filename to extract:
- Python version: `cp312` → 3.12
- ABI: `cp312` → CPython 3.12
- Platform: `manylinux_2_17_x86_64` → needs GLIBC 2.17, needs x86_64
- CUDA: `cu124` → needs CUDA 12.4

Then cross-references against your SystemProfile.

---

## 10. The Agent SDK — AI-Native Binary Intelligence

PyBinaryGuard is designed to be called by AI agents (GPT, LangChain, AutoGen, CrewAI) as a tool:

### 10.1 Tool Registration

```python
from pybinaryguard.agent import as_agent_tool

# One-line integration with any agent framework
tools = as_agent_tool()
```

### 10.2 Schema Export

Exports tool schemas in 3 formats for different agent frameworks:

| Format | Framework | How |
|--------|-----------|-----|
| `openai` | OpenAI function calling, LangChain | `export_tool_schema(format="openai")` |
| `mcp` | Model Context Protocol (Anthropic) | `export_tool_schema(format="mcp")` |
| `json_schema` | Generic JSON Schema | `export_tool_schema(format="json_schema")` |

### 10.3 Agent Tools

| Tool | Input | Output |
|------|-------|--------|
| `scan()` | severity, scan_mode, packages | ActionableReport with safe/review/dangerous actions |
| `check(package)` | package name | AgentCheckResult with findings |
| `simulate_install(spec)` | wheel filename or package spec | Prediction with blockers/warnings/confidence |
| `doctor(error)` | error message string | Diagnosis with root cause and fix plan |
| `profile()` | none | System profile as structured JSON |

### 10.4 Action Classification

Every remediation action is classified for agent safety:

| Classification | Meaning | Example |
|---------------|---------|---------|
| `safe` | Agent can execute automatically | `pip install torch==2.4.0` |
| `review` | Agent should show to human first | `apt upgrade libstdc++6` |
| `dangerous` | Agent must get explicit confirmation | `pip install --force-reinstall --no-deps torch` |

### 10.5 Import Guard

Automatically captures and diagnoses import failures at runtime:

```python
import pybinaryguard.agent.guard  # Installs exception hook

try:
    import torch  # Fails with ImportError
except ImportError:
    pass

# Check what was captured
from pybinaryguard.agent.guard import captured_issues
print(captured_issues)
# [{"error_type": "ImportError", "category": "cuda_missing", "module": "torch", ...}]
```

---

## 11. The Plugin System — Community Extensibility

### 11.1 Extension Points

Plugins can register **7 types of extensions**:

| Hook | Purpose | Example |
|------|---------|---------|
| `add_probe` | New system info source | ROS environment probe |
| `add_rule` | New compatibility check | DeepStream version validator |
| `add_board_detector` | New SBC detection | Custom industrial board |
| `add_framework_checker` | Framework-specific checks | JAX CUDA checker |
| `add_reporter` | New output format | Slack webhook reporter |
| `pre_scan` | Run before scanning | Set up environment |
| `post_scan` | Run after scanning | Upload results |

### 11.2 Creating a Plugin

```python
# my_plugin/__init__.py
from pybinaryguard.plugins.hooks import HookRegistry
from pybinaryguard.rules.base import Rule

class MyCustomRule(Rule):
    rule_id = "MY_CUSTOM_CHECK"
    description = "Check for project-specific requirement"

    def evaluate(self, profile, packages):
        findings = []
        # Custom logic...
        return findings

def register(registry: HookRegistry):
    registry.add_rule(MyCustomRule())
```

```toml
# pyproject.toml
[project.entry-points."pybinaryguard.plugins"]
my_plugin = "my_plugin:register"
```

### 11.3 Built-in Contrib Plugins

| Plugin | Auto-activates When | What It Checks |
|--------|-------------------|---------------|
| `jetson` | `/etc/nv_tegra_release` found | JetPack/L4T/CUDA alignment |
| `opencv` | OpenCV importable | CUDA support, GStreamer, V4L2 |
| `tensorrt` | TensorRT importable | Engine file compatibility, GPU arch |
| `gstreamer` | GStreamer installed | Pipeline elements, hardware acceleration |

---

## 12. Snapshot & Lockfile System — Environment Reproducibility

### 12.1 Creating a Snapshot

```bash
pybinaryguard snapshot -o environment.lock.json
```

Captures a complete **binary-level fingerprint** of your environment:

```json
{
  "version": "1.0.3",
  "timestamp": "2026-02-16T10:30:00Z",
  "system": {
    "python_version": "3.12.1",
    "glibc_version": "2.35",
    "cuda_runtime": "12.2",
    "architecture": "x86_64",
    "detected_board": null,
    "cpu_flags": ["avx", "avx2", "sse4_2"]
  },
  "packages": [
    {
      "name": "torch",
      "version": "2.4.0+cu124",
      "cuda_version": "12.4",
      "manylinux_tag": "manylinux_2_17_x86_64",
      "binary_hashes": {
        "torch/lib/libtorch_cuda.so": "sha256:a1b2c3d4...",
        "torch/lib/libc10.so": "sha256:e5f6a7b8..."
      }
    }
  ]
}
```

### 12.2 Verifying Against a Snapshot

```bash
pybinaryguard verify environment.lock.json
```

Checks:
1. **System match** — Same Python version, GLIBC, CUDA, architecture?
2. **Package versions** — Same packages with same versions installed?
3. **Binary integrity** — SHA256 hashes of .so files match? (catches silent binary changes)
4. **Compatibility** — Runs import failure prediction for each package

This goes far beyond `pip freeze` — it verifies that the **actual binaries** are identical, not just the package versions.

---

## 13. Board Profiles — Embedded Hardware Intelligence

PyBinaryGuard ships with curated profiles for 5 embedded boards, each containing:

### 13.1 Profile Structure

```
Board Profile
├── Detection patterns (device tree, CPU model, revision codes)
├── Hardware specs (CUDA compute, max CUDA version, RAM)
├── Compatibility data
│   ├── Supported Python versions
│   ├── Compatible manylinux tags
│   ├── CUDA versions available
│   └── OpenCV backends
├── Known issues
│   ├── Broken wheels (package + version + reason + fix)
│   └── Fundamentally incompatible packages
├── Recommendations
│   ├── OS choice
│   ├── Thermal management
│   └── Build-from-source suggestions
└── Validated stacks
    └── Tested package combinations that work together
```

### 13.2 Shipped Profiles

| Board | CUDA | Compute | Key Issues |
|-------|------|---------|------------|
| **Jetson Nano** | 10.2 | 5.3 | PyTorch >=2.0 broken, only Python 3.6-3.8 |
| **Jetson Orin NX** | 12.2 | 8.7 | PyTorch <2.0 broken, needs JetPack 6.0 |
| **Raspberry Pi 4** | None | None | No GPU acceleration, GLIBC 2.31 limit |
| **Raspberry Pi 5** | None | None | No GPU, but Python 3.12 support |
| **Google Coral** | None | None (Edge TPU) | Only TFLite, 1GB RAM, Python 3.7-3.9 |

### 13.3 Validated Stacks

Each board profile includes tested package combinations. Example for Jetson Orin NX:

```json
{
  "name": "Full ML Stack (JetPack 6.0)",
  "packages": {
    "torch": "2.1.0",
    "torchvision": "0.16.0",
    "torchaudio": "2.1.0",
    "onnxruntime-gpu": "1.16.0",
    "opencv-python": "4.8.0.76",
    "numpy": "1.24.4"
  }
}
```

---

## 14. Framework-Specific Deep Checks

### 14.1 PyTorch Intelligence

- Extracts CUDA build version from package metadata (`+cu124`, `+cu118`, `+cpu`)
- Detects ROCm (AMD GPU) builds
- Checks C++11 ABI compatibility from GLIBCXX symbols
- Validates torchvision ↔ torch version compatibility matrix
- Detects distributed backends: NCCL, Gloo, MPI

### 14.2 TensorFlow Intelligence

- Checks compute capability requirements (TF 2.11+ needs ≥3.5)
- Validates AVX instruction set requirement (TF 2.x needs AVX)
- CUDA compatibility matrix (TF 2.15→CUDA 12.2, TF 2.14→CUDA 11.8)
- Detects GPU support variants: CUDA, TensorRT, ROCm, XLA
- TFLite schema compatibility checking

### 14.3 TensorRT Intelligence

- Validates TensorRT version vs CUDA version
- Checks serialized engine file compatibility (compute capability encoded in engine)
- Validates TRT version against PyTorch/TF version requirements

### 14.4 ONNX Runtime Intelligence

- Detects execution provider: CPU, CUDA, TensorRT, OpenVINO
- Validates CUDA provider vs system CUDA version
- Checks GPU availability when GPU provider is registered

---

## 15. CLI & Output Formats

### 15.1 Ten Commands

| Command | Purpose |
|---------|---------|
| `scan` | Full environment scan — the primary command |
| `check <pkg>` | Check a single package in depth |
| `profile` | Show system profile only (no compatibility checks) |
| `doctor --error "msg"` | Diagnose a specific error message |
| `inspect <file>` | Analyze a .whl or .so file before installing |
| `snapshot -o file` | Create binary-level environment lockfile |
| `verify <lockfile>` | Verify current env matches snapshot |
| `simulate <spec>` | Predict if a package will be compatible |
| `validate` | Actually test imports in subprocesses |
| `export-tool-schema` | Export AI agent tool definitions |

### 15.2 Three Output Formats

**Table** (default) — Colorized, human-readable:
```
CRITICAL  torch 2.1.0+cu118
          CUDA version mismatch
          Package was built for CUDA 11.8 but your system has CUDA 12.2.
          Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**JSON** — Machine-readable for CI/CD:
```json
{
  "rule_id": "CUDA_RUNTIME_MISMATCH",
  "severity": "critical",
  "package": "torch",
  "title": "CUDA version mismatch",
  "suggestion": "pip install torch --index-url ..."
}
```

**Minimal** — One line per finding for logs:
```
CRITICAL: torch 2.1.0 - CUDA version mismatch
```

### 15.3 Three Scan Modes

| Mode | Speed | What It Does |
|------|-------|-------------|
| `--fast` | < 1s for 100 packages | Metadata only (WHEEL files), skips ELF parsing |
| (default) | < 3s for 100 packages | ELF header parsing, GLIBC extraction, DT_NEEDED |
| `--deep` | < 10s for 100 packages | Full symbol resolution + SHA256 hash verification |

### 15.4 Exit Codes

| Code | Meaning | CI Usage |
|------|---------|----------|
| 0 | All checks passed | Deploy safely |
| 1 | Warnings only | Deploy with caution |
| 2 | Critical issues found | Block deployment |
| 3 | Scanner error | Investigate |

---

## 16. Performance & Architecture Decisions

### 16.1 Why Zero Dependencies

PyBinaryGuard uses only Python's standard library for its core. The ELF parser is written from scratch in pure Python. Why?

**The tool that checks binary compatibility must not itself have binary compatibility issues.**

If PyBinaryGuard depended on `pyelftools` (which is pure Python, but still a dependency), and `pyelftools` had a version conflict with something else in the user's environment, the diagnostic tool itself would fail. Zero dependencies means it always works.

### 16.2 Why Parallel Probes

All 9 probes run simultaneously using `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(probe.collect): probe
        for probe in probes
        if probe.is_applicable()
    }
```

The GPU probe takes ~200ms (ctypes calls). The library probe takes ~100ms (ldconfig subprocess). Running them in parallel cuts probe time from ~800ms to ~250ms.

### 16.3 Why Section-Level ELF Reads

The ELF parser only reads the sections it needs:
- ELF header (64 bytes)
- Section headers (find .dynamic, .gnu.version_r)
- .dynamic section (DT_NEEDED, DT_SONAME, DT_RPATH)
- .gnu.version_r section (GLIBC version requirements)

It **never** reads `.text` (code), `.rodata` (data), or `.debug_*` sections. This means analyzing a 500MB `libtorch_cuda.so` takes < 10ms because it reads < 10KB of the file.

### 16.4 Why Not `ldd`

`ldd` is the standard tool for checking shared library dependencies. But `ldd` is **unsafe on untrusted binaries** — it actually executes the dynamic linker, which runs constructor functions in the binary. A malicious `.so` file could execute arbitrary code through `ldd`.

PyBinaryGuard uses pure static analysis of the ELF format. It never executes any binary. It's safe to run on any file.

---

## 17. Real-World Impact Scenarios

### Scenario 1: University ML Lab

**Before PyBinaryGuard:**
- 50 students install PyTorch on shared server
- 15 get `Illegal instruction` (old CPU without AVX2)
- 10 get CUDA version mismatch
- Each spends 2-4 hours debugging
- TA spends entire week helping

**After PyBinaryGuard:**
```bash
# Added to lab setup script
pybinaryguard scan --ci --severity critical
# Exit code 2 → environment needs fixing before class starts
```
Time saved: ~100 student-hours per semester.

### Scenario 2: MLOps Pipeline

**Before:**
- Model training container works on dev GPU server (CUDA 12.2)
- Deployed to production server with CUDA 11.8
- PyTorch fails at inference time
- 3 hours of downtime, 2 engineers debugging

**After:**
```yaml
# In CI/CD pipeline
- name: Binary compatibility check
  run: pybinaryguard scan --ci --format json
  # Blocks deployment if critical issues found
```
Catches mismatch before deployment reaches production.

### Scenario 3: Jetson Edge Deployment

**Before:**
- Engineer `pip install torch` on Jetson Orin Nano
- Gets x86 wheel (PyPI default)
- `exec format error` on import
- Spends a day finding the correct ARM wheel URL

**After:**
```bash
$ pybinaryguard check torch

CRITICAL  torch 2.4.0
          Architecture mismatch
          Package contains x86_64 binaries but your system is aarch64.
          This binary will not execute on this CPU architecture.
          Fix: Install the ARM build from NVIDIA's wheel index
```
Diagnosis in 2 seconds instead of 1 day.

### Scenario 4: Docker Container Debugging

**Before:**
- Container works locally with `--gpus all`
- Fails in Kubernetes with `libcudart.so.12: cannot open`
- GPU devices not mounted properly
- Takes 4 hours to trace the issue

**After:**
```bash
$ pybinaryguard scan

WARNING   Container detected (Docker) but no GPU devices mounted
          GPU packages are installed but /dev/nvidia0 is not accessible.
          Fix: Run with --gpus all or --device /dev/nvidia0
```

---

## 18. Complete File Map

```
src/pybinaryguard/                    18,041 lines of production code
├── __init__.py                       Public API: scan(), check(), profile(), inspect()
├── __main__.py                       python -m pybinaryguard support
├── scanner.py                        Main orchestrator (904 lines)
│
├── probes/                           9 system probes
│   ├── base.py                       ProbeBase abstract class
│   ├── python_probe.py               Python version, ABI, debug build
│   ├── glibc_probe.py                GLIBC/musl detection (3-layer fallback)
│   ├── cpu_probe.py                  Architecture, flags, AVX/NEON
│   ├── gpu_probe.py                  GPU/CUDA/cuDNN (7-layer detection)
│   ├── os_probe.py                   OS, container, VM detection
│   ├── library_probe.py              ldconfig cache, LD_LIBRARY_PATH
│   ├── board_probe.py                Jetson/RPi/Coral/ARM SBC detection
│   ├── toolchain_probe.py            gcc/clang/cmake/make/rustc
│   └── venv_probe.py                 Virtual environment detection
│
├── analyzers/                        4 binary analyzers
│   ├── base.py                       AnalyzerBase abstract class
│   ├── elf_analyzer.py               Pure-Python ELF parser (754 lines)
│   ├── wheel_analyzer.py             WHEEL/METADATA/RECORD parsing
│   ├── symbol_analyzer.py            GLIBC/GLIBCXX version extraction
│   └── dependency_analyzer.py        DT_NEEDED resolution (ld.so algorithm)
│
├── rules/                            42 compatibility rules
│   ├── base.py                       Rule abstract base class
│   ├── engine.py                     Rule evaluation orchestrator
│   ├── builtin/
│   │   ├── glibc_rules.py            4 rules: GLIBC version, musl conflict
│   │   ├── cuda_rules.py             7 rules: driver, runtime, cuDNN, compute
│   │   ├── python_abi_rules.py       3 rules: ABI, version, debug/release
│   │   ├── arch_rules.py             1 rule: architecture mismatch
│   │   ├── cpu_rules.py              2 rules: AVX2, illegal instruction
│   │   ├── numpy_rules.py            1 rule: NumPy C API version
│   │   ├── container_rules.py        2 rules: GPU mount, driver mismatch
│   │   ├── board_profile_rules.py    5 rules: known broken, incompatible
│   │   ├── framework_rules.py        5 rules: PyTorch/TF/TRT/ONNX
│   │   ├── predictive_rules.py       2 rules: import prediction
│   │   ├── dependency_rules.py       2 rules: version conflict, missing
│   │   ├── source_build_rules.py     3 rules: sdist, compiler, headers
│   │   └── venv_rules.py             4 rules: system python, mixed env
│   └── data/                         Static compatibility data
│       ├── cuda_compat_matrix.json   Driver → max CUDA version
│       ├── pytorch_cuda_matrix.json  PyTorch → CUDA variants
│       ├── tensorflow_cuda_matrix.json TF → CUDA/cuDNN
│       ├── manylinux_policy.json     Manylinux tag → allowed libs
│       └── glibc_distro_map.json     Distro → GLIBC version
│
├── diagnostics/                      Error pattern matching
│   ├── explainer.py                  20+ regex patterns → root cause
│   ├── suggestions.py                OS-specific fix command generation
│   └── findings.py                   Filtering, sorting, deduplication
│
├── scoring/
│   └── engine.py                     Multi-dimensional health scoring v2
│
├── predictor/                        Import failure prediction
│   ├── predictor.py                  Main prediction entry point
│   ├── linker_simulator.py           Linux dynamic linker simulation
│   ├── dependency_graph.py           DT_NEEDED tree builder
│   └── resolver.py                   Symbol resolution
│
├── validators/
│   └── import_validator.py           Subprocess-based import testing
│
├── agent/                            AI agent SDK
│   ├── tool_interface.py             scan/check/simulate/doctor for agents
│   ├── schema.py                     OpenAI/MCP/JSON Schema export
│   ├── simulator.py                  Pre-install compatibility prediction
│   ├── recommender.py                Action generation + safety classification
│   └── guard.py                      Runtime import failure capture
│
├── frameworks/                       AI framework deep checks
│   ├── pytorch.py                    CUDA ABI, torchvision compat
│   ├── tensorflow.py                 Compute capability, AVX, CUDA matrix
│   ├── tensorrt.py                   Engine file validation
│   └── onnxruntime.py                Execution provider validation
│
├── plugins/                          Community extension system
│   ├── hooks.py                      HookRegistry (7 extension points)
│   ├── loader.py                     Entry point discovery
│   └── contrib/                      Built-in plugins
│       ├── jetson.py                 NVIDIA Jetson platform
│       ├── opencv.py                 OpenCV build flags
│       ├── tensorrt.py               TensorRT engines
│       └── gstreamer.py              GStreamer pipeline
│
├── snapshot/                         Environment lockfiles
│   ├── lockfile.py                   Lockfile data format
│   ├── generator.py                  SHA256 hashing, snapshot creation
│   └── verifier.py                   Snapshot verification
│
├── profiles/                         Embedded board profiles
│   ├── engine.py                     Profile matching engine
│   ├── jetson_nano.json              NVIDIA Jetson Nano
│   ├── jetson_orin_nx.json           NVIDIA Jetson Orin NX
│   ├── raspberry_pi_4.json           Raspberry Pi 4
│   ├── raspberry_pi_5.json           Raspberry Pi 5
│   └── coral_dev_board.json          Google Coral Dev Board
│
├── models/                           Core data structures
│   ├── system.py                     SystemProfile (70+ fields)
│   ├── package.py                    PackageBinaryInfo, SharedObjectInfo
│   ├── finding.py                    Finding, ScanReport
│   └── enums.py                      Severity, Architecture, ScanMode
│
├── cli/                              Command-line interface
│   ├── main.py                       Argument parser (10 commands)
│   ├── commands.py                   Command handlers
│   └── formatters.py                 Table/JSON/Minimal output
│
└── _compat/                          Compatibility layer
    └── __init__.py

tests/                                505 tests, all passing
├── test_scanner.py                   Scanner integration tests
├── test_probes.py                    Probe unit tests
├── test_rules.py                     Rule evaluation tests
├── test_elf_analyzer.py              ELF parser tests
├── test_wheel_analyzer.py            Wheel analyzer tests
├── test_scoring.py                   Health score tests
├── test_cli.py                       CLI command tests
├── test_models.py                    Data model tests
├── test_diagnostics.py               Error pattern tests
├── test_plugins.py                   Plugin system tests
├── test_agent_sdk.py                 Agent SDK tests
├── test_rule_engine.py               Rule engine tests
├── test_scan_modes.py                Scan mode tests
└── test_gap_features.py              Gap-closing feature tests
```

---

## Summary

PyBinaryGuard is a **18,041-line, zero-dependency, pure-Python** library that solves a problem no other tool addresses: detecting binary incompatibilities in Python environments before they crash programs.

It does this through:
- **9 system probes** that fingerprint the machine (CPU, GPU, GLIBC, OS, container, board, compiler, venv)
- **4 binary analyzers** that parse ELF shared objects without executing them
- **42 compatibility rules** that cross-reference system capabilities against package requirements
- **A multi-dimensional health score** across 4 categories with dynamic weighting
- **An import validator** that tests real imports in isolated subprocesses
- **A linker simulator** that predicts failures without running any code
- **An AI agent SDK** with OpenAI/MCP tool schemas
- **5 embedded board profiles** with known-good package stacks
- **A plugin system** with 7 extension points for community contributions

It exists because **the developer's code is correct** — but the compiled binaries underneath don't match the system. And until now, no tool caught that.

---

*PyBinaryGuard v1.0.3 — Binary Compatibility Intelligence for Python*
*Created by [Pothihai Selvan (@po-nuvai)](https://github.com/po-nuvai) — Applied Research Scientist at [Nuvai AI Solutions](https://nuvai.dev)*
