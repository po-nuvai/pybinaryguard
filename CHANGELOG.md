# Changelog

All notable changes to PyBinaryGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Maintained by [Pothihai Selvan (@po-nuvai)](https://github.com/po-nuvai) at [Nuvai AI Solutions](https://nuvai.dev).

## [1.0.3] - 2026-08-04

### Fixed

- **Contrib plugins now load correctly.** The `pyproject.toml` declared
  entry points with a `:register` suffix (e.g.
  `pybinaryguard.plugins.contrib.jetson:register`), which caused
  `ep.load()` to return the `register` function directly. The loader
  then tried to look up a `.register` attribute on that function,
  failed, and skipped the plugin — printing the noisy warning
  `Plugin entry point 'jetson' resolved to <function register at 0x...>
  but it has no 'register' function; skipping` at startup. Every
  built-in contrib plugin (Jetson board detection, TensorRT probes,
  OpenCV, GStreamer) was silently disabled as a result.

  Two-part fix: the loader now accepts both conventions (callable
  return OR module with a `.register` attribute), and the redundant
  `[project.entry-points]` block was removed since the built-in
  contrib plugins are already loaded via
  `_load_builtin_contrib_plugins` and would otherwise register twice.

## [1.0.2] - 2026-07-19

### Changed

- **Correct canonical GitHub URLs** in `[project.urls]`. Homepage, Repository,
  Issues, and Changelog now point to `github.com/po-nuvai/pybinaryguard`
  directly instead of `github.com/Nuvai/pybinaryguard` (a redirect). PyPI
  sidebar links now resolve without a redirect hop.
- **Author metadata refined.** Added `[maintainers]` field crediting Nuvai
  AI Solutions; normalised author name to "Pothihai Selvan".
- **README install section rewritten** — `pipx` recommended, with a PEP 668
  note for Ubuntu 24.04+ / Debian 12+ users where system-wide `pip install`
  is blocked by default.
- **Author section expanded** in README with GitHub handle, company link,
  and contact info.

### Fixed

- Fixed hardcoded version strings in CLI banner and JSON output — every
  release will now display the correct version in `pybinaryguard --version`,
  scan headers, and JSON export.

No code / rule-engine changes in this release. Existing users on 1.0.1 can
safely defer the upgrade; new installs will pick up 1.0.2 automatically.

## [1.0.1] - 2026-07-19

### Fixed

- **`PYTHON_ABI_MISMATCH` false positives.** The `PythonProbe` returned the
  raw `sysconfig.SOABI` string (e.g. `"cpython-312-x86_64-linux-gnu"`) as the
  system ABI tag, but wheel `WHEEL` metadata uses PEP 425 tags
  (e.g. `"cp312"`). Direct string comparison flagged every C-extension wheel
  as CRITICAL on healthy environments, dropping the health score by ~40
  points. The probe now returns the PEP 425 wheel-style tag
  (`cp312`, `cp312d` for debug builds, `pypy39_pp73` for PyPy, etc.), which
  matches the namespace used by real wheels.

## [1.0.0] - 2025-02-13

### Added

- **Core Scanner** with full environment scanning and per-package checks
- **System Probes**: Python, CPU, GLIBC, GPU/CUDA, OS, library, and board detection
- **Binary Analyzers**: ELF header parsing, wheel metadata extraction, symbol resolution, dependency analysis
- **20+ Built-in Rules**: GLIBC version, CUDA ABI, CPU instruction sets, architecture mismatches, NumPy ABI, container issues, Python ABI
- **Board Profile Engine**: NVIDIA Jetson (Nano, TX2, Xavier, Orin) and Raspberry Pi (3B, 4B, 5, Zero 2W) support
- **AI Framework Inspection**: PyTorch CUDA ABI, PyTorch+TorchVision matrix, TensorFlow compute capability, TensorRT version, ONNX Runtime providers
- **Predictive Failure Engine**: Import error prediction, unresolved dependency chain detection
- **Health Scoring v2**: Multi-dimensional weighted scoring across 4 categories (Binary Stability 35%, GPU Compatibility 30%, Dependency Health 25%, Platform Risk 10%)
- **Scan Modes**: Fast (metadata only, <1s), Standard (full binary analysis), Deep (symbol resolution + SHA256 hashes)
- **Agent SDK**: Structured output API, action recommendation engine with safety classification, tool schema export (OpenAI/MCP/JSON Schema), pre-install compatibility simulator, runtime import guard
- **Environment Snapshots**: Create and verify binary environment lockfiles
- **Plugin System**: Extensible probes, analyzers, and rules via entry points (Jetson, TensorRT, OpenCV, GStreamer built-in)
- **CLI**: Full command suite (scan, check, profile, doctor, inspect, snapshot, verify, simulate, export-tool-schema)
- **Output Formats**: Table (colored), JSON, minimal
- **CI Mode**: Strict exit codes, minimal output, no color
- **450 tests** with comprehensive coverage
