# Changelog

All notable changes to PyBinaryGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
