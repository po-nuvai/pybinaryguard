"""Predictive failure engine for runtime simulation.

This module predicts ImportErrors and runtime failures before they occur
by simulating dynamic linker resolution and symbol dependencies.
"""

from __future__ import annotations

from .resolver import SymbolResolver, resolve_symbol
from .dependency_graph import DependencyGraph, build_dependency_graph
from .linker_simulator import LinkerSimulator, simulate_import
from .predictor import predict_import_failures, PredictedFailure

__all__ = [
    "SymbolResolver",
    "resolve_symbol",
    "DependencyGraph",
    "build_dependency_graph",
    "LinkerSimulator",
    "simulate_import",
    "predict_import_failures",
    "PredictedFailure",
]
