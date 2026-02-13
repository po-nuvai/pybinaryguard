"""Environment snapshot and verification system.

Captures complete binary compatibility state including hashes, GPU stack,
and system libraries - information that pip freeze/conda export don't provide.
"""

from __future__ import annotations

from .generator import SnapshotGenerator, create_snapshot
from .verifier import SnapshotVerifier, verify_snapshot
from .lockfile import Lockfile, load_lockfile

__all__ = [
    "SnapshotGenerator",
    "create_snapshot",
    "SnapshotVerifier",
    "verify_snapshot",
    "Lockfile",
    "load_lockfile",
]
