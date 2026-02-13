"""Board profile system for embedded device intelligence."""

from __future__ import annotations

from .engine import BoardProfile, ProfileEngine, load_profile, match_board_profile

__all__ = [
    "BoardProfile",
    "ProfileEngine",
    "load_profile",
    "match_board_profile",
]
