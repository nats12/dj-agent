"""Weighted energy calibration — no double-counting between global and genre offsets."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


def recalculate_calibration(
    corrections: list[dict[str, Any]],
    min_for_genre: int = 3,
) -> dict[str, Any]:
    """Recompute calibration offsets from a list of user corrections.

    Each correction dict must contain at least::

        {"original_energy": int, "corrected_energy": int, "genre": str}

    Genre offsets are computed **independently** (not as a delta from global).
    The global offset is the median of corrections for genres that have fewer
    than *min_for_genre* samples.  This prevents double-counting.
    """
    genre_groups: dict[str, list[float]] = {}
    for c in corrections:
        delta = float(c["corrected_energy"] - c["original_energy"])
        genre = (c.get("genre") or "").lower().strip()
        genre_groups.setdefault(genre, []).append(delta)

    genre_offsets: dict[str, float] = {}
    ungrouped: list[float] = []

    for genre, deltas in genre_groups.items():
        if len(deltas) >= min_for_genre:
            genre_offsets[genre] = float(np.median(deltas))
        else:
            ungrouped.extend(deltas)

    global_offset = float(np.median(ungrouped)) if ungrouped else 0.0

    return {
        "global_offset": global_offset,
        "genre_offsets": genre_offsets,
        "updated_at": datetime.now().isoformat(),
    }


def apply_calibration(
    raw_energy: float,
    genre: str,
    calibration: dict[str, Any],
) -> int:
    """Apply calibration offset and clamp to 1-10.

    Uses the genre-specific offset if available; otherwise falls back to the
    global offset.  **Never sums both.**
    """
    genre_lower = genre.lower().strip()
    genre_offsets = calibration.get("genre_offsets", {})

    if genre_lower in genre_offsets:
        offset = genre_offsets[genre_lower]
    else:
        offset = calibration.get("global_offset", 0.0)

    return int(np.clip(np.round(raw_energy + offset), 1, 10))
