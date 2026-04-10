"""Duplicate detection with chunked hashing and blocking fuzzy match.

Fixes over the original:
- Chunked hashing (never loads full file into memory)
- Artist-prefix blocking reduces O(n²) to O(n × block_size)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz  # type: ignore[import-untyped]

from .config import DuplicateConfig
from .types import TrackInfo


# ---------------------------------------------------------------------------
# Pass 1: Exact file hash (chunked)
# ---------------------------------------------------------------------------

def hash_file_chunked(path: str | Path, chunk_size: int = 65536) -> str:
    """Stream-hash a file in chunks.  Never loads the whole file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(
    tracks: list[TrackInfo],
    config: DuplicateConfig | None = None,
) -> list[list[TrackInfo]]:
    """Group tracks that are byte-identical files."""
    chunk = config.hash_chunk_size if config else 65536
    hashes: dict[str, list[TrackInfo]] = {}
    for t in tracks:
        p = Path(t.path)
        if not p.exists():
            continue
        h = hash_file_chunked(p, chunk)
        hashes.setdefault(h, []).append(t)
    return [group for group in hashes.values() if len(group) > 1]


# ---------------------------------------------------------------------------
# Pass 3: Fuzzy metadata match (with blocking)
# ---------------------------------------------------------------------------

def _normalise_for_blocking(text: str) -> str:
    """Lower-case, strip non-alpha, return first 3 chars."""
    cleaned = "".join(c for c in text.lower() if c.isalpha())
    return cleaned[:3] if len(cleaned) >= 3 else cleaned


def find_fuzzy_duplicates(
    tracks: list[TrackInfo],
    config: DuplicateConfig | None = None,
) -> list[tuple[TrackInfo, TrackInfo, int]]:
    """Find probable duplicates using artist-prefix blocking + fuzzy match.

    Returns a list of ``(track_a, track_b, similarity_ratio)`` tuples.
    """
    if config is None:
        from .config import get_config
        config = get_config().duplicates

    threshold = config.fuzzy_threshold
    dur_tol = config.duration_tolerance_sec

    # Build blocks by normalised artist prefix
    blocks: dict[str, list[TrackInfo]] = {}
    for t in tracks:
        key = _normalise_for_blocking(t.artist) if t.artist else "zzz"
        blocks.setdefault(key, []).append(t)

    results: list[tuple[TrackInfo, TrackInfo, int]] = []
    for block in blocks.values():
        for i, a in enumerate(block):
            a_label = f"{a.artist} - {a.title}".lower()
            for b in block[i + 1 :]:
                b_label = f"{b.artist} - {b.title}".lower()
                ratio = fuzz.ratio(a_label, b_label)
                if ratio >= threshold:
                    if abs(a.duration - b.duration) < dur_tol:
                        results.append((a, b, ratio))
    return results


# ---------------------------------------------------------------------------
# Pass 2: Acoustic fingerprint (catches transcodes, alternate masters)
# ---------------------------------------------------------------------------

def find_fingerprint_duplicates(
    tracks: list[TrackInfo],
    threshold: float = 0.8,
) -> list[tuple[TrackInfo, TrackInfo, float]]:
    """Find near-duplicates using acoustic fingerprinting.

    Catches the duplicates DJs actually care about: transcodes (MP3→FLAC),
    alternate masters, edits with near-identical audio, and same-audio
    different-metadata files.

    Requires: pip install pyacoustid (+ fpcalc binary)
    Returns list of (track_a, track_b, fingerprint_similarity) tuples.
    """
    try:
        import acoustid  # type: ignore[import-untyped]
    except ImportError:
        return []  # pyacoustid not installed — skip fingerprint pass

    import logging
    _log = logging.getLogger(__name__)

    # Generate fingerprints
    fps: list[tuple[TrackInfo, str]] = []
    for t in tracks:
        p = Path(t.path)
        if not p.exists():
            continue
        try:
            duration, fingerprint = acoustid.fingerprint_file(str(p))
            if fingerprint:
                fps.append((t, fingerprint))
        except Exception:
            continue

    if len(fps) < 2:
        return []

    _log.info("Fingerprinted %d/%d tracks for duplicate detection", len(fps), len(tracks))

    # Compare fingerprints (O(n²) but fingerprint comparison is fast)
    results: list[tuple[TrackInfo, TrackInfo, float]] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            try:
                score = acoustid.compare_fingerprints(fps[i][1], fps[j][1])
                if score and score >= threshold:
                    results.append((fps[i][0], fps[j][0], float(score)))
            except Exception:
                continue

    return results


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def find_all_duplicates(
    tracks: list[TrackInfo],
    config: DuplicateConfig | None = None,
) -> dict[str, Any]:
    """Run all duplicate detection passes and return grouped results.

    Pass 1: Exact file hash (byte-identical)
    Pass 2: Acoustic fingerprint (transcodes, alternate masters)
    Pass 3: Fuzzy metadata (similar artist+title)
    """
    if config is None:
        from .config import get_config
        config = get_config().duplicates

    return {
        "exact": find_exact_duplicates(tracks, config),
        "fingerprint": find_fingerprint_duplicates(tracks),
        "fuzzy": find_fuzzy_duplicates(tracks, config),
    }
