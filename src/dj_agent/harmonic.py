"""Harmonic mixing — Camelot wheel, key compatibility, transition suggestions."""

from __future__ import annotations

from .types import HarmonicSuggestion, TrackInfo


# ---------------------------------------------------------------------------
# Camelot wheel
# ---------------------------------------------------------------------------

# Camelot number (1-12) + letter (A=minor, B=major)
CAMELOT_MAP: dict[str, str] = {
    # Major keys → B
    "C major": "8B",  "Db major": "3B", "D major": "10B", "Eb major": "5B",
    "E major": "12B", "F major": "7B",  "Gb major": "2B", "G major": "9B",
    "Ab major": "4B", "A major": "11B", "Bb major": "6B", "B major": "1B",
    # Minor keys → A
    "C minor": "5A",  "Db minor": "12A","D minor": "7A",  "Eb minor": "2A",
    "E minor": "9A",  "F minor": "4A",  "Gb minor": "11A","G minor": "6A",
    "Ab minor": "1A", "A minor": "8A",  "Bb minor": "3A", "B minor": "10A",
    # Sharp equivalents
    "C# major": "3B", "D# major": "5B", "F# major": "2B",
    "G# major": "4B", "A# major": "6B",
    "C# minor": "12A","D# minor": "2A", "F# minor": "11A",
    "G# minor": "1A", "A# minor": "3A",
}

# Open Key notation (used by some DJ software)
OPEN_KEY_TO_CAMELOT: dict[str, str] = {
    "1d": "1B", "1m": "1A", "2d": "2B", "2m": "2A",
    "3d": "3B", "3m": "3A", "4d": "4B", "4m": "4A",
    "5d": "5B", "5m": "5A", "6d": "6B", "6m": "6A",
    "7d": "7B", "7m": "7A", "8d": "8B", "8m": "8A",
    "9d": "9B", "9m": "9A", "10d": "10B", "10m": "10A",
    "11d": "11B", "11m": "11A", "12d": "12B", "12m": "12A",
}


def to_camelot(key: str | None) -> str | None:
    """Convert any key representation to Camelot notation.

    Accepts: "8B", "C major", "Cm", "5A", "1d", None, etc.
    """
    if not key:
        return None
    key = key.strip()

    # Already Camelot?
    if _is_camelot(key):
        return key.upper()

    # Open Key?
    if key.lower() in OPEN_KEY_TO_CAMELOT:
        return OPEN_KEY_TO_CAMELOT[key.lower()]

    # Musical key?
    if key in CAMELOT_MAP:
        return CAMELOT_MAP[key]

    # Try adding "major" / "minor"
    for suffix in (" major", " minor"):
        full = key + suffix
        if full in CAMELOT_MAP:
            return CAMELOT_MAP[full]

    # Rekordbox format: "5B", "8A" etc (ScaleName)
    return key.upper() if _is_camelot(key) else None


def _is_camelot(key: str) -> bool:
    """Check if a string is valid Camelot notation."""
    key = key.strip().upper()
    if len(key) < 2 or len(key) > 3:
        return False
    return key[-1] in ("A", "B") and key[:-1].isdigit() and 1 <= int(key[:-1]) <= 12


def get_compatible_keys(camelot_key: str) -> list[tuple[str, str]]:
    """Return compatible keys with relationship labels.

    Returns list of ``(camelot_key, relationship)`` tuples:
    - "same": identical key
    - "adjacent_up": +1 on wheel (energy boost)
    - "adjacent_down": -1 on wheel
    - "relative": switch A↔B (relative major/minor)
    - "energy_boost": +2 on wheel (bold move)
    """
    key = camelot_key.upper().strip()
    if not _is_camelot(key):
        return []

    num = int(key[:-1])
    letter = key[-1]
    other_letter = "A" if letter == "B" else "B"

    # Camelot wheel wraps 1-12. Formula: ((num - 1 + delta) % 12) + 1
    up = ((num - 1 + 1) % 12) + 1     # +1: 1→2, 12→1
    down = ((num - 1 - 1) % 12) + 1   # -1: 1→12, 2→1
    boost = ((num - 1 + 2) % 12) + 1  # +2: 1→3, 11→1, 12→2

    return [
        (key, "same"),
        (f"{up}{letter}", "adjacent_up"),
        (f"{down}{letter}", "adjacent_down"),
        (f"{num}{other_letter}", "relative"),
        (f"{boost}{letter}", "energy_boost"),
    ]


def camelot_distance(key_a: str, key_b: str) -> int:
    """Circular distance on the Camelot wheel (0-6).

    0 = same position, 1 = adjacent, etc. Mode change (A↔B) adds 0.
    """
    a = to_camelot(key_a)
    b = to_camelot(key_b)
    if not a or not b:
        return 6  # unknown → max distance

    num_a = int(a[:-1])
    num_b = int(b[:-1])

    # Circular distance on 12-position wheel
    diff = abs(num_a - num_b)
    return min(diff, 12 - diff)


# ---------------------------------------------------------------------------
# Transition scoring
# ---------------------------------------------------------------------------

def score_transition(
    track_a: TrackInfo,
    track_b: TrackInfo,
    energy_a: int = 0,
    energy_b: int = 0,
) -> float:
    """Score a transition between two tracks (0.0 = terrible, 1.0 = perfect).

    Weights: harmonic 0.35, BPM 0.25, energy 0.20, genre 0.20.
    Pass energy values (1-10) for accurate scoring; defaults to neutral 0.5.
    """
    # Harmonic compatibility
    key_dist = camelot_distance(track_a.key, track_b.key)
    if key_dist == 0:
        harmonic = 1.0
    elif key_dist == 1:
        harmonic = 0.8
    elif key_dist == 2:
        harmonic = 0.4
    else:
        harmonic = 0.0

    # BPM compatibility (symmetric — uses max of both BPMs as denominator)
    max_bpm = max(track_a.bpm, track_b.bpm, 1.0)
    bpm_diff_pct = abs(track_a.bpm - track_b.bpm) / max_bpm * 100
    if bpm_diff_pct <= 3:
        bpm_score = 1.0
    elif bpm_diff_pct <= 6:
        bpm_score = 0.6
    elif bpm_diff_pct <= 10:
        bpm_score = 0.2
    else:
        bpm_score = 0.0

    # Genre compatibility (simple: same genre = 1.0, different = 0.3)
    genre_a = (track_a.genre or "").lower().strip()
    genre_b = (track_b.genre or "").lower().strip()
    genre_score = 1.0 if genre_a == genre_b else 0.3

    # Energy flow (smooth transitions: difference ≤ 2)
    if energy_a and energy_b:
        e_diff = abs(energy_a - energy_b)
        energy_score = max(0.0, 1.0 - e_diff * 0.2)  # 0 diff=1.0, 5 diff=0.0
    else:
        energy_score = 0.5  # neutral when energy unknown

    return (
        0.35 * harmonic
        + 0.25 * bpm_score
        + 0.20 * energy_score
        + 0.20 * genre_score
    )


# ---------------------------------------------------------------------------
# Harmonic mixing suggestions
# ---------------------------------------------------------------------------

def suggest_harmonic_transitions(
    track: TrackInfo,
    library: list[TrackInfo],
    bpm_tolerance_pct: float = 6.0,
    top_k: int = 10,
) -> list[HarmonicSuggestion]:
    """Suggest tracks from the library that mix well with the given track."""
    current_camelot = to_camelot(track.key)
    if not current_camelot:
        return []

    compatible = get_compatible_keys(current_camelot)
    compatible_keys = {k for k, _ in compatible}

    suggestions: list[HarmonicSuggestion] = []
    for candidate in library:
        if candidate.db_content_id == track.db_content_id:
            continue

        cand_camelot = to_camelot(candidate.key)
        if not cand_camelot or cand_camelot not in compatible_keys:
            continue

        bpm_diff = abs(candidate.bpm - track.bpm)
        bpm_pct = (bpm_diff / max(track.bpm, 1.0)) * 100
        if bpm_pct > bpm_tolerance_pct:
            continue

        # Find relationship
        relation = "same"
        for k, rel in compatible:
            if k == cand_camelot:
                relation = rel
                break

        score = score_transition(track, candidate)

        suggestions.append(HarmonicSuggestion(
            track=candidate,
            score=score,
            key_relation=relation,
            bpm_diff=bpm_diff,
            energy_diff=0,
        ))

    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions[:top_k]
