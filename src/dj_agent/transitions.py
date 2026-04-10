"""Transition compatibility scoring — multi-factor analysis for DJ mixing."""

from __future__ import annotations

from .harmonic import camelot_distance, to_camelot
from .types import TrackInfo


def score_transition(
    track_a: TrackInfo,
    track_b: TrackInfo,
    energy_a: int = 0,
    energy_b: int = 0,
    vocal_a: bool = False,
    vocal_b: bool = False,
) -> dict[str, float]:
    """Score a transition between two tracks with detailed breakdown.

    Returns a dict with individual factor scores and the weighted total.

    Weights: harmonic 0.30, BPM 0.25, energy 0.20, genre 0.15, vocal 0.10
    """
    # Harmonic
    key_dist = camelot_distance(track_a.key, track_b.key)
    harmonic = {0: 1.0, 1: 0.85, 2: 0.4}.get(key_dist, 0.0)

    # BPM
    bpm_diff = abs(track_a.bpm - track_b.bpm)
    bpm_pct = (bpm_diff / max(track_a.bpm, 1.0)) * 100
    if bpm_pct <= 2:
        bpm = 1.0
    elif bpm_pct <= 4:
        bpm = 0.8
    elif bpm_pct <= 6:
        bpm = 0.5
    elif bpm_pct <= 10:
        bpm = 0.2
    else:
        bpm = 0.0

    # Energy flow (smooth = diff ≤ 2, jarring = diff > 3)
    if energy_a and energy_b:
        e_diff = abs(energy_a - energy_b)
        if e_diff == 0:
            energy = 1.0
        elif e_diff <= 1:
            energy = 0.9
        elif e_diff <= 2:
            energy = 0.7
        elif e_diff <= 3:
            energy = 0.4
        else:
            energy = 0.1
    else:
        energy = 0.5  # unknown

    # Genre
    ga = (track_a.genre or "").lower().strip()
    gb = (track_b.genre or "").lower().strip()
    if ga and gb and ga == gb:
        genre = 1.0
    elif ga and gb and (ga in gb or gb in ga):
        genre = 0.7  # subgenre match
    else:
        genre = 0.3

    # Vocal overlap risk
    if vocal_a and vocal_b:
        vocal = 0.1  # two vocals = bad
    elif vocal_a or vocal_b:
        vocal = 0.8  # one vocal = fine
    else:
        vocal = 1.0  # both instrumental = perfect

    total = (
        0.30 * harmonic
        + 0.25 * bpm
        + 0.20 * energy
        + 0.15 * genre
        + 0.10 * vocal
    )

    return {
        "total": round(total, 3),
        "harmonic": round(harmonic, 2),
        "bpm": round(bpm, 2),
        "energy": round(energy, 2),
        "genre": round(genre, 2),
        "vocal": round(vocal, 2),
        "bpm_diff": round(bpm_diff, 1),
        "key_distance": key_dist,
    }


def rate_transition(score: float) -> str:
    """Convert a total score to a green/yellow/red rating."""
    if score >= 0.7:
        return "green"
    elif score >= 0.4:
        return "yellow"
    return "red"
