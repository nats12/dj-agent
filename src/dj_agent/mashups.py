"""Mashup idea engine — find tracks that work together for mashups.

Scores pairs by key compatibility, BPM similarity, and vocal/instrumental
complementarity (best mashup: one vocal + one instrumental).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .harmonic import camelot_distance, to_camelot
from .types import TrackInfo


@dataclass
class MashupScore:
    """Detailed mashup compatibility breakdown."""
    total: float
    harmonic: float
    bpm: float
    vocal_complement: float
    energy_match: float
    bpm_diff: float
    key_distance: int


@dataclass
class MashupSuggestion:
    """A suggested mashup partner."""
    track: TrackInfo
    score: MashupScore
    tip: str = ""  # e.g. "Pitch shift +2 to match key perfectly"


def find_mashup_candidates(
    track: TrackInfo,
    library: list[TrackInfo],
    tags: dict[str, dict[str, Any]] | None = None,
    bpm_tolerance_pct: float = 3.0,
    top_k: int = 10,
) -> list[MashupSuggestion]:
    """Find tracks from the library that could be mashed with the given track.

    Parameters
    ----------
    track : the anchor track
    library : all available tracks
    tags : dict mapping content_id → tag dict with 'vocal' (bool), 'energy' (int)
    bpm_tolerance_pct : max BPM difference as percentage (3% = pitchable range)
    top_k : number of results to return
    """
    tags = tags or {}
    track_tags = tags.get(track.db_content_id, {})
    suggestions: list[MashupSuggestion] = []

    for candidate in library:
        if candidate.db_content_id == track.db_content_id:
            continue

        cand_tags = tags.get(candidate.db_content_id, {})

        ms = score_mashup(track, candidate, track_tags, cand_tags)
        if ms.total < 0.3:
            continue

        # Generate tip
        tip = _generate_tip(track, candidate, ms)

        suggestions.append(MashupSuggestion(
            track=candidate,
            score=ms,
            tip=tip,
        ))

    suggestions.sort(key=lambda s: s.score.total, reverse=True)
    return suggestions[:top_k]


def score_mashup(
    track_a: TrackInfo,
    track_b: TrackInfo,
    tags_a: dict[str, Any] | None = None,
    tags_b: dict[str, Any] | None = None,
) -> MashupScore:
    """Score mashup compatibility between two tracks.

    Weights:
    - Harmonic compatibility: 0.35
    - BPM similarity: 0.25
    - Vocal/instrumental complementarity: 0.25
    - Energy match: 0.15
    """
    tags_a = tags_a or {}
    tags_b = tags_b or {}

    # Harmonic (Camelot distance)
    key_dist = camelot_distance(track_a.key, track_b.key)
    harmonic = {0: 1.0, 1: 0.85, 2: 0.5}.get(key_dist, 0.1)

    # BPM (within 3% = mashable with pitch adjustment)
    bpm_diff = abs(track_a.bpm - track_b.bpm)
    bpm_pct = (bpm_diff / max(track_a.bpm, 1.0)) * 100

    # Also check half/double tempo compatibility
    half_diff = abs(track_a.bpm - track_b.bpm * 2)
    double_diff = abs(track_a.bpm - track_b.bpm / 2)
    min_diff_pct = min(
        bpm_pct,
        (half_diff / max(track_a.bpm, 1.0)) * 100,
        (double_diff / max(track_a.bpm, 1.0)) * 100,
    )

    if min_diff_pct <= 1:
        bpm_score = 1.0
    elif min_diff_pct <= 3:
        bpm_score = 0.8
    elif min_diff_pct <= 6:
        bpm_score = 0.4
    else:
        bpm_score = 0.0

    # Vocal/instrumental complementarity
    # Best mashup: one vocal + one instrumental
    vocal_a = tags_a.get("vocal", None)
    vocal_b = tags_b.get("vocal", None)

    if vocal_a is not None and vocal_b is not None:
        if vocal_a != vocal_b:
            vocal_comp = 1.0  # perfect: one vocal, one instrumental
        elif not vocal_a and not vocal_b:
            vocal_comp = 0.6  # both instrumental: ok for layering
        else:
            vocal_comp = 0.2  # both vocal: risky, vocals clash
    else:
        vocal_comp = 0.5  # unknown

    # Energy match (similar energy levels work best for mashups)
    energy_a = tags_a.get("energy", 5)
    energy_b = tags_b.get("energy", 5)
    energy_diff = abs(energy_a - energy_b)
    if energy_diff <= 1:
        energy_match = 1.0
    elif energy_diff <= 2:
        energy_match = 0.7
    elif energy_diff <= 3:
        energy_match = 0.4
    else:
        energy_match = 0.1

    total = (
        0.35 * harmonic
        + 0.25 * bpm_score
        + 0.25 * vocal_comp
        + 0.15 * energy_match
    )

    return MashupScore(
        total=round(total, 3),
        harmonic=round(harmonic, 2),
        bpm=round(bpm_score, 2),
        vocal_complement=round(vocal_comp, 2),
        energy_match=round(energy_match, 2),
        bpm_diff=round(bpm_diff, 1),
        key_distance=key_dist,
    )


def _generate_tip(
    track: TrackInfo,
    candidate: TrackInfo,
    score: MashupScore,
) -> str:
    """Generate a human-readable tip for the mashup."""
    tips: list[str] = []

    if score.key_distance == 0:
        tips.append("Same key — perfect harmonic match")
    elif score.key_distance == 1:
        tips.append("Adjacent key — smooth blend")
    elif score.key_distance > 1:
        from .pitchshift import semitones_between_keys
        delta = semitones_between_keys(candidate.key, track.key)
        if delta != 0:
            direction = "up" if delta > 0 else "down"
            tips.append(f"Pitch shift {direction} {abs(delta)} semitones to match key")

    if score.bpm_diff > 0:
        tips.append(f"BPM diff: {score.bpm_diff:.0f} ({track.bpm:.0f} vs {candidate.bpm:.0f})")

    return ". ".join(tips)
