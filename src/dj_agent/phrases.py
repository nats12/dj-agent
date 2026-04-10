"""Phrase detection — identify 4/8/16/32 bar musical phrases.

Uses madmom for downbeat tracking when available, falls back to
librosa beat tracking with bar grouping.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .types import Phrase


def detect_phrases(
    path: str | Path,
    bpm: float,
    expected_bars_per_phrase: int = 8,
) -> list[Phrase]:
    """Detect musical phrases in a track.

    Priority: All-In-One (Transformer structure analysis) → madmom (RNN
    downbeat tracking) → librosa (beat tracking fallback).
    """
    path = Path(path)

    import logging
    _log = logging.getLogger(__name__)

    try:
        return _allin1_phrases(path)
    except ImportError:
        pass
    except Exception as e:
        _log.warning("All-In-One failed, falling back to madmom: %s", e)

    try:
        return _madmom_phrases(path, expected_bars_per_phrase)
    except ImportError:
        pass
    except Exception as e:
        _log.warning("madmom failed, falling back to librosa: %s", e)

    return _librosa_phrases(path, bpm, expected_bars_per_phrase)


# ---------------------------------------------------------------------------
# All-In-One (preferred — Transformer-based functional structure analysis)
# ---------------------------------------------------------------------------

def _allin1_phrases(path: Path) -> list[Phrase]:
    """Use All-In-One model for functional structure analysis.

    Returns phrases with semantic labels (intro, verse, chorus, bridge, outro).
    Requires: pip install allin1
    """
    import allin1  # type: ignore[import-untyped]

    result = allin1.analyze(str(path))

    # Map allin1 segment labels to DJ labels with confidence scores.
    # These are heuristic mappings — source labels are preserved separately.
    label_map: dict[str, tuple[str, float]] = {
        "intro": ("intro", 0.95),        # intro→intro is reliable
        "verse": ("build", 0.5),          # verse→build is a guess
        "chorus": ("drop", 0.7),          # chorus=drop in EDM, less so in other genres
        "drop": ("drop", 0.95),           # drop→drop is reliable
        "bridge": ("breakdown", 0.6),     # bridge→breakdown is approximate
        "breakdown": ("breakdown", 0.9),  # breakdown→breakdown is reliable
        "outro": ("outro", 0.95),         # outro→outro is reliable
        "inst": ("build", 0.4),           # instrumental section — weak mapping
        "solo": ("build", 0.3),           # solo→build is a guess
    }

    phrases: list[Phrase] = []
    for segment in result.segments:
        src_label = segment.label.lower()
        dj_label, conf = label_map.get(src_label, ("build", 0.3))
        phrases.append(Phrase(
            start_ms=int(segment.start * 1000),
            end_ms=int(segment.end * 1000),
            bar_count=0,  # allin1 doesn't report bar counts
            label=dj_label,
            energy=0.0,  # filled by caller if needed
            confidence=conf,
            source_label=src_label,  # preserve original model label
        ))

    return phrases


# ---------------------------------------------------------------------------
# madmom (preferred)
# ---------------------------------------------------------------------------

def _madmom_phrases(path: Path, bars_per_phrase: int) -> list[Phrase]:
    """Use madmom RNN downbeat processor for phrase detection."""
    from madmom.features.downbeats import (  # type: ignore[import-untyped]
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    act = RNNDownBeatProcessor()(str(path))
    downbeats = proc(act)

    # downbeats: array of [time, beat_position] where beat_position 1 = downbeat
    bar_starts = [float(db[0]) for db in downbeats if db[1] == 1]

    if len(bar_starts) < 2:
        return []

    # Load audio once for energy analysis
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    return _group_bars_into_phrases(bar_starts, bars_per_phrase, y, sr)


# ---------------------------------------------------------------------------
# librosa fallback
# ---------------------------------------------------------------------------

def _librosa_phrases(
    path: Path,
    bpm: float,
    bars_per_phrase: int,
) -> list[Phrase]:
    """Estimate phrases from librosa beat tracking."""
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 8:
        return []

    # Group beats into bars (4 beats per bar for 4/4 time)
    bar_starts: list[float] = []
    for i in range(0, len(beat_times), 4):
        bar_starts.append(float(beat_times[i]))

    return _group_bars_into_phrases(bar_starts, bars_per_phrase, y, sr)


# ---------------------------------------------------------------------------
# Shared: group bars into phrases + classify
# ---------------------------------------------------------------------------

def _group_bars_into_phrases(
    bar_starts: list[float],
    bars_per_phrase: int,
    y: np.ndarray,
    sr: int,
) -> list[Phrase]:
    """Group bar start times into phrases and classify by energy."""
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    duration = librosa.get_duration(y=y, sr=sr)

    phrases: list[Phrase] = []

    for i in range(0, len(bar_starts), bars_per_phrase):
        start = bar_starts[i]
        end_idx = min(i + bars_per_phrase, len(bar_starts) - 1)
        end = bar_starts[end_idx] if end_idx > i else duration
        actual_bars = min(bars_per_phrase, end_idx - i)

        if actual_bars < 2:
            continue

        # Compute energy for this phrase
        sf = librosa.time_to_frames(start, sr=sr, hop_length=512)
        ef = min(librosa.time_to_frames(end, sr=sr, hop_length=512), len(rms))
        energy = float(np.mean(rms[sf:ef])) if ef > sf else 0.0

        phrases.append(Phrase(
            start_ms=int(start * 1000),
            end_ms=int(end * 1000),
            bar_count=actual_bars,
            label="",
            energy=energy,
        ))

    if not phrases:
        return phrases

    # Normalise energy
    max_e = max(p.energy for p in phrases)
    if max_e > 0:
        for p in phrases:
            p.energy = p.energy / max_e

    # Classify phrases by position and energy
    _label_phrases(phrases, duration)

    return phrases


def _label_phrases(phrases: list[Phrase], duration: float) -> None:
    """Assign labels to phrases based on position and energy."""
    n = len(phrases)
    for i, p in enumerate(phrases):
        pos_ratio = (p.start_ms / 1000.0) / max(duration, 1.0)

        if pos_ratio < 0.10:
            p.label = "intro"
        elif pos_ratio > 0.85:
            p.label = "outro"
        elif p.energy > 0.75:
            if i > 0 and phrases[i - 1].energy < 0.5:
                p.label = "drop"  # low-to-high transition = drop
            else:
                p.label = "peak"  # sustained high energy
        elif p.energy < 0.35:
            if pos_ratio > 0.15:
                p.label = "breakdown"
            else:
                p.label = "intro"
        elif i > 0 and phrases[i - 1].energy < p.energy:
            p.label = "build"
        else:
            p.label = "sustain"  # mid-energy, not rising
