"""Beat grid verification — detect wrong BPM (half/double) and tempo drift.

Uses madmom neural beat tracking when available, falls back to librosa.
Can read Rekordbox ANLZ beat grids via pyrekordbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np


# Genre-aware BPM ranges for half/double detection
GENRE_BPM_RANGES: dict[str, tuple[float, float]] = {
    "ambient": (60, 100),
    "downtempo": (80, 115),
    "deep house": (118, 125),
    "house": (118, 132),
    "tech house": (124, 132),
    "techno": (125, 150),
    "hard techno": (140, 165),
    "trance": (128, 145),
    "drum and bass": (160, 180),
    "dnb": (160, 180),
    "bouncy": (140, 155),
    "acid": (130, 150),
    "disco": (115, 130),
    "psytrance": (138, 150),
    "dubstep": (138, 152),
    "breaks": (120, 140),
}

DEFAULT_BPM_RANGE = (80, 180)


def verify_bpm(
    path: str | Path,
    rekordbox_bpm: float,
    genre: str | None = None,
    skip_genre_correction: bool = False,
) -> dict[str, Any]:
    """Compare Rekordbox BPM against detected BPM.

    Parameters
    ----------
    skip_genre_correction : if True, don't apply half/double genre-range
        correction. Useful for transition tracks or multi-genre sets where
        "illegal" BPMs are intentional (e.g., 100 BPM in a Techno set).

    Returns a dict with:
    - ``detected_bpm``: BPM from audio analysis
    - ``rekordbox_bpm``: the value from Rekordbox
    - ``match``: True if they agree (within 2%)
    - ``issue``: None, "half", "double", "wrong", or "drift"
    - ``suggested_bpm``: corrected BPM if issue detected
    """
    path = Path(path)
    detected = _detect_bpm(path)

    if detected == 0:
        return {
            "detected_bpm": 0,
            "rekordbox_bpm": rekordbox_bpm,
            "match": False,
            "issue": "detection_failed",
            "suggested_bpm": rekordbox_bpm,
        }

    # Fix half/double using genre ranges (unless explicitly skipped)
    if not skip_genre_correction:
        detected = _fix_half_double(detected, genre)

    # Compare
    diff_pct = abs(detected - rekordbox_bpm) / max(rekordbox_bpm, 1.0) * 100

    if diff_pct <= 2.0:
        return {
            "detected_bpm": round(detected, 1),
            "rekordbox_bpm": rekordbox_bpm,
            "match": True,
            "issue": None,
            "suggested_bpm": rekordbox_bpm,
        }

    # Check if Rekordbox has a half BPM error (Rekordbox says 65, should be 130)
    doubled_rb = rekordbox_bpm * 2
    if abs(detected - doubled_rb) / max(doubled_rb, 1) * 100 <= 2:
        return {
            "detected_bpm": round(detected, 1),
            "rekordbox_bpm": rekordbox_bpm,
            "match": False,
            "issue": "half",
            "suggested_bpm": round(detected, 1),
        }

    # Check if Rekordbox has a double BPM error (Rekordbox says 260, should be 130)
    halved_rb = rekordbox_bpm / 2
    if abs(detected - halved_rb) / max(halved_rb, 1) * 100 <= 2:
        return {
            "detected_bpm": round(detected, 1),
            "rekordbox_bpm": rekordbox_bpm,
            "match": False,
            "issue": "double",
            "suggested_bpm": round(detected, 1),
        }

    return {
        "detected_bpm": round(detected, 1),
        "rekordbox_bpm": rekordbox_bpm,
        "match": False,
        "issue": "wrong",
        "suggested_bpm": round(detected, 1),
    }


def detect_tempo_drift(path: str | Path, window_sec: float = 10.0) -> list[dict]:
    """Detect tempo changes within a track.

    Returns a list of segments where BPM differs significantly from the
    track's dominant BPM.  Empty list = stable tempo.
    """
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    window_samples = int(window_sec * sr)
    segments: list[dict] = []

    tempos: list[float] = []
    times: list[float] = []

    for start in range(0, len(y) - window_samples, window_samples // 2):
        chunk = y[start: start + window_samples]
        tempo, _ = librosa.beat.beat_track(y=chunk, sr=sr)
        t = float(tempo)
        if t > 0:
            tempos.append(t)
            times.append(start / sr)

    if len(tempos) < 3:
        return []

    median_tempo = float(np.median(tempos))

    for i, (t, bpm) in enumerate(zip(times, tempos)):
        drift_pct = abs(bpm - median_tempo) / median_tempo * 100
        if drift_pct > 3.0:
            segments.append({
                "time": round(t, 1),
                "bpm": round(bpm, 1),
                "median_bpm": round(median_tempo, 1),
                "drift_pct": round(drift_pct, 1),
            })

    return segments


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _detect_bpm(path: Path) -> float:
    """Detect BPM using best available engine.

    Priority: Beat This! (Transformer) → madmom (RNN) → librosa (onset).
    """
    import logging
    _log = logging.getLogger(__name__)

    try:
        return _beat_this_bpm(path)
    except ImportError:
        pass  # not installed — expected
    except Exception as e:
        _log.warning("Beat This! failed, falling back to madmom: %s", e)

    try:
        return _madmom_bpm(path)
    except ImportError:
        pass
    except Exception as e:
        _log.warning("madmom failed, falling back to librosa: %s", e)

    return _librosa_bpm(path)


_beat_this_model = None
_beat_this_lock = __import__("threading").Lock()


def _get_beat_this():
    """Cached Beat This! model (loaded once, thread-safe)."""
    global _beat_this_model
    if _beat_this_model is not None:
        return _beat_this_model
    with _beat_this_lock:
        if _beat_this_model is not None:
            return _beat_this_model
        from beat_this.inference import File2Beats  # type: ignore[import-untyped]
        _beat_this_model = File2Beats(checkpoint_path="final0", device="cpu")
        return _beat_this_model


def _beat_this_bpm(path: Path) -> float:
    """Beat tracking using Beat This! (ISMIR 2024 Transformer, SOTA).

    Requires: pip install https://github.com/CPJKU/beat_this/archive/main.zip
    """
    file2beats = _get_beat_this()
    beats, downbeats = file2beats(str(path))

    if len(beats) < 4:
        return 0.0

    intervals = np.diff(beats)
    median_interval = float(np.median(intervals))
    return 60.0 / median_interval if median_interval > 0 else 0.0


def _madmom_bpm(path: Path) -> float:
    from madmom.features.beats import (  # type: ignore[import-untyped]
        DBNBeatTrackingProcessor,
        RNNBeatProcessor,
    )
    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(str(path))
    beats = proc(act)
    if len(beats) < 4:
        return 0.0
    intervals = np.diff(beats)
    median_interval = float(np.median(intervals))
    return 60.0 / median_interval if median_interval > 0 else 0.0


def _librosa_bpm(path: Path) -> float:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, "__len__"):
        return float(np.mean(tempo))
    return float(tempo)


def _fix_half_double(bpm: float, genre: str | None) -> float:
    """Fix half/double BPM using nearest-octave selection.

    Instead of oscillating multiply/divide loops, evaluates all candidates
    (bpm, bpm*2, bpm/2) and picks whichever is closest to the genre range.
    """
    if bpm <= 0:
        return bpm

    if genre:
        genre_lower = genre.lower().strip()
        lo, hi = GENRE_BPM_RANGES.get(genre_lower, DEFAULT_BPM_RANGE)
    else:
        lo, hi = DEFAULT_BPM_RANGE

    centroid = (lo + hi) / 2.0
    candidates = [bpm, bpm * 2, bpm / 2]

    # Pick the candidate closest to the genre centroid
    best = min(candidates, key=lambda b: abs(b - centroid) if b > 0 else float("inf"))

    return best
