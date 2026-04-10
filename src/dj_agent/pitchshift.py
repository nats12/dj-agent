"""Pitch shifting — change a song's key by N semitones.

Uses pedalboard.PitchShift (C++ backend, preserves tempo).
Includes key-aware mode that calculates the semitone delta automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .harmonic import to_camelot


# Chromatic note order for semitone calculation
_CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic equivalents
_ENHARMONIC: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B",
}


def pitch_shift(
    path: str | Path,
    semitones: float,
    output_path: str | Path | None = None,
) -> Path:
    """Shift pitch by N semitones (preserves tempo).

    Parameters
    ----------
    path : input audio file
    semitones : positive = up, negative = down. Supports fractional (e.g. 0.5).
    output_path : if None, auto-generates with suffix ``_shifted``

    Uses pyrubberband (Ableton's engine) if available for highest quality,
    falls back to pedalboard.PitchShift.
    """
    path = Path(path)
    if output_path is None:
        sign = "up" if semitones > 0 else "down"
        st_str = f"{abs(semitones):.1f}".rstrip("0").rstrip(".")
        output_path = path.with_stem(f"{path.stem}_{sign}{st_str}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try pyrubberband first (highest quality)
    try:
        return _pitch_shift_rubberband(path, semitones, output_path)
    except ImportError:
        pass

    # Fallback to pedalboard
    return _pitch_shift_pedalboard(path, semitones, output_path)


def _pitch_shift_rubberband(
    path: Path, semitones: float, output_path: Path,
) -> Path:
    """High-quality pitch shift using Rubber Band (Ableton's engine)."""
    import pyrubberband as pyrb  # type: ignore[import-untyped]

    data, sr = sf.read(str(path))
    shifted = pyrb.pitch_shift(data, sr, n_steps=semitones)
    sf.write(str(output_path), shifted, sr)
    return output_path


def _pitch_shift_pedalboard(
    path: Path, semitones: float, output_path: Path,
) -> Path:
    """Pitch shift using Spotify's pedalboard (fast, decent quality)."""
    from pedalboard import Pedalboard, PitchShift
    from pedalboard.io import AudioFile

    board = Pedalboard([PitchShift(semitones=semitones)])

    with AudioFile(str(path)) as f:
        if f.frames == 0:
            raise ValueError(f"Empty audio file: {path}")
        audio = f.read(f.frames)
        sr = f.samplerate
        channels = f.num_channels

    processed = board(audio, sr)

    suffix = output_path.suffix.lower()
    if suffix == ".mp3":
        with AudioFile(str(output_path), "w", sr, channels,
                       format="mp3", quality="320k") as out:
            out.write(processed)
    else:
        data = processed.T if processed.ndim == 2 else processed
        sf.write(str(output_path), data, sr)

    return output_path


def shift_to_key(
    path: str | Path,
    current_key: str,
    target_key: str,
    output_path: str | Path | None = None,
) -> Path:
    """Shift a track's pitch to match a target key.

    Parameters
    ----------
    path : input audio file
    current_key : current key (e.g. "8A", "A minor", "Am")
    target_key : target key (e.g. "5A", "C minor")
    """
    delta = semitones_between_keys(current_key, target_key)
    if delta == 0:
        # No shift needed — just copy
        path = Path(path)
        if output_path is None:
            return path
        import shutil
        output_path = Path(output_path)
        shutil.copy2(path, output_path)
        return output_path

    return pitch_shift(path, delta, output_path)


def semitones_between_keys(from_key: str, to_key: str) -> int:
    """Calculate the semitone distance to shift from one key to another.

    Accounts for scale mode: relative major/minor pairs (e.g., C major and
    A minor) share the same key signature and require 0 semitones shift.
    Pitch-shifting preserves intervals — it cannot change a track's mode.

    Returns the smallest shift (positive or negative, range -6 to +6).
    """
    from_note, from_scale = _parse_key(from_key)
    to_note, to_scale = _parse_key(to_key)

    from_idx = _note_index(from_note)
    to_idx = _note_index(to_note)

    if from_idx is None or to_idx is None:
        return 0

    # If scales differ, compare using relative key equivalents.
    # C major's relative minor is A minor (offset -3 semitones).
    # So to shift from C major to A minor, the actual delta is 0.
    if from_scale != to_scale:
        if from_scale == "major":
            # Convert source to relative minor: root - 3 semitones
            from_idx = (from_idx - 3) % 12
        else:
            # Convert source to relative major: root + 3 semitones
            from_idx = (from_idx + 3) % 12

    # Raw difference (0-11)
    diff = (to_idx - from_idx) % 12

    # Choose smallest shift: if diff > 6, go the other way
    if diff > 6:
        diff -= 12

    return diff


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _parse_key(key: str) -> tuple[str, str]:
    """Parse key string into (note, scale).

    Accepts: "A minor", "Am", "8A" (Camelot), "A", etc.
    """
    key = key.strip()

    # Camelot notation → musical key
    if to_camelot(key):
        camelot = to_camelot(key)
        # Reverse lookup from CAMELOT_MAP
        from .harmonic import CAMELOT_MAP
        for musical, cam in CAMELOT_MAP.items():
            if cam == camelot:
                parts = musical.split()
                return parts[0], parts[1] if len(parts) > 1 else "major"

    # "A minor", "C# major"
    parts = key.split()
    if len(parts) >= 2:
        return parts[0], parts[1]

    # "Am", "Cm", "C#m"
    if key.endswith("m") and len(key) >= 2:
        return key[:-1], "minor"

    # Just a note name: assume major
    return key, "major"


def _note_index(note: str) -> int | None:
    """Get chromatic index (0-11) for a note name."""
    # Normalize enharmonics
    note = _ENHARMONIC.get(note, note)
    try:
        return _CHROMATIC.index(note)
    except ValueError:
        return None
