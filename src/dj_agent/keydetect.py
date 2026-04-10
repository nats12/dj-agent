"""Harmonic key detection — detect musical key with Camelot notation.

Primary: Essentia KeyExtractor with EDMA profile (tuned for electronic music).
Fallback: librosa chroma + Krumhansl-Schmuckler template matching.
Includes piano chord verification audio generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .harmonic import CAMELOT_MAP, to_camelot


@dataclass
class KeyResult:
    """Result of key detection."""
    key: str           # e.g. "A minor"
    scale: str         # "major" or "minor"
    camelot: str       # e.g. "8A"
    confidence: float  # 0.0 - 1.0
    method: str        # "essentia" or "librosa"


# ---------------------------------------------------------------------------
# Note frequencies for chord generation
# ---------------------------------------------------------------------------

_NOTE_FREQS: dict[str, float] = {
    "C": 261.63, "C#": 277.18, "Db": 277.18,
    "D": 293.66, "D#": 311.13, "Eb": 311.13,
    "E": 329.63, "F": 349.23, "F#": 369.99, "Gb": 369.99,
    "G": 392.00, "G#": 415.30, "Ab": 415.30,
    "A": 440.00, "A#": 466.16, "Bb": 466.16, "B": 493.88,
}

# Semitone offsets for major and minor triads
_MAJOR_TRIAD = [0, 4, 7]   # root, major third, perfect fifth
_MINOR_TRIAD = [0, 3, 7]   # root, minor third, perfect fifth


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_key(path: str | Path) -> KeyResult:
    """Detect the musical key of a track.

    Uses Essentia EDMA profile if available, falls back to librosa.
    """
    path = Path(path)
    try:
        return _essentia_detect(path)
    except ImportError:
        pass
    return _librosa_detect(path)


def detect_tuning(path: str | Path, sr: int = 22050) -> float:
    """Detect tuning offset from A=440Hz.

    Returns deviation in fractions of a semitone.
    A track tuned to 432Hz returns approximately -0.318.
    A standard 440Hz track returns approximately 0.0.
    """
    path = Path(path)
    y, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    return float(librosa.estimate_tuning(y=y, sr=actual_sr))


# ---------------------------------------------------------------------------
# Essentia (preferred — EDMA profile for electronic music)
# ---------------------------------------------------------------------------

def _essentia_detect(path: Path) -> KeyResult:
    """Detect key using Essentia's KeyExtractor with EDMA profile."""
    import essentia.standard as es  # type: ignore[import-untyped]

    audio = es.MonoLoader(filename=str(path), sampleRate=44100)()
    key_extractor = es.KeyExtractor(profileType="edma")
    key, scale, strength = key_extractor(audio)

    key_full = f"{key} {scale}"
    camelot = to_camelot(key_full) or ""

    return KeyResult(
        key=key_full,
        scale=scale,
        camelot=camelot,
        confidence=float(strength),
        method="essentia",
    )


# ---------------------------------------------------------------------------
# librosa fallback (Krumhansl-Schmuckler)
# ---------------------------------------------------------------------------

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _librosa_detect(path: Path) -> KeyResult:
    """Detect key using librosa chroma + Krumhansl-Schmuckler correlation."""
    y, sr = librosa.load(str(path), sr=22050, mono=True)

    # Compute chroma using CQT (better frequency resolution for key detection)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # 12-bin pitch class distribution

    best_corr = float("-inf")
    best_key = "C"
    best_scale = "major"

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)

        corr_major = float(np.corrcoef(rotated, _KS_MAJOR)[0, 1])
        if np.isfinite(corr_major) and corr_major > best_corr:
            best_corr = corr_major
            best_key = _PITCH_CLASSES[i]
            best_scale = "major"

        corr_minor = float(np.corrcoef(rotated, _KS_MINOR)[0, 1])
        if np.isfinite(corr_minor) and corr_minor > best_corr:
            best_corr = corr_minor
            best_key = _PITCH_CLASSES[i]
            best_scale = "minor"

    key_full = f"{best_key} {best_scale}"
    camelot = to_camelot(key_full) or ""
    confidence = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))

    return KeyResult(
        key=key_full,
        scale=best_scale,
        camelot=camelot,
        confidence=confidence,
        method="librosa",
    )


# ---------------------------------------------------------------------------
# Piano chord verification audio
# ---------------------------------------------------------------------------

def generate_key_verification_audio(
    key: str,
    duration: float = 2.0,
    sr: int = 44100,
) -> np.ndarray:
    """Generate a piano-like chord for the given key.

    Returns a stereo numpy array (samples, 2) suitable for playback or saving.
    Uses additive synthesis with harmonics + ADSR envelope for a piano-like tone.
    """
    # Parse key
    parts = key.strip().split()
    root = parts[0] if parts else "C"
    scale = parts[1] if len(parts) > 1 else "major"

    root_freq = _NOTE_FREQS.get(root, 261.63)
    intervals = _MAJOR_TRIAD if scale == "major" else _MINOR_TRIAD

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # ADSR envelope (piano-like: fast attack, slow decay)
    attack = int(0.01 * sr)
    decay = int(0.3 * sr)
    sustain_level = 0.4
    release = int(0.5 * sr)
    n = len(t)

    envelope = np.ones(n) * sustain_level
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)
    if release < n:
        envelope[-release:] = np.linspace(sustain_level, 0, release)

    # Synthesize chord (root + third + fifth, each with harmonics)
    signal = np.zeros(n)
    for semitone_offset in intervals:
        freq = root_freq * (2 ** (semitone_offset / 12.0))
        # Fundamental + 2 harmonics with decreasing amplitude
        signal += 0.5 * np.sin(2 * np.pi * freq * t)
        signal += 0.2 * np.sin(2 * np.pi * freq * 2 * t)  # octave
        signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t)  # fifth harmonic

    signal *= envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7  # normalize

    # Stereo
    return np.column_stack([signal, signal]).astype(np.float32)
