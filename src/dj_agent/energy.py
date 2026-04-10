"""LUFS-based energy calculation.

Replaces the original RMS-only approach with pyloudnorm LUFS measurement
as the primary loudness factor, keeping spectral centroid, onset density,
BPM, dynamic range, and bass energy as supporting factors.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .audio import load_audio, measure_loudness
from .calibration import apply_calibration
from .config import Config, EnergyConfig
from .types import EnergyResult, LoudnessResult


def analyse_track(
    file_path: str | Path,
    rekordbox_bpm: float,
    genre: str | None = None,
    config: Config | None = None,
    calibration: dict | None = None,
) -> EnergyResult:
    """Analyse a single track and return an :class:`EnergyResult`.

    BPM and key are always read from Rekordbox — never recalculated.
    """
    from .config import get_config
    full_config = config or get_config()
    energy_config = full_config.energy

    file_path = Path(file_path)

    # LUFS (needs native SR, stereo) — single load for loudness
    loudness = measure_loudness(file_path)
    # Guard against -inf on blank/silent audio
    if not np.isfinite(loudness.integrated_lufs):
        loudness = LoudnessResult(
            integrated_lufs=-60.0, sample_peak_dbfs=-60.0,
            loudness_range_lu=0.0, short_term_max_lufs=-60.0,
        )

    # Load mono at 22050 for feature extraction (second load — necessary
    # because LUFS needs native SR stereo, features need 22050 mono)
    y, sr = load_audio(file_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    raw = calculate_energy(
        y, sr,
        bpm=rekordbox_bpm,
        genre=genre,
        loudness_lufs=loudness.integrated_lufs,
        config=energy_config,
    )

    calibrated = raw.calibrated_score  # already 1-10 from calculate_energy
    if calibration:
        calibrated = apply_calibration(raw.calibrated_score, genre or "", calibration)

    # AI Reasoning (Vibe/Texture) — only in "full" processing tier
    vibe_description = None
    texture_tags = {}

    if full_config.processing_tier == "full":
        from .reasoning import analyze_vibe, classify_nuance
        try:
            vibe_description = analyze_vibe(file_path)
            texture_tags = classify_nuance(file_path)
        except Exception:
            pass  # fail gracefully if LLM is unavailable

    return EnergyResult(
        integrated_lufs=loudness.integrated_lufs,
        short_term_max_lufs=loudness.short_term_max_lufs,
        spectral_centroid_mean=raw.spectral_centroid_mean,
        onset_density=raw.onset_density,
        bass_ratio=raw.bass_ratio,
        dynamic_range=raw.dynamic_range,
        raw_score=raw.raw_score,
        calibrated_score=calibrated,
        vibe_description=vibe_description,
        texture_tags=texture_tags,
    )


def calculate_energy(
    y: np.ndarray,
    sr: int,
    bpm: float,
    genre: str | None = None,
    loudness_lufs: float = -20.0,
    config: EnergyConfig | None = None,
) -> EnergyResult:
    """Compute composite energy score from audio features.

    Returns an :class:`EnergyResult` where ``raw_score`` is 0–1 and
    ``calibrated_score`` is the un-calibrated 1–10 mapping.
    """
    if config is None:
        from .config import get_config
        config = get_config().energy

    # 1. LUFS loudness  — maps -30 LUFS → 0,  -5 LUFS → 1
    lufs_score = float(np.clip((loudness_lufs + 30) / 25, 0, 1))

    # 2. Spectral centroid (brightness / aggression)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_score = float(np.clip(centroid_mean / 4500, 0, 1))

    # 3. Onset density (how busy / driving)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = max(librosa.get_duration(y=y, sr=sr), 1.0)
    onset_density = len(onsets) / duration
    onset_score = float(np.clip(onset_density / 7.0, 0, 1))

    # 4. BPM normalised to genre range
    if genre and genre.lower() in config.genre_bpm_ranges:
        lo, hi = config.genre_bpm_ranges[genre.lower()]
    else:
        lo, hi = config.default_bpm_range
    bpm_score = float(np.clip((bpm - lo) / max(hi - lo, 1), 0, 1))

    # 5. Dynamic range (inverted — less dynamic = more energy)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    dynamic_range = float(np.std(rms_db))
    dyn_score = 1.0 - float(np.clip(dynamic_range / 20.0, 0, 1))

    # 6. Bass energy ratio (below 150 Hz vs full spectrum)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs <= 150
    bass_energy = float(np.mean(S[bass_mask, :])) if bass_mask.any() else 0.0
    total_energy = float(np.mean(S)) + 1e-8
    bass_ratio = bass_energy / total_energy
    bass_score = float(np.clip(bass_ratio / 0.4, 0, 1))

    # Weighted sum
    raw = (
        config.lufs_weight * lufs_score
        + config.spectral_centroid_weight * centroid_score
        + config.onset_density_weight * onset_score
        + config.bpm_weight * bpm_score
        + config.dynamic_range_weight * dyn_score
        + config.bass_energy_weight * bass_score
    )

    score_1_10 = int(np.clip(np.round(raw * 9 + 1), 1, 10))

    return EnergyResult(
        integrated_lufs=0.0,  # filled by caller
        short_term_max_lufs=0.0,
        spectral_centroid_mean=centroid_mean,
        onset_density=onset_density,
        bass_ratio=bass_ratio,
        dynamic_range=dynamic_range,
        raw_score=float(raw),
        calibrated_score=score_1_10,
    )


# ---------------------------------------------------------------------------
# Colour mapping (unchanged from original)
# ---------------------------------------------------------------------------

ENERGY_COLOURS = {
    range(1, 3): ("blue", 1),    # Ambient / warm-up
    range(3, 5): ("green", 3),   # Low energy / chill
    range(5, 7): ("yellow", 5),  # Mid energy
    range(7, 9): ("orange", 7),  # Peak time
    range(9, 11): ("red", 9),    # Maximum intensity
}


def energy_to_colour(energy: int) -> str:
    """Map energy 1-10 to a colour name."""
    for rng, (colour, _) in ENERGY_COLOURS.items():
        if energy in rng:
            return colour
    return "red"


def energy_to_colour_id(energy: int) -> int:
    """Map energy 1-10 to a Rekordbox colour ID."""
    if energy <= 2:
        return 1   # Blue
    if energy <= 4:
        return 3   # Green
    if energy <= 6:
        return 5   # Yellow
    if energy <= 8:
        return 7   # Orange
    return 9        # Red


def energy_to_rating(energy: int) -> int:
    """Map energy 1-10 to Rekordbox rating 0-255."""
    if not energy:
        return 0
    return int((energy / 10) * 255)
