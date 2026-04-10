"""Shared audio loading and LUFS measurement utilities."""

from __future__ import annotations
import logging

from pathlib import Path

import librosa
import numpy as np

from .types import LoudnessResult


def load_audio(path: str | Path, sr: int | None = None,
               mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio from *path*, returning ``(samples, sample_rate)``.

    Parameters
    ----------
    sr : int | None
        Target sample rate.  ``None`` keeps the file's native rate (recommended
        for LUFS measurement).  ``22050`` is fine for feature extraction.
    mono : bool
        Down-mix to mono.
    """
    y, actual_sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, actual_sr


def load_audio_stereo(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio preserving stereo channels for LUFS measurement.

    Returns shape ``(samples, channels)`` as required by pyloudnorm.
    """
    import soundfile as sf

    data, actual_sr = sf.read(str(path))
    if sr is not None and actual_sr != sr:
        # Resample each channel
        if data.ndim == 1:
            data = librosa.resample(data, orig_sr=actual_sr, target_sr=sr)
        else:
            channels = [librosa.resample(data[:, c], orig_sr=actual_sr, target_sr=sr)
                        for c in range(data.shape[1])]
            data = np.column_stack(channels)
        actual_sr = sr
    return data, actual_sr


def measure_loudness(path: str | Path) -> LoudnessResult:
    """Measure integrated LUFS, true peak, and loudness range.

    Uses pyloudnorm (ITU-R BS.1770-4).
    """
    import pyloudnorm as pyln

    data, rate = load_audio_stereo(path)

    meter = pyln.Meter(rate)
    integrated = meter.integrated_loudness(data)

    # Sample peak in dBFS (not ITU-R BS.1770-4 true peak which requires
    # 4x oversampling — use Essentia TruePeakDetector for compliant dBTP).
    peak_linear = float(np.max(np.abs(data)))
    sample_peak_dbfs = 20.0 * np.log10(peak_linear + 1e-12)

    # Short-term loudness (3-second windows)
    window = int(3.0 * rate)
    hop = int(1.0 * rate)
    st_values: list[float] = []
    n_samples = data.shape[0]
    for start in range(0, n_samples - window, hop):
        block = data[start: start + window]
        try:
            st = meter.integrated_loudness(block)
            if np.isfinite(st):
                st_values.append(st)
        except Exception:
            pass  # individual window may fail — not critical
    short_term_max = max(st_values) if st_values else integrated

    # Loudness range (LRA) — approximate as difference between 10th and 95th
    # percentile of short-term loudness values.
    if len(st_values) >= 4:
        lra = float(np.percentile(st_values, 95) - np.percentile(st_values, 10))
    else:
        lra = 0.0

    return LoudnessResult(
        integrated_lufs=integrated,
        sample_peak_dbfs=sample_peak_dbfs,  # sample peak, not ITU true peak
        loudness_range_lu=lra,
        short_term_max_lufs=short_term_max,
    )
