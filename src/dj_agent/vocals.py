"""Vocal / voice detection — classify tracks as vocal, instrumental, or partial.

Two tiers:
- Fast: Essentia voice_instrumental TF model (~2-5s/track)
- Thorough: Demucs stem separation + vocal RMS ratio (~30s/track GPU)

Note: inaSpeechSegmenter classifies singing as "music" not "speech" — it is
NOT suitable for vocal vs instrumental classification.  Use Essentia or Demucs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import VocalResult


# ---------------------------------------------------------------------------
# Fast pass — Essentia voice/instrumental classifier
# ---------------------------------------------------------------------------

def detect_vocals_fast(path: str | Path) -> VocalResult:
    """Classify a track as vocal or instrumental using Essentia TF model.

    Requires ``essentia-tensorflow`` and the voice_instrumental model files.
    Falls back to a librosa-based heuristic if Essentia is unavailable.
    """
    path = Path(path)

    try:
        return _essentia_voice_instrumental(path)
    except (ImportError, AttributeError, Exception):
        pass  # Essentia without TF models, or model files missing

    # Fallback: harmonic-to-percussive ratio heuristic
    return _librosa_heuristic(path)


def _essentia_voice_instrumental(path: Path) -> VocalResult:
    """Use Essentia's pre-trained voice/instrumental classifier."""
    import essentia.standard as es  # type: ignore[import-untyped]

    audio = es.MonoLoader(filename=str(path), sampleRate=16000)()

    import os
    _models_dir = os.path.expanduser("~/.essentia/models")

    # Get embeddings (shared with mood classifiers)
    embeddings = es.TensorflowPredictMusiCNN(
        graphFilename=os.path.join(_models_dir, "msd-musicnn-1.pb"),
        output="model/dense/BiasAdd",
    )(audio)

    predictions = es.TensorflowPredict2D(
        graphFilename=os.path.join(_models_dir, "voice_instrumental-msd-musicnn-1.pb"),
        output="model/Softmax",
    )(embeddings)

    # predictions shape: (n_patches, 2) → [instrumental, vocal]
    mean_preds = np.mean(predictions, axis=0)
    vocal_prob = float(mean_preds[1]) if len(mean_preds) > 1 else 0.5

    has_vocals = vocal_prob > 0.5
    classification = _classify_vocal_level(vocal_prob)

    return VocalResult(
        has_vocals=has_vocals,
        vocal_probability=vocal_prob,
        method="essentia",
        classification=classification,
    )


def _librosa_heuristic(path: Path) -> VocalResult:
    """Rough vocal detection using harmonic-percussive separation.

    Tracks with strong harmonic content in the vocal frequency range
    (300-4000 Hz) relative to the full spectrum are more likely to have vocals.
    This is a crude fallback — prefer Essentia or Demucs.
    """
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=60)

    # Harmonic-percussive separation
    y_harmonic, _ = librosa.effects.hpss(y)

    # Spectral centroid of harmonic component
    S_harm = np.abs(librosa.stft(y_harmonic))
    freqs = librosa.fft_frequencies(sr=sr)

    # Vocal frequency range: 300-4000 Hz
    vocal_mask = (freqs >= 300) & (freqs <= 4000)
    vocal_energy = float(np.mean(S_harm[vocal_mask, :])) if vocal_mask.any() else 0.0
    total_energy = float(np.mean(S_harm)) + 1e-8

    vocal_ratio = vocal_energy / total_energy
    # Heuristic: ratio > 1.5 suggests strong vocal presence
    vocal_prob = float(np.clip((vocal_ratio - 1.0) / 1.0, 0, 1))

    return VocalResult(
        has_vocals=vocal_prob > 0.5,
        vocal_probability=vocal_prob,
        method="librosa_heuristic",
        classification=_classify_vocal_level(vocal_prob),
    )


# ---------------------------------------------------------------------------
# Thorough pass — Demucs stem separation
# ---------------------------------------------------------------------------

def detect_vocals_thorough(path: str | Path) -> VocalResult:
    """Separate vocals with Demucs and measure vocal energy ratio.

    Requires ``demucs`` (``pip install demucs``).
    """
    import demucs.api  # type: ignore[import-untyped]

    path = Path(path)

    separator = demucs.api.Separator(model="htdemucs")
    origin, separated = separator.separate_audio_file(str(path))

    vocals = separated["vocals"].numpy()
    full_mix = origin.numpy()

    vocal_rms = float(np.sqrt(np.mean(vocals ** 2)))
    mix_rms = float(np.sqrt(np.mean(full_mix ** 2))) + 1e-8
    vocal_ratio = vocal_rms / mix_rms

    # Thresholds from research:
    # < 0.05 = instrumental
    # 0.05-0.15 = partial vocal
    # > 0.15 = vocal
    has_vocals = vocal_ratio > 0.10
    vocal_prob = float(np.clip(vocal_ratio / 0.25, 0, 1))

    return VocalResult(
        has_vocals=has_vocals,
        vocal_probability=vocal_prob,
        method="demucs",
        classification=_classify_vocal_level(vocal_prob),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_vocal_level(prob: float) -> str:
    if prob < 0.25:
        return "instrumental"
    elif prob < 0.6:
        return "partial_vocal"
    else:
        return "vocal"
