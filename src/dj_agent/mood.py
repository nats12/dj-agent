"""Mood / vibe classification — Essentia models and CLAP zero-shot.

Adapted from AI-Music-Library-Normalization-Suite critic.py.

Two tiers:
- Essentia TF models (~3s/track): aggressive, happy, party, relaxed, sad + arousal/valence
- CLAP zero-shot (~5s/track): custom DJ-specific labels
"""

from __future__ import annotations
import logging

from pathlib import Path
from typing import Any

import numpy as np

from .types import MoodResult


# ---------------------------------------------------------------------------
# Default mood labels for CLAP zero-shot
# ---------------------------------------------------------------------------

DEFAULT_MOOD_LABELS = [
    "Dark and Hypnotic warehouse techno",
    "Happy and Euphoric festival anthem",
    "Aggressive and Hard industrial",
    "Chill and Relaxed deep house",
    "Melancholic and Emotional progressive",
]

DEFAULT_COMMERCIAL_LABELS = [
    "Mainstream Pop EDM commercial radio hit",
    "Underground dark warehouse techno minimal",
]


# ---------------------------------------------------------------------------
# Essentia mood classification (fast)
# ---------------------------------------------------------------------------

def classify_mood_essentia(path: str | Path) -> MoodResult:
    """Classify mood using Essentia pre-trained models.

    Requires ``essentia-tensorflow`` and downloaded .pb model files.
    Falls back to a librosa energy heuristic if Essentia is unavailable.
    """
    path = Path(path)

    try:
        return _essentia_mood(path)
    except (ImportError, AttributeError, Exception):
        pass  # Essentia without TF models, or model files missing

    return _librosa_mood_fallback(path)


def _essentia_mood(path: Path) -> MoodResult:
    """Run all Essentia mood classifiers."""
    import os
    import essentia.standard as es  # type: ignore[import-untyped]

    _models_dir = os.path.expanduser("~/.essentia/models")
    audio = es.MonoLoader(filename=str(path), sampleRate=16000)()

    # Shared embeddings
    embeddings = es.TensorflowPredictMusiCNN(
        graphFilename=os.path.join(_models_dir, "msd-musicnn-1.pb"),
        output="model/dense/BiasAdd",
    )(audio)

    mood_models = {
        "aggressive": "mood_aggressive-msd-musicnn-1.pb",
        "happy": "mood_happy-msd-musicnn-1.pb",
        "party": "mood_party-msd-musicnn-1.pb",
        "relaxed": "mood_relaxed-msd-musicnn-1.pb",
        "sad": "mood_sad-msd-musicnn-1.pb",
    }

    scores: dict[str, float] = {}
    for mood_name, model_file in mood_models.items():
        try:
            preds = es.TensorflowPredict2D(
                graphFilename=os.path.join(_models_dir, model_file),
                output="model/Softmax",
            )(embeddings)
            scores[mood_name] = float(np.mean(preds[:, 1]))
        except Exception:
            scores[mood_name] = 0.0

    primary = max(scores, key=scores.get) if scores else "unknown"

    # Arousal/valence (if models available)
    arousal = 0.5
    valence = 0.5
    try:
        a_preds = es.TensorflowPredict2D(
            graphFilename=os.path.join(_models_dir, "emomusic-msd-musicnn-2.pb"),
            output="model/Softmax",
        )(embeddings)
        arousal = float(np.clip(np.mean(a_preds), 0, 1))
    except Exception as _e:
        logging.getLogger(__name__).debug("Arousal model unavailable: %s", _e)
    try:
        v_preds = es.TensorflowPredict2D(
            graphFilename=os.path.join(_models_dir, "deam-msd-musicnn-2.pb"),
            output="model/Softmax",
        )(embeddings)
        valence = float(np.clip(np.mean(v_preds), 0, 1))
    except Exception as _e:
        logging.getLogger(__name__).debug("Valence model unavailable: %s", _e)

    return MoodResult(
        primary_mood=primary,
        mood_scores=scores,
        arousal=arousal,
        valence=valence,
        method="essentia",
    )


def _librosa_mood_fallback(path: Path) -> MoodResult:
    """Very rough mood estimation from audio features when Essentia is missing."""
    import librosa

    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=60)

    # Spectral centroid → brightness (aggressive/dark proxy)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa may return array in newer versions
    tempo = float(np.asarray(tempo).item()) if not isinstance(tempo, (int, float)) else float(tempo)

    # RMS energy
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # Rough heuristic classification
    if centroid > 3500 and rms > 0.08:
        primary = "aggressive"
    elif centroid > 2500 and tempo > 125:
        primary = "party"
    elif rms < 0.04:
        primary = "relaxed"
    elif centroid < 2000:
        primary = "sad"
    else:
        primary = "happy"

    return MoodResult(
        primary_mood=primary,
        mood_scores={primary: 0.6},
        arousal=float(np.clip(rms * 10, 0, 1)),
        valence=float(np.clip(centroid / 5000, 0, 1)),
        method="librosa_heuristic",
    )


# ---------------------------------------------------------------------------
# CLAP zero-shot classification (flexible)
# ---------------------------------------------------------------------------

def classify_mood_clap(
    path: str | Path,
    labels: list[str] | None = None,
) -> MoodResult:
    """Zero-shot mood classification using CLAP.

    Requires ``laion-clap`` (``pip install laion-clap``).
    Labels are configurable — defaults to DJ-specific mood categories.
    """
    path = Path(path)
    if labels is None:
        labels = DEFAULT_MOOD_LABELS

    model = _get_clap_model()

    audio_embed = model.get_audio_embedding_from_filelist(
        x=[str(path)], use_tensor=False,
    )
    text_embed = model.get_text_embedding(labels)

    similarity = audio_embed @ text_embed.T
    best_idx = int(np.argmax(similarity))

    scores = {label: float(similarity[0][i]) for i, label in enumerate(labels)}

    return MoodResult(
        primary_mood=labels[best_idx],
        mood_scores=scores,
        method="clap",
    )


# ---------------------------------------------------------------------------
# CLAP model cache (heavy to load — reuse across calls)
# ---------------------------------------------------------------------------

_clap_model = None
_clap_lock = __import__("threading").Lock()


def _get_clap_model():
    """Get or create the cached CLAP model (thread-safe)."""
    global _clap_model
    if _clap_model is not None:
        return _clap_model
    with _clap_lock:
        if _clap_model is not None:
            return _clap_model
        import laion_clap  # type: ignore[import-untyped]
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        _clap_model.load_ckpt(ckpt="music_audioset_epoch_15_esc_90.14.pt")
        return _clap_model


# ---------------------------------------------------------------------------
# Commercial vs Underground
# ---------------------------------------------------------------------------

def classify_commercial(
    path: str | Path,
    labels: list[str] | None = None,
) -> float:
    """Return 0.0 (underground) to 1.0 (commercial) using CLAP.

    Adapted from normalization suite critic.py.
    """
    path = Path(path)
    if labels is None:
        labels = DEFAULT_COMMERCIAL_LABELS

    model = _get_clap_model()

    audio_embed = model.get_audio_embedding_from_filelist(
        x=[str(path)], use_tensor=False,
    )
    text_embed = model.get_text_embedding(labels)

    similarity = audio_embed @ text_embed.T

    # First label is commercial, second is underground
    commercial_score = float(similarity[0][0])
    underground_score = float(similarity[0][1])
    total = commercial_score + underground_score + 1e-8
    return commercial_score / total


# ---------------------------------------------------------------------------
# Hardness scoring
# ---------------------------------------------------------------------------

def calculate_hardness(
    energy: float,
    bpm: float,
    mood: str,
) -> int:
    """Calculate hardness score 1-10 from energy, BPM, and mood.

    Adapted from normalization suite critic.py _calculate_hardness.
    """
    hardness = energy * 8.0

    if bpm > 140:
        hardness += 1.5
    elif bpm > 130:
        hardness += 1.0
    elif bpm < 110:
        hardness -= 1.0

    mood_lower = mood.lower()
    if any(w in mood_lower for w in ("aggressive", "hard", "intense")):
        hardness += 2.0
    elif any(w in mood_lower for w in ("dark", "hypnotic")):
        hardness += 1.0
    elif any(w in mood_lower for w in ("chill", "relaxed", "calm")):
        hardness -= 2.0

    return max(1, min(10, int(round(hardness))))
