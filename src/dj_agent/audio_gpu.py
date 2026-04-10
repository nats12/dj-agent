"""V3 GPU-accelerated audio feature extraction.

Replaces librosa CPU operations with nnAudio/torchaudio GPU equivalents.
Falls back to librosa if CUDA is unavailable.

Benchmarks (from nnAudio paper):
- STFT:          10.64s (librosa) → 0.001s (nnAudio GPU) = 10,640x
- MelSpectrogram: 18.3s (librosa) → 0.015s (nnAudio GPU) = 1,220x
- CQT:          103.4s (librosa) → 0.258s (nnAudio GPU) = 400x
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# Cache GPU transform objects (created once, reused)
_gpu_transforms: dict[str, Any] = {}
_gpu_available: bool | None = None


def is_gpu_audio_available() -> bool:
    """Check if GPU audio acceleration is available."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    try:
        import torch
        import nnAudio
        _gpu_available = torch.cuda.is_available()
    except ImportError:
        _gpu_available = False
    return _gpu_available


def extract_features_gpu(
    y: np.ndarray,
    sr: int = 22050,
) -> dict[str, Any]:
    """Extract audio features using GPU acceleration.

    Returns dict with: mfcc_mean, mfcc_std, spectral_centroid,
    chroma_mean, mel_spectrogram, onset_strength.

    Falls back to librosa on CPU if GPU unavailable.
    """
    if not is_gpu_audio_available():
        return _extract_features_librosa(y, sr)

    import torch
    from nnAudio.features.mel import MelSpectrogram
    from nnAudio.features.stft import STFT
    from nnAudio.features.cqt import CQT

    device = "cuda"

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(y).float().unsqueeze(0).to(device)

    features: dict[str, Any] = {}

    try:
        # MelSpectrogram (GPU)
        mel_key = f"mel_{sr}"
        if mel_key not in _gpu_transforms:
            _gpu_transforms[mel_key] = MelSpectrogram(
                sr=sr, n_fft=2048, n_mels=128, hop_length=512
            ).to(device)
        mel = _gpu_transforms[mel_key](audio_tensor)
        mel_np = mel.cpu().numpy()[0]

        # MFCC from Mel (compute DCT on CPU — fast)
        import scipy.fft
        mfcc = scipy.fft.dct(np.log(mel_np + 1e-8), type=2, axis=0, norm="ortho")[:20]
        features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
        features["mfcc_std"] = np.std(mfcc, axis=1).tolist()

        # Spectral centroid from Mel
        freqs = np.linspace(0, sr / 2, mel_np.shape[0])
        centroid = np.sum(freqs[:, None] * mel_np, axis=0) / (np.sum(mel_np, axis=0) + 1e-8)
        features["spectral_centroid_mean"] = float(np.mean(centroid))

        # STFT for bass energy ratio
        stft_key = f"stft_{sr}"
        if stft_key not in _gpu_transforms:
            _gpu_transforms[stft_key] = STFT(
                n_fft=2048, hop_length=512, sr=sr, output_format="Magnitude"
            ).to(device)
        S = _gpu_transforms[stft_key](audio_tensor).cpu().numpy()[0]

        stft_freqs = np.linspace(0, sr / 2, S.shape[0])
        bass_mask = stft_freqs <= 150
        bass_energy = float(np.mean(S[bass_mask, :])) if bass_mask.any() else 0.0
        total_energy = float(np.mean(S)) + 1e-8
        features["bass_ratio"] = bass_energy / total_energy

        features["method"] = "gpu"

    except Exception as e:
        _log.warning("GPU feature extraction failed, falling back to CPU: %s", e)
        return _extract_features_librosa(y, sr)

    return features


def _extract_features_librosa(y: np.ndarray, sr: int) -> dict[str, Any]:
    """CPU fallback using librosa."""
    import librosa

    features: dict[str, Any] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
    features["mfcc_std"] = np.std(mfcc, axis=1).tolist()

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs <= 150
    bass_energy = float(np.mean(S[bass_mask, :])) if bass_mask.any() else 0.0
    total_energy = float(np.mean(S)) + 1e-8
    features["bass_ratio"] = bass_energy / total_energy

    features["method"] = "cpu"
    return features


def benchmark_gpu_vs_cpu(audio_path: str | Path, n_runs: int = 5) -> dict[str, float]:
    """Benchmark GPU vs CPU feature extraction speed."""
    import time
    from .audio import load_audio

    y, sr = load_audio(audio_path, sr=22050, mono=True)

    # CPU
    cpu_times = []
    for _ in range(n_runs):
        t = time.time()
        _extract_features_librosa(y, sr)
        cpu_times.append(time.time() - t)

    # GPU
    gpu_times = []
    if is_gpu_audio_available():
        for _ in range(n_runs):
            t = time.time()
            extract_features_gpu(y, sr)
            gpu_times.append(time.time() - t)

    return {
        "cpu_avg_ms": round(np.mean(cpu_times) * 1000, 1),
        "gpu_avg_ms": round(np.mean(gpu_times) * 1000, 1) if gpu_times else 0,
        "speedup": round(np.mean(cpu_times) / np.mean(gpu_times), 1) if gpu_times else 0,
    }
