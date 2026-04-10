"""Tests for LUFS-based energy calculation."""

import numpy as np
import pytest

from dj_agent.config import EnergyConfig
from dj_agent.energy import calculate_energy


@pytest.fixture
def config():
    return EnergyConfig()


def _make_audio(amplitude: float, duration: float = 3.0, sr: int = 22050) -> np.ndarray:
    """Generate a sine wave at the given amplitude."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_louder_audio_scores_higher(config: EnergyConfig):
    sr = 22050
    quiet = _make_audio(0.01, sr=sr)
    loud = _make_audio(0.5, sr=sr)

    r_quiet = calculate_energy(quiet, sr, bpm=128.0, loudness_lufs=-30.0, config=config)
    r_loud = calculate_energy(loud, sr, bpm=128.0, loudness_lufs=-6.0, config=config)

    assert r_loud.raw_score > r_quiet.raw_score


def test_score_maps_to_1_10(config: EnergyConfig):
    sr = 22050
    audio = _make_audio(0.3, sr=sr)
    result = calculate_energy(audio, sr, bpm=128.0, loudness_lufs=-12.0, config=config)
    assert 1 <= result.calibrated_score <= 10


def test_weights_sum_to_one(config: EnergyConfig):
    total = (
        config.lufs_weight
        + config.spectral_centroid_weight
        + config.onset_density_weight
        + config.bpm_weight
        + config.dynamic_range_weight
        + config.bass_energy_weight
    )
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"
