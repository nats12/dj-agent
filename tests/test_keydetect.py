"""Tests for key detection."""

from pathlib import Path

import numpy as np
import soundfile as sf

from dj_agent.keydetect import (
    KeyResult,
    _librosa_detect,
    detect_key,
    generate_key_verification_audio,
)


def test_detect_key_returns_result(sample_sine_wav: Path):
    result = detect_key(sample_sine_wav)
    assert isinstance(result, KeyResult)
    assert result.method in ("essentia", "librosa")
    assert result.scale in ("major", "minor")
    assert 0.0 <= result.confidence <= 1.0
    assert result.camelot  # should not be empty


def test_a440_detected_as_a(tmp_path: Path):
    """A pure 440Hz sine should detect as A (major or minor)."""
    sr = 44100
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path = tmp_path / "a440.wav"
    sf.write(str(path), stereo, sr)

    result = detect_key(path)
    # A 440Hz sine should be detected as A (the note A is 440 Hz)
    assert "A" in result.key


def test_verification_audio_shape():
    audio = generate_key_verification_audio("A minor", duration=2.0, sr=44100)
    assert audio.shape == (44100 * 2, 2)  # 2 seconds, stereo
    assert audio.dtype == np.float32


def test_verification_audio_not_silent():
    audio = generate_key_verification_audio("C major")
    assert np.max(np.abs(audio)) > 0.1


def test_verification_audio_different_keys():
    """Different keys should produce different audio."""
    a_minor = generate_key_verification_audio("A minor")
    c_major = generate_key_verification_audio("C major")
    # The audio should be different (different frequencies)
    assert not np.allclose(a_minor, c_major, atol=0.01)
