"""Tests for audio quality validation."""

from pathlib import Path

import numpy as np
import pytest

from dj_agent.quality import detect_clipping, detect_silence, check_audio_quality
from dj_agent.types import SilenceRegion


def test_detect_clipping_clean_file(sample_sine_wav: Path):
    """A clean sine at 0.5 amplitude should have zero clipping."""
    count = detect_clipping(sample_sine_wav, threshold=0.99)
    assert count == 0


def test_detect_clipping_clipped_file(tmp_path: Path):
    """A file with samples at max amplitude should detect clipping."""
    import soundfile as sf

    sr = 44100
    # Generate a clipped signal
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t) * 2.0  # amplitude > 1.0
    signal = np.clip(signal, -1.0, 1.0)  # creates flat tops = clipping
    stereo = np.column_stack([signal, signal])

    path = tmp_path / "clipped.wav"
    sf.write(str(path), stereo, sr)

    count = detect_clipping(path, threshold=0.99)
    assert count > 0


def test_detect_silence(sample_sine_wav: Path):
    """A sine wave should have no silence regions."""
    regions = detect_silence(sample_sine_wav, silence_thresh_db=-60, min_silence_ms=500)
    # A continuous 440Hz sine has no silence
    assert len(regions) == 0


def test_silence_with_actual_silence(tmp_path: Path):
    """A file starting with silence should detect leading silence."""
    import soundfile as sf

    sr = 44100
    # 2 seconds silence + 1 second sine
    silence = np.zeros(sr * 2)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    signal = np.concatenate([silence, sine])
    stereo = np.column_stack([signal, signal]).astype(np.float32)

    path = tmp_path / "silence_then_sine.wav"
    sf.write(str(path), stereo, sr)

    regions = detect_silence(path, silence_thresh_db=-40, min_silence_ms=1000)
    assert len(regions) >= 1
    assert regions[0].start_ms == 0  # leading silence


def test_check_audio_quality(sample_sine_wav: Path):
    """Full quality check should return a valid report."""
    report = check_audio_quality(sample_sine_wav)
    assert report.path == str(sample_sine_wav)
    assert report.clipping_count == 0
    # Note: a pure sine wave has limited spectral content so the fake-lossless
    # detector may flag it.  We only check the report is structurally valid.
    assert isinstance(report.is_fake_lossless, bool)
    assert isinstance(report.warnings, list)
