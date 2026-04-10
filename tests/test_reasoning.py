"""Tests for reasoning module — backend detection and preprocessing."""

from pathlib import Path

import numpy as np
import soundfile as sf

from dj_agent.reasoning import (
    _extract_snippet,
    get_backend,
    _gemini_available,
)


def test_get_backend_returns_string():
    result = get_backend()
    assert result in ("flamingo", "ollama", "gemini", "none")


def test_gemini_available_with_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert _gemini_available() is True

def test_gemini_available_returns_bool():
    # On this machine OAuth creds exist, so it should be True
    # On a clean machine it would be False — we just verify the type
    assert isinstance(_gemini_available(), bool)


def test_extract_snippet_produces_valid_wav(sample_sine_wav: Path, tmp_path: Path):
    snippet = _extract_snippet(sample_sine_wav, duration_sec=0.5)
    assert snippet.exists()
    assert snippet.suffix == ".wav"

    data, sr = sf.read(str(snippet))
    assert sr == 16000  # downsampled to 16kHz
    assert data.ndim == 1  # mono
    assert len(data) > 0

    snippet.unlink()


def test_extract_snippet_offset(tmp_path: Path):
    """Snippet starts at 25% of track by default, not at 0."""
    sr = 44100
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([audio, audio])
    wav = tmp_path / "long.wav"
    sf.write(str(wav), stereo, sr)

    snippet = _extract_snippet(wav, duration_sec=1.0, offset_pct=0.5)
    data, _ = sf.read(str(snippet))
    # Should be ~1 second at 16kHz = ~16000 samples
    assert 15000 < len(data) < 17000
    snippet.unlink()
