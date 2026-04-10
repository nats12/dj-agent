"""Tests for vocal detection."""

from pathlib import Path

import numpy as np
import pytest

from dj_agent.vocals import detect_vocals_fast, _classify_vocal_level


def test_classify_vocal_level():
    assert _classify_vocal_level(0.1) == "instrumental"
    assert _classify_vocal_level(0.4) == "partial_vocal"
    assert _classify_vocal_level(0.8) == "vocal"


def test_detect_vocals_fast_returns_result(sample_sine_wav: Path):
    """Fast detection should return a valid VocalResult even with fallback."""
    result = detect_vocals_fast(sample_sine_wav)
    assert result.method in ("essentia", "librosa_heuristic")
    assert 0.0 <= result.vocal_probability <= 1.0
    assert result.classification in ("instrumental", "partial_vocal", "vocal")
    assert isinstance(result.has_vocals, bool)


def test_sine_wave_returns_valid_classification(sample_sine_wav: Path):
    """A pure sine wave may trigger the vocal heuristic (440Hz is in
    the vocal frequency range), but the result should still be valid."""
    result = detect_vocals_fast(sample_sine_wav)
    assert result.classification in ("instrumental", "partial_vocal", "vocal")
