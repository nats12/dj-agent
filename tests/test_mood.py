"""Tests for mood classification."""

from pathlib import Path

import pytest

from dj_agent.mood import classify_mood_essentia, calculate_hardness


def test_classify_mood_returns_result(sample_sine_wav: Path):
    """Mood classification should return a valid MoodResult even with fallback."""
    result = classify_mood_essentia(sample_sine_wav)
    assert result.method in ("essentia", "librosa_heuristic")
    assert result.primary_mood in ("aggressive", "happy", "party", "relaxed", "sad", "unknown")
    assert isinstance(result.mood_scores, dict)
    assert 0.0 <= result.arousal <= 1.0
    assert 0.0 <= result.valence <= 1.0


def test_hardness_high_energy_fast_aggressive():
    h = calculate_hardness(energy=0.9, bpm=145, mood="Aggressive and Hard")
    assert h >= 8


def test_hardness_low_energy_slow_chill():
    h = calculate_hardness(energy=0.2, bpm=100, mood="Chill and Relaxed")
    assert h <= 3


def test_hardness_clamps_to_range():
    assert 1 <= calculate_hardness(energy=0.0, bpm=60, mood="chill relaxed calm") <= 10
    assert 1 <= calculate_hardness(energy=1.0, bpm=180, mood="aggressive hard intense") <= 10


def test_hardness_mid_range():
    h = calculate_hardness(energy=0.5, bpm=128, mood="neutral")
    assert 3 <= h <= 7
