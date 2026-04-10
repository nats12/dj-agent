"""Tests for phrase detection."""

from pathlib import Path

import numpy as np

from dj_agent.phrases import detect_phrases, _label_phrases
from dj_agent.types import Phrase


def test_detect_phrases_returns_list(sample_sine_wav: Path):
    """Phrase detection should return a list even for short audio."""
    result = detect_phrases(sample_sine_wav, bpm=128.0)
    assert isinstance(result, list)


def test_label_phrases_assigns_intro_and_outro():
    """First phrase should be intro, last should be outro."""
    phrases = [
        Phrase(start_ms=0, end_ms=10000, bar_count=8, label="", energy=0.3),
        Phrase(start_ms=10000, end_ms=20000, bar_count=8, label="", energy=0.6),
        Phrase(start_ms=20000, end_ms=30000, bar_count=8, label="", energy=0.9),
        Phrase(start_ms=90000, end_ms=100000, bar_count=8, label="", energy=0.2),
    ]
    _label_phrases(phrases, duration=100.0)

    assert phrases[0].label == "intro"
    assert phrases[-1].label == "outro"


def test_label_phrases_detects_drop():
    """High energy after low energy should be labeled 'drop'."""
    phrases = [
        Phrase(start_ms=0, end_ms=10000, bar_count=8, label="", energy=0.2),
        Phrase(start_ms=15000, end_ms=25000, bar_count=8, label="", energy=0.3),
        Phrase(start_ms=25000, end_ms=35000, bar_count=8, label="", energy=0.9),
    ]
    _label_phrases(phrases, duration=100.0)

    assert phrases[2].label == "drop"


def test_label_phrases_detects_breakdown():
    """Low energy mid-track should be labeled 'breakdown'."""
    phrases = [
        Phrase(start_ms=0, end_ms=10000, bar_count=8, label="", energy=0.5),
        Phrase(start_ms=30000, end_ms=40000, bar_count=8, label="", energy=0.2),
        Phrase(start_ms=50000, end_ms=60000, bar_count=8, label="", energy=0.8),
    ]
    _label_phrases(phrases, duration=100.0)

    assert phrases[1].label == "breakdown"


def test_label_phrases_no_dead_conditionals():
    """High energy sustained should be 'peak', not always 'drop'."""
    phrases = [
        Phrase(start_ms=0, end_ms=10000, bar_count=8, label="", energy=0.3),
        Phrase(start_ms=20000, end_ms=30000, bar_count=8, label="", energy=0.8),
        Phrase(start_ms=30000, end_ms=40000, bar_count=8, label="", energy=0.9),
    ]
    _label_phrases(phrases, duration=100.0)

    # Second phrase is drop (preceded by low energy)
    assert phrases[1].label == "drop"
    # Third phrase is peak (preceded by HIGH energy, not low)
    assert phrases[2].label == "peak"
