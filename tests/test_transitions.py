"""Tests for transition compatibility scoring."""

from dj_agent.transitions import rate_transition, score_transition
from dj_agent.types import TrackInfo


def _t(bpm: float, key: str, genre: str = "Techno") -> TrackInfo:
    return TrackInfo("1", "/a", "Art", "Song", genre, bpm, key, 300)


def test_perfect_transition():
    result = score_transition(
        _t(128, "8B"), _t(128, "8B"),
        energy_a=7, energy_b=7,
    )
    assert result["total"] > 0.85
    assert result["harmonic"] == 1.0
    assert result["bpm"] == 1.0
    assert result["energy"] == 1.0


def test_bad_key_transition():
    result = score_transition(
        _t(128, "8B"), _t(128, "2A"),
        energy_a=7, energy_b=7,
    )
    assert result["harmonic"] == 0.0
    # Harmonic is 0 but same BPM/genre/energy still contribute
    assert result["total"] < result["bpm"]  # total dragged down by harmonic


def test_bad_bpm_transition():
    result = score_transition(
        _t(128, "8B"), _t(90, "8B"),
        energy_a=7, energy_b=7,
    )
    assert result["bpm"] == 0.0


def test_vocal_overlap_penalty():
    result = score_transition(
        _t(128, "8B"), _t(128, "8B"),
        vocal_a=True, vocal_b=True,
    )
    assert result["vocal"] == 0.1  # penalty for two vocals


def test_one_vocal_ok():
    result = score_transition(
        _t(128, "8B"), _t(128, "8B"),
        vocal_a=True, vocal_b=False,
    )
    assert result["vocal"] == 0.8


def test_energy_jump_penalty():
    result = score_transition(
        _t(128, "8B"), _t(128, "8B"),
        energy_a=3, energy_b=10,
    )
    assert result["energy"] < 0.2


def test_rate_transition():
    assert rate_transition(0.8) == "green"
    assert rate_transition(0.5) == "yellow"
    assert rate_transition(0.2) == "red"
