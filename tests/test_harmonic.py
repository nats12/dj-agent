"""Tests for harmonic mixing — Camelot wheel, compatibility, scoring."""

from dj_agent.harmonic import (
    camelot_distance,
    get_compatible_keys,
    score_transition,
    suggest_harmonic_transitions,
    to_camelot,
)
from dj_agent.types import TrackInfo


class TestCamelotWheel:
    def test_to_camelot_already_camelot(self):
        assert to_camelot("8B") == "8B"
        assert to_camelot("5a") == "5A"

    def test_to_camelot_musical_key(self):
        assert to_camelot("C major") == "8B"
        assert to_camelot("A minor") == "8A"

    def test_to_camelot_unknown(self):
        assert to_camelot("nonsense") is None

    def test_distance_same_key(self):
        assert camelot_distance("8B", "8B") == 0

    def test_distance_adjacent(self):
        assert camelot_distance("8B", "9B") == 1
        assert camelot_distance("8B", "7B") == 1

    def test_distance_wraps_around(self):
        assert camelot_distance("1B", "12B") == 1
        assert camelot_distance("12A", "1A") == 1

    def test_distance_opposite(self):
        assert camelot_distance("1B", "7B") == 6

    def test_compatible_keys(self):
        keys = get_compatible_keys("8B")
        labels = {k: rel for k, rel in keys}
        assert labels["8B"] == "same"
        assert labels["9B"] == "adjacent_up"
        assert labels["7B"] == "adjacent_down"
        assert labels["8A"] == "relative"
        assert labels["10B"] == "energy_boost"


class TestTransitionScoring:
    def test_perfect_transition(self):
        a = TrackInfo("1", "/a", "Art", "Song", "Techno", 128.0, "8B", 300)
        b = TrackInfo("2", "/b", "Art", "Song", "Techno", 128.0, "8B", 300)
        score = score_transition(a, b)
        assert score > 0.8

    def test_terrible_transition(self):
        a = TrackInfo("1", "/a", "Art", "Song", "Techno", 128.0, "8B", 300)
        b = TrackInfo("2", "/b", "Art", "Song", "House", 90.0, "2A", 300)
        score = score_transition(a, b)
        assert score < 0.4

    def test_suggest_harmonic(self):
        target = TrackInfo("1", "/a", "Art", "Song", "Techno", 128.0, "8B", 300)
        library = [
            TrackInfo("2", "/b", "Art", "B", "Techno", 129.0, "8B", 300),
            TrackInfo("3", "/c", "Art", "C", "Techno", 128.0, "9B", 300),
            TrackInfo("4", "/d", "Art", "D", "Techno", 128.0, "2A", 300),  # incompatible key
        ]
        results = suggest_harmonic_transitions(target, library, top_k=5)
        ids = [s.track.db_content_id for s in results]
        assert "2" in ids  # same key
        assert "3" in ids  # adjacent
        assert "4" not in ids  # distant key
