"""Tests for pitch shifting."""

from dj_agent.pitchshift import semitones_between_keys, _parse_key


class TestSemitones:
    def test_same_key(self):
        assert semitones_between_keys("A minor", "A minor") == 0

    def test_up_one_semitone(self):
        assert semitones_between_keys("C major", "C# major") == 1

    def test_down_one_semitone(self):
        assert semitones_between_keys("C# major", "C major") == -1

    def test_up_five_semitones(self):
        assert semitones_between_keys("C major", "F major") == 5

    def test_wraps_shortest_path(self):
        # C to A = either +9 or -3; should return -3 (shortest)
        result = semitones_between_keys("C major", "A major")
        assert result == -3

    def test_camelot_notation(self):
        # 8B = C major, 9B = G major = +7 semitones, but shortest = -5
        result = semitones_between_keys("8B", "9B")
        assert abs(result) <= 6  # always chooses shortest path

    def test_handles_minor(self):
        result = semitones_between_keys("A minor", "C minor")
        assert result == 3  # A→C = 3 semitones up


class TestParseKey:
    def test_full_name(self):
        note, scale = _parse_key("A minor")
        assert note == "A"
        assert scale == "minor"

    def test_shorthand(self):
        note, scale = _parse_key("Am")
        assert note == "A"
        assert scale == "minor"

    def test_just_note(self):
        note, scale = _parse_key("C")
        assert note == "C"
        assert scale == "major"  # default

    def test_camelot(self):
        note, scale = _parse_key("8A")
        assert scale == "minor"  # A = minor in Camelot
