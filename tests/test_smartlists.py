"""Tests for smart playlist rules engine."""

from dj_agent.smartlists import filter_tracks, matches_rule, parse_rule
from dj_agent.types import TrackInfo


def _lib() -> list[TrackInfo]:
    return [
        TrackInfo("1", "/a", "Bicep", "Glue", "Techno", 130.0, "8B", 300),
        TrackInfo("2", "/b", "Peggy Gou", "Starry Night", "House", 124.0, "5A", 350),
        TrackInfo("3", "/c", "Charlotte de Witte", "Overdrive", "Techno", 140.0, "3B", 280),
        TrackInfo("4", "/d", "Disclosure", "Latch", "House", 122.0, "9B", 260),
    ]


def _tags() -> dict[str, dict]:
    return {
        "1": {"energy": 7, "mood": "dark", "vocal": False},
        "2": {"energy": 5, "mood": "chill", "vocal": True},
        "3": {"energy": 9, "mood": "aggressive", "vocal": False},
        "4": {"energy": 6, "mood": "happy", "vocal": True},
    }


class TestParseRule:
    def test_simple_field(self):
        rule = parse_rule("genre:Techno")
        assert rule["type"] == "field"
        assert rule["field"] == "genre"
        assert rule["value"] == "Techno"

    def test_range(self):
        rule = parse_rule("bpm:125-135")
        assert rule["op"] == "range"
        assert rule["lo"] == 125
        assert rule["hi"] == 135

    def test_gte(self):
        rule = parse_rule("energy:8+")
        assert rule["op"] == "gte"
        assert rule["value"] == 8

    def test_and(self):
        rule = parse_rule("genre:Techno AND energy:8+")
        assert rule["type"] == "and"
        assert len(rule["children"]) == 2

    def test_or(self):
        rule = parse_rule("genre:Techno OR genre:House")
        assert rule["type"] == "or"

    def test_not(self):
        rule = parse_rule("NOT vocal")
        assert rule["type"] == "not"

    def test_parentheses(self):
        rule = parse_rule("(genre:House OR genre:Techno) AND energy:7+")
        assert rule["type"] == "and"
        assert len(rule["children"]) == 2
        assert rule["children"][0]["type"] == "or"
        assert rule["children"][1]["op"] == "gte"

    def test_nested_parentheses(self):
        rule = parse_rule("(genre:House)")
        assert rule["type"] == "field"
        assert rule["field"] == "genre"
        assert rule["value"] == "House"


class TestFilterTracks:
    def test_genre_filter(self):
        result = filter_tracks(_lib(), "genre:Techno")
        assert len(result) == 2
        assert all(t.genre == "Techno" for t in result)

    def test_bpm_range(self):
        result = filter_tracks(_lib(), "bpm:120-130")
        bpms = [t.bpm for t in result]
        assert all(120 <= b <= 130 for b in bpms)

    def test_energy_threshold(self):
        result = filter_tracks(_lib(), "energy:7+", _tags())
        ids = [t.db_content_id for t in result]
        assert "1" in ids  # energy 7
        assert "3" in ids  # energy 9
        assert "2" not in ids  # energy 5

    def test_compound_and(self):
        result = filter_tracks(_lib(), "genre:Techno AND energy:8+", _tags())
        assert len(result) == 1
        assert result[0].db_content_id == "3"  # Techno + energy 9

    def test_or_rule(self):
        result = filter_tracks(_lib(), "genre:Techno OR genre:House")
        assert len(result) == 4  # all tracks

    def test_not_rule(self):
        result = filter_tracks(_lib(), "genre:Techno AND NOT energy:8+", _tags())
        assert len(result) == 1
        assert result[0].db_content_id == "1"  # Techno, energy 7 (not 8+)

    def test_artist_filter(self):
        result = filter_tracks(_lib(), "artist:Bicep")
        assert len(result) == 1
        assert result[0].artist == "Bicep"
