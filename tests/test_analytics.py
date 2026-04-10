"""Tests for library analytics."""

from dj_agent.analytics import analyse_library, format_analytics
from dj_agent.types import TrackInfo


def _make_tracks() -> list[TrackInfo]:
    return [
        TrackInfo("1", "/a", "Art A", "Song 1", "Techno", 130.0, "8B", 300),
        TrackInfo("2", "/b", "Art B", "Song 2", "Techno", 132.0, "9B", 280),
        TrackInfo("3", "/c", "Art C", "Song 3", "House", 124.0, "5A", 350),
        TrackInfo("4", "/d", "", "Song 4", "", 0.0, "", 400),  # missing metadata
        TrackInfo("5", "/e", "Art E", "Song 5", "Techno", 128.0, "8B", 320),
    ]


def test_analyse_library():
    tracks = _make_tracks()
    report = analyse_library(tracks)

    assert report["total"] == 5
    assert report["missing_genre"] == 1
    assert report["missing_artist"] == 1
    assert report["missing_key"] == 1
    assert report["missing_bpm"] == 1
    assert 0 < report["metadata_completeness"] < 100


def test_genre_distribution():
    tracks = _make_tracks()
    report = analyse_library(tracks)
    genres = dict(report["genre_distribution"])
    assert genres.get("Techno", 0) == 3
    assert genres.get("House", 0) == 1


def test_bpm_stats():
    tracks = _make_tracks()
    report = analyse_library(tracks)
    assert report["bpm_min"] > 0
    assert report["bpm_max"] >= report["bpm_min"]
    assert report["bpm_avg"] > 0


def test_key_coverage():
    tracks = _make_tracks()
    report = analyse_library(tracks)
    # We have 3 distinct keys: 8B, 9B, 5A → 3/24
    assert report["key_coverage"] == "3/24"


def test_format_analytics():
    tracks = _make_tracks()
    report = analyse_library(tracks)
    output = format_analytics(report)
    assert "LIBRARY ANALYTICS" in output
    assert "Techno" in output
    assert "completeness" in output.lower()


def test_empty_library():
    report = analyse_library([])
    assert report["total"] == 0
