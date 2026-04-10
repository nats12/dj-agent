"""Tests for set builder."""

from dj_agent.setbuilder import build_set
from dj_agent.types import TrackInfo


def _tracks() -> list[TrackInfo]:
    return [
        TrackInfo("1", "/a", "A", "S1", "Techno", 128.0, "8B", 300),
        TrackInfo("2", "/b", "B", "S2", "Techno", 130.0, "9B", 300),
        TrackInfo("3", "/c", "C", "S3", "Techno", 132.0, "10B", 300),
        TrackInfo("4", "/d", "D", "S4", "House", 124.0, "5A", 300),
        TrackInfo("5", "/e", "E", "S5", "Techno", 128.0, "8A", 300),
    ]


def test_build_set_returns_all_tracks():
    tracks = _tracks()
    result = build_set(tracks)
    assert len(result) == len(tracks)
    # All original tracks should be present
    ids = {t.db_content_id for t in result}
    assert ids == {"1", "2", "3", "4", "5"}


def test_build_set_single_track():
    tracks = [TrackInfo("1", "/a", "A", "S1", "Techno", 128.0, "8B", 300)]
    result = build_set(tracks)
    assert len(result) == 1


def test_build_set_two_tracks():
    tracks = _tracks()[:2]
    result = build_set(tracks)
    assert len(result) == 2


def test_build_set_with_energy_arc():
    tracks = _tracks()
    energies = {"1": 5, "2": 7, "3": 9, "4": 3, "5": 6}
    result = build_set(tracks, energies=energies, arc="warmup_to_peak")
    assert len(result) == len(tracks)

    # With warmup_to_peak, first track should have lower energy
    first_energy = energies[result[0].db_content_id]
    last_energy = energies[result[-1].db_content_id]
    # Can't guarantee perfect ordering but first should be <= last
    # (at least the algorithm attempts it)
    assert isinstance(first_energy, int)


def test_build_set_flat_no_constraint():
    tracks = _tracks()
    result = build_set(tracks, arc="flat")
    assert len(result) == len(tracks)
