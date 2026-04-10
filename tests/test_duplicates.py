"""Tests for duplicate detection."""

from pathlib import Path

from dj_agent.config import DuplicateConfig
from dj_agent.duplicates import find_exact_duplicates, find_fuzzy_duplicates, hash_file_chunked
from dj_agent.types import TrackInfo


def test_chunked_hash_matches_full(tmp_path: Path):
    """Chunked hashing should produce the same result regardless of chunk size."""
    content = b"x" * 200_000  # 200 KB
    f = tmp_path / "test.bin"
    f.write_bytes(content)

    h1 = hash_file_chunked(f, chunk_size=1024)
    h2 = hash_file_chunked(f, chunk_size=65536)
    h3 = hash_file_chunked(f, chunk_size=999999)
    assert h1 == h2 == h3


def test_exact_duplicates(tmp_path: Path):
    content = b"identical audio data here"
    f1 = tmp_path / "a.mp3"
    f2 = tmp_path / "b.mp3"
    f1.write_bytes(content)
    f2.write_bytes(content)

    tracks = [
        TrackInfo("1", str(f1), "Art", "Song", "Techno", 128, "5A", 300),
        TrackInfo("2", str(f2), "Art", "Song", "Techno", 128, "5A", 300),
    ]
    groups = find_exact_duplicates(tracks)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_fuzzy_duplicates():
    config = DuplicateConfig(fuzzy_threshold=85, duration_tolerance_sec=10.0)
    tracks = [
        TrackInfo("1", "/a.mp3", "Bicep", "Glue (Extended Mix)", "House", 128, "5A", 272),
        TrackInfo("2", "/b.mp3", "Bicep", "Glue (Extended Remix)", "House", 128, "5A", 275),
        TrackInfo("3", "/c.mp3", "Totally Different", "Something Else", "Techno", 140, "3B", 180),
    ]
    results = find_fuzzy_duplicates(tracks, config)
    # Bicep tracks should match (very similar), the different one should not
    assert len(results) == 1
    assert results[0][2] >= 85


def test_blocking_reduces_comparisons():
    """Tracks with different artist prefixes should not be compared."""
    config = DuplicateConfig(fuzzy_threshold=50, duration_tolerance_sec=999)
    tracks = [
        TrackInfo("1", "/a.mp3", "AAAA", "Song", "T", 128, "5A", 300),
        TrackInfo("2", "/b.mp3", "ZZZZ", "Song", "T", 128, "5A", 300),
    ]
    results = find_fuzzy_duplicates(tracks, config)
    # Different artist prefixes → different blocks → no comparison
    assert len(results) == 0
