"""Shared test fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dj_agent.config import Config, set_config
from dj_agent.types import TrackInfo


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the global config singleton between tests."""
    set_config(Config.default())
    yield
    set_config(Config.default())


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_sine_wav(tmp_path: Path) -> Path:
    """Generate a short 1-second 440 Hz sine wave as WAV."""
    import soundfile as sf

    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    path = tmp_path / "sine.wav"
    sf.write(str(path), stereo, sr)
    return path


@pytest.fixture
def sample_tracks() -> list[TrackInfo]:
    """A handful of fake TrackInfo objects for testing."""
    return [
        TrackInfo(
            db_content_id="1", path="/music/a.mp3",
            artist="Artist A", title="Track One",
            genre="Techno", bpm=130.0, key="5A", duration=300.0,
        ),
        TrackInfo(
            db_content_id="2", path="/music/b.mp3",
            artist="Artist A", title="Track One (Remix)",
            genre="Techno", bpm=132.0, key="5A", duration=305.0,
        ),
        TrackInfo(
            db_content_id="3", path="/music/c.flac",
            artist="Artist B", title="Different Song",
            genre="House", bpm=124.0, key="8B", duration=420.0,
        ),
    ]


@pytest.fixture
def v1_memory(tmp_path: Path) -> Path:
    """Create a v1-format memory JSON file."""
    data = {
        "version": 1,
        "processed_tracks": {
            "abc123": {
                "path": "file://localhost/music/test.mp3",
                "artist": "Test",
                "title": "Track",
                "energy": 7,
                "energy_source": "auto",
                "tags_source": "auto",
                "analysed_at": "2026-03-01T12:00:00",
            }
        },
        "energy_corrections": [],
        "energy_calibration": {"global_offset": 0.0, "genre_offsets": {}},
        "custom_tag_rules": [],
        "tag_corrections": [],
        "artist_corrections": [],
        "settings": {},
    }
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(data))
    return path
