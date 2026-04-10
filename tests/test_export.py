"""Tests for multi-platform DJ export."""

import xml.etree.ElementTree as ET
from pathlib import Path

from dj_agent.export import (
    _build_serato_cue_entry,
    _build_serato_markers2_payload,
    _traktor_split_path,
    write_traktor_nml,
    write_virtualdj_xml,
)


class TestTraktor:
    def test_path_splitting(self):
        dir_part, file_part = _traktor_split_path("/Users/dj/Music/track.mp3")
        assert file_part == "track.mp3"
        assert "/:Users/:" in dir_part

    def test_write_nml(self, tmp_path: Path):
        tracks = [
            {
                "path": "/music/test.flac",
                "title": "Test Track",
                "artist": "Test Artist",
                "bpm": 128.0,
                "key": "A minor",
                "duration": 300,
                "cues": [
                    {"name": "Intro", "position_ms": 0, "colour": "green"},
                    {"name": "Drop", "position_ms": 32000, "colour": "red"},
                ],
            }
        ]
        output = tmp_path / "test.nml"
        result = write_traktor_nml(tracks, output)
        assert result.exists()

        tree = ET.parse(str(result))
        root = tree.getroot()
        assert root.tag == "NML"
        assert root.get("VERSION") == "19"

        entries = root.findall(".//ENTRY")
        assert len(entries) == 1
        assert entries[0].get("TITLE") == "Test Track"

        cues = root.findall(".//CUE_V2")
        assert len(cues) == 2
        assert cues[0].get("NAME") == "Intro"
        assert cues[1].get("NAME") == "Drop"

    def test_musical_key_value(self, tmp_path: Path):
        tracks = [{"path": "/a.mp3", "title": "T", "artist": "A",
                    "bpm": 128, "key": "A minor", "duration": 300, "cues": []}]
        output = tmp_path / "key_test.nml"
        write_traktor_nml(tracks, output)

        tree = ET.parse(str(output))
        mk = tree.find(".//MUSICAL_KEY")
        assert mk is not None
        assert mk.get("VALUE") == "21"  # A minor = 21 in Traktor


class TestSerato:
    def test_cue_entry_binary(self):
        cue = {"name": "Drop", "position_ms": 32000, "colour": "red"}
        data = _build_serato_cue_entry(cue, index=1)
        assert len(data) > 10
        assert data[1] == 1  # index passed as parameter

    def test_markers2_payload(self):
        cues = [
            {"name": "Intro", "position_ms": 0, "colour": "green", "index": 0},
            {"name": "Drop", "position_ms": 32000, "colour": "red", "index": 1},
        ]
        payload = _build_serato_markers2_payload(cues)
        # Should start with version bytes
        assert payload[:2] == b"\x01\x01"
        # Rest should be base64
        assert len(payload) > 10


class TestVirtualDJ:
    def test_write_xml(self, tmp_path: Path):
        tracks = [
            {
                "path": "/music/test.mp3",
                "title": "Test",
                "artist": "Artist",
                "genre": "Techno",
                "bpm": 128.0,
                "cues": [
                    {"name": "Drop", "position_ms": 30000, "colour": "red"},
                ],
            }
        ]
        output = tmp_path / "vdj.xml"
        result = write_virtualdj_xml(tracks, output)
        assert result.exists()

        tree = ET.parse(str(result))
        root = tree.getroot()
        assert root.tag == "VirtualDJ_Database"

        songs = root.findall("Song")
        assert len(songs) == 1

        pois = songs[0].findall("Poi")
        assert len(pois) == 1
        assert pois[0].get("Name") == "Drop"
        assert float(pois[0].get("Pos")) == 30.0  # ms → seconds
