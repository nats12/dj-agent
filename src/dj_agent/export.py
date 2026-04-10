"""Multi-platform DJ export — Traktor NML, Serato GEOB, Engine DJ, VirtualDJ.

Extends the existing Rekordbox XML export in sync.py to support all major
DJ platforms.
"""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .sync import COLOUR_MAP


# ---------------------------------------------------------------------------
# Colour mapping per platform
# ---------------------------------------------------------------------------

_RGB = {
    "green": (0, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 100, 255),
    "yellow": (255, 255, 0),
}

# Traktor musical key values (chromatic: major 0-11, minor 12-23)
_TRAKTOR_KEY_MAP: dict[str, int] = {
    "C major": 0, "Db major": 1, "D major": 2, "Eb major": 3,
    "E major": 4, "F major": 5, "Gb major": 6, "G major": 7,
    "Ab major": 8, "A major": 9, "Bb major": 10, "B major": 11,
    "C minor": 12, "Db minor": 13, "D minor": 14, "Eb minor": 15,
    "E minor": 16, "F minor": 17, "Gb minor": 18, "G minor": 19,
    "Ab minor": 20, "A minor": 21, "Bb minor": 22, "B minor": 23,
    # Sharp equivalents
    "C# major": 1, "D# major": 3, "F# major": 6,
    "G# major": 8, "A# major": 10,
    "C# minor": 13, "D# minor": 15, "F# minor": 18,
    "G# minor": 20, "A# minor": 22,
}


# ---------------------------------------------------------------------------
# Traktor NML
# ---------------------------------------------------------------------------

def write_traktor_nml(
    tracks: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Generate a Traktor NML collection file.

    Each track dict needs: path, title, artist, bpm, key, duration, cues.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("NML", VERSION="19")
    ET.SubElement(root, "HEAD", COMPANY="www.native-instruments.com",
                  PROGRAM="Traktor Pro 3")
    ET.SubElement(root, "MUSICFOLDERS")
    collection = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(tracks)))

    for t in tracks:
        entry = ET.SubElement(collection, "ENTRY",
                              TITLE=t.get("title", ""),
                              ARTIST=t.get("artist", ""))

        # LOCATION — Traktor colon-delimited path
        raw_path = _strip_file_uri(t.get("path", ""))
        dir_part, file_part = _traktor_split_path(raw_path)
        ET.SubElement(entry, "LOCATION", DIR=dir_part, FILE=file_part,
                      VOLUME="", VOLUMEID="")

        # INFO
        key_str = t.get("key", "")
        ET.SubElement(entry, "INFO",
                      KEY=key_str,
                      PLAYTIME=str(int(t.get("duration", 0))),
                      COMMENT=t.get("comment", ""))

        # TEMPO
        bpm = t.get("bpm", 0)
        if bpm:
            ET.SubElement(entry, "TEMPO", BPM=f"{bpm:.2f}", BPM_QUALITY="100.0")

        # MUSICAL_KEY
        traktor_key_val = _TRAKTOR_KEY_MAP.get(key_str)
        if traktor_key_val is not None:
            ET.SubElement(entry, "MUSICAL_KEY", VALUE=str(traktor_key_val))

        # CUE_V2 elements
        for i, cue in enumerate(t.get("cues", [])[:8]):
            rgb = _RGB.get(cue.get("colour", "green"), (0, 255, 0))
            ET.SubElement(entry, "CUE_V2",
                          NAME=cue.get("name", ""),
                          DISPL_ORDER=str(i),
                          TYPE="0",
                          START=f"{cue.get('position_ms', 0):.1f}",
                          LEN="0.0",
                          REPEATS="-1",
                          HOTCUE=str(i),
                          COLOR_RED=str(rgb[0]),
                          COLOR_GREEN=str(rgb[1]),
                          COLOR_BLUE=str(rgb[2]))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path


def _traktor_split_path(path: str) -> tuple[str, str]:
    """Split path into Traktor's DIR + FILE format (cross-platform)."""
    # Normalize backslashes to forward slashes for consistent parsing
    normalized = path.replace("\\", "/")
    parts = normalized.rsplit("/", 1)
    if len(parts) == 2:
        dir_path = parts[0].replace("/", "/:") + "/:"
        return dir_path, parts[1]
    return "/:", path


# ---------------------------------------------------------------------------
# Serato Markers2 (ID3 GEOB frames)
# ---------------------------------------------------------------------------

def write_serato_markers(tracks: list[dict[str, Any]]) -> int:
    """Write Serato Markers2 cue points directly into audio files.

    Supports MP3 and AIFF (ID3 GEOB frames).
    FLAC is NOT supported — Serato uses a different Vorbis comment format
    for FLAC that is not implemented here.
    Returns the number of tracks successfully written.
    """
    from mutagen.id3 import ID3, GEOB

    count = 0
    errors: list[str] = []
    for t in tracks:
        cues = t.get("cues", [])
        if not cues:
            continue

        raw_path = _strip_file_uri(t.get("path", ""))
        if not Path(raw_path).exists():
            errors.append(f"File not found: {raw_path}")
            continue

        # Only MP3 and AIFF support ID3 GEOB frames; skip FLAC/WAV/OGG
        suffix = Path(raw_path).suffix.lower()
        if suffix not in (".mp3", ".aiff", ".aif"):
            errors.append(f"Serato markers not supported for {suffix} files: {raw_path}")
            continue

        # Validate cue data types before building binary payload
        for i, cue in enumerate(cues):
            if not isinstance(cue.get("position_ms", 0), (int, float)):
                errors.append(f"Invalid position_ms type in cue {i} for {raw_path}")
                continue

        try:
            payload = _build_serato_markers2_payload(cues)

            try:
                tags = ID3(raw_path)
            except Exception:
                # Pristine file with no ID3 header — create one
                tags = ID3()
                tags.save(raw_path)
            # Remove existing Serato Markers2
            for key in list(tags.keys()):
                frame = tags[key]
                if isinstance(frame, GEOB) and frame.desc == "Serato Markers2":
                    del tags[key]

            tags.add(GEOB(
                encoding=0,
                mime="application/octet-stream",
                desc="Serato Markers2",
                data=payload,
            ))
            tags.save()
            count += 1

        except Exception as e:
            errors.append(f"Failed to write Serato markers to {raw_path}: {e}")

    if errors:
        import warnings
        warnings.warn(f"Serato export: {len(errors)} errors: {errors[:3]}")

    return count


def _build_serato_markers2_payload(cues: list[dict[str, Any]]) -> bytes:
    """Build Serato Markers2 binary payload."""
    # Inner payload (before base64)
    inner = bytearray(b"\x01\x01")  # version

    for i, cue in enumerate(cues[:8]):
        entry = _build_serato_cue_entry(cue, index=i)
        inner += b"CUE\x00"
        inner += struct.pack(">I", len(entry))
        inner += entry

    # Outer: version + base64 of inner
    encoded = base64.b64encode(bytes(inner))
    return b"\x01\x01" + encoded


def _build_serato_cue_entry(cue: dict[str, Any], index: int = 0) -> bytes:
    """Build a single CUE entry for Serato Markers2."""
    rgb = _RGB.get(cue.get("colour", "green"), (0, 255, 0))
    name = cue.get("name", "").encode("utf-8") + b"\x00"
    position_ms = int(cue.get("position_ms", 0))

    data = bytearray()
    data.append(0x00)  # padding
    data.append(index & 0xFF)
    data += struct.pack(">I", position_ms)
    data.append(0x00)
    data += bytes(rgb)
    data += b"\x00\x00"
    data += name
    return bytes(data)


# ---------------------------------------------------------------------------
# Engine DJ (SQLite)
# ---------------------------------------------------------------------------

def write_engine_cues(
    tracks: list[dict[str, Any]],
    db_path: str | Path,
) -> int:
    """Write cue points to an Engine DJ SQLite database.

    Returns number of tracks written.
    """
    import sqlite3

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Engine DJ database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    count = 0

    try:
        for t in tracks:
            track_id = t.get("engine_track_id")
            if not track_id:
                continue

            for i, cue in enumerate(t.get("cues", [])[:8]):
                rgb = _RGB.get(cue.get("colour", "green"), (0, 255, 0))
                color_int = (0xFF << 24) | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                position_sec = cue.get("position_ms", 0) / 1000.0

                conn.execute("""
                    INSERT OR REPLACE INTO CuePoint
                    (trackId, type, label, length, time, isEnabled, color, hotCueNumber)
                    VALUES (?, 0, ?, 0.0, ?, 1, ?, ?)
                """, (track_id, cue.get("name", ""), position_sec, color_int, i))

            count += 1
        conn.commit()
    finally:
        conn.close()

    return count


# ---------------------------------------------------------------------------
# VirtualDJ XML
# ---------------------------------------------------------------------------

def write_virtualdj_xml(
    tracks: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Generate a VirtualDJ database XML fragment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("VirtualDJ_Database", Version="8")

    for t in tracks:
        raw_path = _strip_file_uri(t.get("path", ""))
        from urllib.parse import quote
        tags_str = (
            f"artist={quote(t.get('artist', ''), safe='')}"
            f"&title={quote(t.get('title', ''), safe='')}"
            f"&genre={quote(t.get('genre', ''), safe='')}"
            f"&bpm={t.get('bpm', 0):.2f}"
        )

        song = ET.SubElement(root, "Song",
                             FilePath=raw_path,
                             Tags=tags_str)

        for i, cue in enumerate(t.get("cues", [])[:8]):
            rgb = _RGB.get(cue.get("colour", "green"), (0, 255, 0))
            color_hex = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            pos_sec = cue.get("position_ms", 0) / 1000.0
            ET.SubElement(song, "Poi",
                          Type="cue",
                          Num=str(i + 1),
                          Pos=f"{pos_sec:.3f}",
                          Name=cue.get("name", ""),
                          Color=color_hex)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path


# ---------------------------------------------------------------------------
# Unified export
# ---------------------------------------------------------------------------

def export_cues(
    tracks: list[dict[str, Any]],
    output_format: str,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Export cues to any supported format.

    Parameters
    ----------
    output_format : "rekordbox", "traktor", "serato", "engine", "virtualdj"
    """
    if output_format == "traktor":
        return write_traktor_nml(tracks, output_path or "traktor_collection.nml")
    elif output_format == "serato":
        return write_serato_markers(tracks)
    elif output_format == "engine":
        return write_engine_cues(tracks, kwargs.get("db_path", output_path))
    elif output_format == "virtualdj":
        return write_virtualdj_xml(tracks, output_path or "virtualdj_database.xml")
    elif output_format == "rekordbox":
        from .sync import generate_cue_xml
        return generate_cue_xml(tracks)
    else:
        raise ValueError(f"Unknown format: {output_format}. Supported: rekordbox, traktor, serato, engine, virtualdj")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_file_uri(path: str) -> str:
    """Remove file://localhost prefix if present."""
    if path.startswith("file://localhost"):
        return path[len("file://localhost"):]
    if path.startswith("file://"):
        from urllib.parse import unquote, urlparse
        return unquote(urlparse(path).path)
    return path
