"""Sync results back to Rekordbox — DB writes + XML export.

Fixes:
- Artist is set via ArtistID (not the read-only ArtistName proxy)
- XML uses manual ElementTree (never pyrekordbox.RekordboxXml)
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RekordboxConfig


# Path substrings that identify Rekordbox built-in tracks (sampler loops,
# demo tracks) which must never be matched by filename-based sync, because
# they use generic filenames like "House1.wav", "Techno1.wav", "Demo Track
# 1.mp3" that collide with real user library entries. Anything whose
# FolderPath contains one of these substrings should be skipped by the
# caller's matching loop.
BUILTIN_PATH_MARKERS: tuple[str, ...] = (
    "rekordbox/Sampler",
    "PioneerDJ/Demo",
)


def match_memory_to_db(
    mem_by_fn: dict[str, dict],
    db_contents: list,
) -> list[tuple]:
    """Match memory entries to DB tracks by filename, with a prefix
    fallback for truncated filenames.

    The Linux analysis pipeline sometimes truncates long filenames
    (Beatport store IDs, remix parentheses). A prefix match recovers
    these without false positives as long as the shared prefix is ≥15
    characters.

    Returns a list of ``(content, memory_entry)`` tuples for matched
    tracks. Built-in Rekordbox tracks (Sampler/Demo) are skipped.
    """
    def _basename(p: str) -> str:
        return (p or "").replace("\\", "/").rsplit("/", 1)[-1]

    def _stem(fn: str) -> str:
        return fn.rsplit(".", 1)[0] if "." in fn else fn

    # Build a stem→filename index for prefix matching
    mem_by_stem: dict[str, list[str]] = {}
    for fn in mem_by_fn:
        s = _stem(fn)
        mem_by_stem.setdefault(s, []).append(fn)

    matches: list[tuple] = []
    for c in db_contents:
        if is_builtin_rekordbox_path(c.FolderPath or ""):
            continue
        fn = _basename(c.FolderPath or "")

        # Exact match (fast path)
        mem = mem_by_fn.get(fn)
        if mem is not None:
            matches.append((c, mem))
            continue

        # Prefix fallback for truncated filenames
        db_stem = _stem(fn)
        best_fn = None
        best_len = 0
        for mem_s, mem_fns in mem_by_stem.items():
            if len(mem_s) < 15:
                continue
            if db_stem.startswith(mem_s) or mem_s.startswith(db_stem):
                shared = min(len(db_stem), len(mem_s))
                if shared > best_len:
                    best_len = shared
                    best_fn = mem_fns[0]
        if best_fn is not None:
            matches.append((c, mem_by_fn[best_fn]))

    return matches


def is_builtin_rekordbox_path(folder_path: str) -> bool:
    """Return True if the given FolderPath is a Rekordbox built-in track.

    Filename-based sync must skip these — they use generic filenames that
    collide with real library tracks (e.g. House1.wav, Techno1.wav).
    """
    normalized = (folder_path or "").replace("\\", "/")
    return any(marker in normalized for marker in BUILTIN_PATH_MARKERS)


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def set_artist(db: Any, content: Any, artist_name: str) -> None:
    """Set an artist on a track via ArtistID (not the read-only proxy).

    Creates the DjmdArtist row if it doesn't exist yet.
    """
    from pyrekordbox.db6 import tables  # type: ignore[import-untyped]

    existing = (
        db.session.query(tables.DjmdArtist)
        .filter_by(Name=artist_name)
        .first()
    )
    if not existing:
        now = datetime.now(timezone.utc)
        existing = tables.DjmdArtist(
            ID=str(uuid.uuid4().int % 4_294_967_295 + 1),
            Name=artist_name,
            UUID=str(uuid.uuid4()),
            rb_data_status=0,
            rb_local_data_status=0,
            rb_local_deleted=0,
            rb_local_synced=0,
            rb_local_usn=100,
            created_at=now,
            updated_at=now,
        )
        db.session.add(existing)
        db.session.flush()

    content.ArtistID = existing.ID
    content.updated_at = datetime.now(timezone.utc)


def update_title(content: Any, title: str) -> None:
    """Safely set the track title (direct column write)."""
    content.Title = title
    content.updated_at = datetime.now(timezone.utc)


def update_genre(content: Any, genre: str) -> None:
    """Set genre (always title-cased)."""
    content.GenreName = genre.title()
    content.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# XML export (hot cues only)
# ---------------------------------------------------------------------------
#
# Rekordbox XML import for cues requires a CLEAN-SLATE workflow:
#
#   1. Delete existing DjmdCue + ContentCue rows for each track
#      (Rekordbox 7 refuses to overwrite existing cues via XML import —
#      "Import to Collection" silently no-ops on tracks that already
#      have any cue data, even when the XML file contains new cues).
#
#   2. Generate XML with these requirements (learned from comparing
#      Rekordbox's own XML export against our broken output):
#      - TRACK elements MUST include TotalTime (seconds) — without it,
#        Rekordbox may skip the track entirely on import.
#      - Hot cue POSITION_MARK (Num="0".."7") includes RGB color attrs.
#      - Memory cue POSITION_MARK (Num="-1") must NOT include RGB —
#        Rekordbox's own export omits Red/Green/Blue from memory cues.
#      - A PLAYLISTS section with at least one playlist node is needed
#        for the sidebar right-click → "Import to Collection" target.
#
#   3. User imports via: Preferences → Advanced → Database → rekordbox
#      xml → Browse to the XML → OK → sidebar → expand rekordbox xml →
#      right-click the playlist → Import to Collection.
#
#   4. Cache defeat: if Rekordbox doesn't pick up the new XML, point
#      the Imported Library to a DIFFERENT xml file first, OK, then
#      point back to the real one to force a re-read.
#
# Rekordbox 7 architecture note: DjmdCue rows are a secondary index
# that Rekordbox rebuilds from ContentCue.Cues (a JSON blob) on
# startup. Direct DjmdCue writes without matching ContentCue updates
# get silently reverted. The XML import path handles both tables
# correctly, which is why it's the recommended approach.
# ---------------------------------------------------------------------------

def _format_cue_name(cue: dict) -> str:
    """Format a cue name for XML export.

    segment_energy is stored in the cue dict for programmatic use
    (analytics, set builder, smart playlists) but is NOT included in
    the display name. Rekordbox 7.2.13's XML import silently drops
    all cues after the first if any cue name contains a colon (the
    "E:8" suffix we used to append). Plain names import reliably.
    """
    return cue.get("name") or "Cue"


COLOUR_MAP = {
    "green": {"Red": "40", "Green": "226", "Blue": "20"},
    "red": {"Red": "230", "Green": "50", "Blue": "50"},
    "blue": {"Red": "50", "Green": "100", "Blue": "230"},
    "yellow": {"Red": "230", "Green": "200", "Blue": "30"},
}


def rb_url_encode(path: str) -> str:
    """Encode a file path the way Rekordbox does in its XML.

    Rekordbox encodes spaces, ampersands, and other URI-unsafe characters
    but leaves parentheses, commas, hash signs, and drive-letter colons
    literal.

    Output format: ``file://localhost/<abs path>`` — the leading slash
    after ``localhost`` is always required. On Unix the path already
    starts with ``/``; on Windows the path starts with ``C:/...`` and
    we must prepend a ``/`` so the final URL is
    ``file://localhost/C:/...`` (with the colon kept literal).
    """
    from urllib.parse import quote
    if not path.startswith("/"):
        path = "/" + path
    # safe list includes ":" so Windows drive letters stay literal.
    return "file://localhost" + quote(path, safe="/:(),#;!~")


def generate_cue_xml(
    tracks: list[dict[str, Any]],
    config: RekordboxConfig | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Generate a Rekordbox-compatible XML with hot cues.

    Parameters
    ----------
    tracks : list[dict]
        Each dict needs: ``path``, ``title``, ``artist``, ``db_content_id``,
        ``cues`` (list of :class:`CuePoint`-like dicts).
    config : RekordboxConfig, optional
        Used only when ``output_path`` is not given; the XML is written
        as ``<xml_output_dir>/rekordbox_YYYY-MM-DD_HHMMSS.xml``.
    output_path : Path or str, optional
        Explicit output file path. When set, overrides the timestamped
        default. Useful for overwriting an existing ``rekordbox.xml``
        that the user already has configured as their Imported Library.

    Returns
    -------
    Path to the generated XML file.
    """
    if output_path is not None:
        xml_path = Path(output_path).expanduser()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if config is None:
            from .config import get_config
            config = get_config().rekordbox

        output_dir = Path(config.xml_output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        xml_path = output_dir / f"rekordbox_{stamp}.xml"

    # Filter to tracks that have cues (so Entries count matches actual TRACK elements)
    tracks_with_cues = [t for t in tracks if t.get("cues")]

    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="rekordbox", Version="7.2.8", Company="AlphaTheta")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks_with_cues)))

    for t in tracks_with_cues:

        raw_path = t.get("path", "")
        # Strip file://localhost prefix to get absolute path for encoding
        if raw_path.startswith("file://localhost"):
            raw_path = raw_path[len("file://localhost"):]

        # Populate TRACK with the same attributes Rekordbox's own XML
        # export includes. TotalTime is critical — without it, Rekordbox
        # may skip the track entirely on "Import to Collection".
        track_attrs = {
            "TrackID": str(t.get("db_content_id", "0")),
            "Name": t.get("title", ""),
            "Artist": t.get("artist", ""),
            "Location": rb_url_encode(raw_path),
        }
        if t.get("total_time"):
            track_attrs["TotalTime"] = str(int(t["total_time"]))
        if t.get("bpm"):
            track_attrs["AverageBpm"] = f"{float(t['bpm']):.2f}"
        if t.get("tonality"):
            track_attrs["Tonality"] = str(t["tonality"])

        track_el = ET.SubElement(collection, "TRACK", **track_attrs)

        # Rekordbox's XML format requires every hot cue to appear TWICE:
        # once in a hot-cue slot (Num="0".."7", i.e. Hot Cue A..H) and once
        # as a memory cue (Num="-1"). Cues flagged ``memory_only`` get only
        # the Num="-1" entry (structural markers beyond the 8 hot-cue limit
        # — the user still sees them in Rekordbox as memory cues).
        all_cues = t.get("cues", [])
        hot_cues = [c for c in all_cues if not c.get("memory_only")][:8]
        memory_only_cues = [c for c in all_cues if c.get("memory_only")]

        # Hot cue slots (Num 0..7 = A..H) — with RGB colour attributes
        for idx, cue in enumerate(hot_cues):
            rgb = COLOUR_MAP.get(cue.get("colour", "green"), COLOUR_MAP["green"])
            pos_sec = cue.get("position_ms", 0) / 1000.0
            ET.SubElement(
                track_el,
                "POSITION_MARK",
                Name=_format_cue_name(cue),
                Type="0",
                Start=f"{pos_sec:.3f}",
                Num=str(idx),
                **rgb,
            )
        # Memory cue counterparts for hot cues — NO RGB attributes.
        # Rekordbox's own XML export omits Red/Green/Blue from memory
        # cue entries (Num="-1"); including them may cause import to
        # silently skip the track.
        for cue in hot_cues:
            pos_sec = cue.get("position_ms", 0) / 1000.0
            ET.SubElement(
                track_el,
                "POSITION_MARK",
                Name=_format_cue_name(cue),
                Type="0",
                Start=f"{pos_sec:.3f}",
                Num="-1",
            )
        # Memory-only cues (overflow beyond the 8 hot-cue limit) — no RGB
        for cue in memory_only_cues:
            pos_sec = cue.get("position_ms", 0) / 1000.0
            ET.SubElement(
                track_el,
                "POSITION_MARK",
                Name=_format_cue_name(cue),
                Type="0",
                Start=f"{pos_sec:.3f}",
                Num="-1",
            )

    # PLAYLISTS section — required for reliable "Import to Collection".
    # Without at least one playlist node, the left sidebar under
    # "rekordbox xml" shows only a flat "All Tracks" view which has no
    # consistent right-click import action across Rekordbox versions.
    # Wrap the synced tracks in a single playlist the user can target.
    playlists = ET.SubElement(root, "PLAYLISTS")
    playlist_root = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    playlist = ET.SubElement(
        playlist_root,
        "NODE",
        Name="dj-agent sync",
        Type="1",                    # 1 = playlist (0 = folder)
        KeyType="0",                 # 0 = reference tracks by TrackID
        Entries=str(len(tracks_with_cues)),
    )
    for t in tracks_with_cues:
        ET.SubElement(playlist, "TRACK", Key=str(t.get("db_content_id", "0")))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)

    return xml_path
