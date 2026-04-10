"""Memory system — load, save, migrate, validate, and backup.

Tracks are keyed by content hash (SHA-256 of the audio file) so that
moving or renaming a file does not orphan the entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import MemoryConfig

CURRENT_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_memory(config: MemoryConfig) -> dict[str, Any]:
    """Load memory from disk with automatic migration and validation."""
    path = _resolve(config.path)
    if not path.exists():
        return _empty()

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("version", 1) < CURRENT_SCHEMA_VERSION:
        data = _migrate_v1_to_v2(data)

    _validate(data)
    return data


def save_memory(data: dict[str, Any], config: MemoryConfig) -> None:
    """Atomic write with backup rotation."""
    path = _resolve(config.path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Rotate backups
    if path.exists() and config.auto_backup:
        _rotate_backups(path, config.backup_count)

    # Stamp
    data["version"] = CURRENT_SCHEMA_VERSION
    data["last_run"] = datetime.now().isoformat()

    # Atomic write: .tmp → rename
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(path)
    except OSError:
        # Rename failed (file locked on Windows, permissions, etc.)
        # Clean up temp file to avoid stale .tmp
        tmp.unlink(missing_ok=True)
        raise


def store_track_analysis(
    memory: dict[str, Any],
    path: str | Path,
    analysis: dict[str, Any],
) -> str:
    """Store a full track analysis result in memory.

    Keys by content hash so entries survive file moves/renames.
    Returns the content hash used as key.
    """
    content_hash = hash_file_content(path)

    entry = {
        "path": str(path),
        "content_hash": content_hash,
        "analysed_at": datetime.now().isoformat(),
    }
    entry.update(analysis)

    memory["processed_tracks"][content_hash] = entry
    return content_hash


def get_track_analysis(
    memory: dict[str, Any],
    path: str | Path,
) -> dict[str, Any] | None:
    """Retrieve a stored analysis by content hash. Returns None if not found."""
    content_hash = hash_file_content(path)
    return memory["processed_tracks"].get(content_hash)


def hash_file_content(path: str | Path, chunk_size: int = 65536) -> str:
    """Stream-hash a file in chunks.  Never loads the whole file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve(raw: str) -> Path:
    return Path(os.path.expanduser(raw))


def _empty() -> dict[str, Any]:
    return {
        "version": CURRENT_SCHEMA_VERSION,
        "processed_tracks": {},
        "energy_corrections": [],
        "energy_calibration": {"global_offset": 0.0, "genre_offsets": {}},
        "custom_tag_rules": [],
        "tag_corrections": [],
        "artist_corrections": [],
        "settings": {},
        "last_run": None,
    }


def _rotate_backups(path: Path, keep: int) -> None:
    """Rotate path → path.1 → path.2 → … → path.{keep}."""
    for i in range(keep, 0, -1):
        src = path.with_suffix(f".json.{i - 1}") if i > 1 else path
        dst = path.with_suffix(f".json.{i}")
        if src.exists():
            shutil.copy2(src, dst)


def _resolve_file_path(raw_path: str) -> str:
    """Convert a Rekordbox file:// URI to a local absolute path."""
    if raw_path.startswith("file://"):
        raw_path = unquote(urlparse(raw_path).path)
    return raw_path


def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Re-key processed_tracks by content hash instead of path hash.

    For files that no longer exist at the stored path the old key is kept
    and ``needs_rehash`` is set so a future session can fix it.
    """
    old_tracks = data.get("processed_tracks", {})
    new_tracks: dict[str, Any] = {}

    for old_key, entry in old_tracks.items():
        raw = entry.get("path", "")
        local = _resolve_file_path(raw)
        if Path(local).is_file():
            content_hash = hash_file_content(local)
            entry["content_hash"] = content_hash
            new_tracks[content_hash] = entry
        else:
            entry["needs_rehash"] = True
            new_tracks[old_key] = entry

    data["processed_tracks"] = new_tracks
    data["version"] = CURRENT_SCHEMA_VERSION
    return data


def _validate(data: dict[str, Any]) -> None:
    """Basic structural validation."""
    if not isinstance(data.get("processed_tracks"), dict):
        raise ValueError("processed_tracks must be a dict")
    if not isinstance(data.get("energy_calibration"), dict):
        raise ValueError("energy_calibration must be a dict")
    if data.get("version") != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"expected version {CURRENT_SCHEMA_VERSION}, got {data.get('version')}"
        )
