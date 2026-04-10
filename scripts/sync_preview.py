"""Dry-run preview of /sync: show exactly what dj-agent would write
to Rekordbox, without touching the DB or writing any XML.

Matching strategy:
- Memory entries are matched to DB tracks by filename
- Built-in Rekordbox paths (Sampler loops, PioneerDJ demos) are skipped
  to avoid generic-filename collisions

Write strategy (mirrors run_sync.py):
- Energy My Tags + colour come from memory (audio-derived)
- Title/artist cleanups apply cleanup_title / cleanup_artist to the
  CURRENT DB value and report only when the cleaner differs (never
  overwrites user-curated metadata with filename-parsed values)
- Cues are counted for an XML export
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

from pyrekordbox import Rekordbox6Database

from dj_agent.cleanup import cleanup_artist, cleanup_title
from dj_agent.config import get_config
from dj_agent.sync import is_builtin_rekordbox_path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def basename(p: str) -> str:
    return (p or "").replace("\\", "/").rsplit("/", 1)[-1]


def main() -> None:
    config = get_config()
    memory_path = Path(config.memory.path).expanduser()

    print(f"Memory: {memory_path}")
    with open(memory_path, encoding="utf-8") as f:
        memory = json.load(f)

    mem_by_fn: dict[str, dict] = {}
    for entry in memory["processed_tracks"].values():
        fn = entry.get("filename") or basename(entry.get("path", ""))
        if fn in mem_by_fn:
            if entry.get("analysed_at", "") > mem_by_fn[fn].get("analysed_at", ""):
                mem_by_fn[fn] = entry
        else:
            mem_by_fn[fn] = entry
    print(f"Memory entries: {len(memory['processed_tracks'])}")

    db = Rekordbox6Database()
    matches: list[tuple[object, dict]] = []
    skipped_builtin = 0
    unmatched_db = 0
    for c in db.get_content():
        if is_builtin_rekordbox_path(c.FolderPath or ""):
            skipped_builtin += 1
            continue
        fn = basename(c.FolderPath or "")
        mem = mem_by_fn.get(fn)
        if mem is None:
            unmatched_db += 1
            continue
        matches.append((c, mem))

    print(f"DB matched (by filename):    {len(matches)}")
    print(f"DB skipped (built-in paths): {skipped_builtin}")
    print(f"DB without memory entry:     {unmatched_db}")

    energy_writes = 0
    colour_writes = 0
    title_updates: list[tuple] = []
    artist_updates: list[tuple] = []
    cues_to_export = 0
    energy_distribution: Counter = Counter()

    for c, mem in matches:
        if mem.get("energy"):
            energy_writes += 1
            energy_distribution[mem["energy"]] += 1

        if mem.get("energy_colour_id") is not None:
            try:
                new_colour = int(mem["energy_colour_id"])
                if (c.ColorID or 0) != new_colour:
                    colour_writes += 1
            except (TypeError, ValueError):
                pass

        db_title = c.Title or ""
        if db_title:
            cleaned, changes = cleanup_title(db_title)
            if cleaned and cleaned != db_title:
                title_updates.append((c.ID, db_title, cleaned, changes))

        db_artist = c.ArtistName or ""
        if db_artist:
            cleaned, changes = cleanup_artist(db_artist)
            if cleaned and cleaned != db_artist:
                artist_updates.append((c.ID, db_artist, cleaned, changes))

        if mem.get("cues"):
            cues_to_export += 1

    print()
    print("=== Planned writes (DRY RUN) ===")
    print(f"Energy My Tags to write: {energy_writes}")
    print(f"  distribution: {dict(sorted(energy_distribution.items()))}")
    print(f"Colour updates:          {colour_writes}")
    print(f"Title updates:           {len(title_updates)}")
    for db_id, old, new, changes in title_updates[:20]:
        print(f"    {db_id}: {old!r} -> {new!r}  {changes}")
    if len(title_updates) > 20:
        print(f"    ... and {len(title_updates) - 20} more")
    print(f"Artist updates:          {len(artist_updates)}")
    for db_id, old, new, changes in artist_updates[:20]:
        print(f"    {db_id}: {old!r} -> {new!r}  {changes}")
    if len(artist_updates) > 20:
        print(f"    ... and {len(artist_updates) - 20} more")
    print(f"Cues to export (XML):    {cues_to_export}")

    print()
    print("=== NOT touching ===")
    print("  BPM, Key, Rating, Comments, manual cue points")


if __name__ == "__main__":
    main()
