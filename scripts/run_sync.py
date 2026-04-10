"""Execute the full sync: DB writes + cues XML export.

Matching strategy:
- Memory entries are matched to DB tracks by filename
- Built-in Rekordbox paths (Sampler loops, PioneerDJ demos) are skipped
  to avoid generic-filename collisions

Write strategy:
- Energy My Tags + colour from memory (audio-derived)
- Title/artist cleanups apply cleanup_title / cleanup_artist to the
  CURRENT DB value and write only when the cleaner differs (never
  overwrites user-curated metadata with filename-parsed values)
- Cues XML uses the CLEANED title/artist for each track so Rekordbox's
  XML import cannot clobber the DB values we just wrote
- master.db is backed up before any write; if --xml-out points at an
  existing file, it is backed up too

Usage:
    python scripts/run_sync.py
    python scripts/run_sync.py --xml-out /path/to/rekordbox.xml

Without --xml-out the XML is written to
``<rekordbox.xml_output_dir>/rekordbox_YYYY-MM-DD_HHMMSS.xml`` (per
config.yaml). Pass --xml-out to overwrite a specific file — e.g. the
Imported Library path you have set in Rekordbox Preferences.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyrekordbox import Rekordbox6Database

from dj_agent.cleanup import cleanup_artist, cleanup_title
from dj_agent.config import get_config
from dj_agent.sync import (
    generate_cue_xml,
    is_builtin_rekordbox_path,
    set_artist,
    update_title,
)
from dj_agent.tags import write_energy_mytags

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def basename(p: str) -> str:
    return (p or "").replace("\\", "/").rsplit("/", 1)[-1]


def backup_file(path: Path, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup = path.with_name(f"{path.name}.{tag}-{stamp}")
    shutil.copy2(path, backup)
    return backup


def default_db_path() -> Path | None:
    """Best-effort default location of Rekordbox's master.db.

    Mirrors ``Rekordbox6Database.__init__`` which tries ``rekordbox7``
    first and falls back to ``rekordbox6``. On Rekordbox 7 installs the
    ``rekordbox6`` section is empty, so the v6-only lookup returns None
    and no backup is made. Check v7 first.
    """
    try:
        from pyrekordbox.config import get_config as rb_get_config  # type: ignore[import-untyped]
        for section in ("rekordbox7", "rekordbox6"):
            rb_cfg = rb_get_config(section)
            if isinstance(rb_cfg, dict):
                db_path = rb_cfg.get("db_path")
                if db_path:
                    return Path(db_path)
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml-out",
        type=Path,
        default=None,
        help="Explicit XML output path. Defaults to a timestamped file "
             "under config.rekordbox.xml_output_dir.",
    )
    args = parser.parse_args()

    config = get_config()
    memory_path = Path(config.memory.path).expanduser()

    print("=== dj-agent sync ===")
    print(f"Memory:    {memory_path}")
    db_hint = default_db_path()
    if db_hint:
        print(f"DB:        {db_hint}")
    if args.xml_out:
        print(f"XML out:   {args.xml_out}  (explicit)")
    else:
        print(
            f"XML out:   {Path(config.rekordbox.xml_output_dir).expanduser()}"
            f"/rekordbox_<timestamp>.xml"
        )
    print()

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

    # Backup DB (only if we can locate it deterministically)
    if db_hint and db_hint.exists():
        print(f"\nBacking up {db_hint.name}...")
        db_backup = backup_file(db_hint, "sync-backup")
        print(f"  -> {db_backup.name}")

    db = Rekordbox6Database()
    matches: list[tuple[object, dict]] = []
    skipped_builtin = 0
    for c in db.get_content():
        if is_builtin_rekordbox_path(c.FolderPath or ""):
            skipped_builtin += 1
            continue
        fn = basename(c.FolderPath or "")
        mem = mem_by_fn.get(fn)
        if mem is not None:
            matches.append((c, mem))
    print(f"DB matched:          {len(matches)}")
    print(f"Skipped (built-in):  {skipped_builtin}")

    xml_tracks: list[dict] = []
    title_updates = 0
    artist_updates = 0
    colour_updates = 0
    energy_results: list[dict] = []
    now = datetime.now(timezone.utc)

    for c, mem in matches:
        # Title cleanup in place
        db_title = c.Title or ""
        new_title = db_title
        if db_title:
            cleaned, _ = cleanup_title(db_title)
            if cleaned and cleaned != db_title:
                update_title(c, cleaned)
                new_title = cleaned
                title_updates += 1

        # Artist cleanup in place
        db_artist = c.ArtistName or ""
        new_artist = db_artist
        if db_artist:
            cleaned, _ = cleanup_artist(db_artist)
            if cleaned and cleaned != db_artist:
                set_artist(db, c, cleaned)
                new_artist = cleaned
                artist_updates += 1

        # Colour from energy
        if mem.get("energy_colour_id") is not None:
            try:
                new_colour = int(mem["energy_colour_id"])
                if (c.ColorID or 0) != new_colour:
                    c.ColorID = new_colour
                    c.updated_at = now
                    colour_updates += 1
            except (TypeError, ValueError):
                pass

        # Energy My Tag (batched write below)
        if mem.get("energy"):
            energy_results.append(
                {"db_content_id": str(c.ID), "energy": int(mem["energy"])}
            )

        # Collect for XML with cleaned values so import cannot clobber DB
        if mem.get("cues"):
            disk_path = (c.FolderPath or "").replace("\\", "/")
            xml_tracks.append(
                {
                    "db_content_id": str(c.ID),
                    "title": new_title,
                    "artist": new_artist,
                    "path": disk_path,
                    "cues": mem["cues"],
                }
            )

    print("\nWriting energy My Tags...")
    write_energy_mytags(db, energy_results)
    print(f"  {len(energy_results)} tracks tagged")

    print("\nCommitting DB changes...")
    db.flush()
    db.commit()
    print(f"  title updates:  {title_updates}")
    print(f"  artist updates: {artist_updates}")
    print(f"  colour updates: {colour_updates}")
    print(f"  energy tags:    {len(energy_results)}")

    print(f"\nGenerating cues XML ({len(xml_tracks)} tracks)...")
    if args.xml_out and args.xml_out.exists():
        backup = backup_file(args.xml_out, "sync-backup")
        print(f"  backed up existing XML -> {backup.name}")
    xml_path = generate_cue_xml(
        xml_tracks,
        config=config.rekordbox,
        output_path=args.xml_out,
    )
    print(f"  wrote {xml_path}")

    memory["last_run"] = datetime.now().isoformat()
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated memory last_run -> {memory['last_run']}")

    print("\n=== done ===")
    print("Next step: open Rekordbox, Preferences -> Advanced -> Database ->")
    print("rekordbox xml -> set Imported Library to the XML above -> OK.")
    print("In the sidebar, expand rekordbox xml -> right-click the playlist")
    print("-> Import to Collection.")


if __name__ == "__main__":
    main()
