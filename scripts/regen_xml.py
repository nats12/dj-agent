"""Regenerate only the cues XML, reading CURRENT DB values for title,
artist, and path. Use this after run_sync.py when only the XML needs
rewriting — e.g. after fixing a bug in generate_cue_xml — without
re-running any DB writes.

Usage:
    python scripts/regen_xml.py
    python scripts/regen_xml.py --xml-out /path/to/rekordbox.xml
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pyrekordbox import Rekordbox6Database

from dj_agent.config import get_config
from dj_agent.sync import generate_cue_xml, match_memory_to_db

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def basename(p: str) -> str:
    return (p or "").replace("\\", "/").rsplit("/", 1)[-1]


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

    db_path = Path.home() / ".local" / "share" / "Pioneer" / "rekordbox" / "master.db"
    if db_path.exists():
        db = Rekordbox6Database(path=db_path)
    else:
        db = Rekordbox6Database()
    all_content = list(db.get_content())
    matched = match_memory_to_db(mem_by_fn, all_content)

    xml_tracks: list[dict] = []
    for c, mem in matched:
        if not mem.get("cues"):
            continue
        bpm = (c.BPM or 0) / 100.0 if c.BPM else None
        key_obj = c.Key
        tonality = key_obj.ScaleName if key_obj else None
        total_time = c.Length  # seconds

        xml_tracks.append(
            {
                "db_content_id": str(c.ID),
                "title": c.Title or "",
                "artist": c.ArtistName or "",
                "path": (c.FolderPath or "").replace("\\", "/"),
                "cues": mem["cues"],
                "total_time": total_time,
                "bpm": bpm,
                "tonality": tonality,
            }
        )

    print(f"XML tracks: {len(xml_tracks)}  (from {len(matched)} matched, built-in skipped)")

    if args.xml_out and args.xml_out.exists():
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup = args.xml_out.with_name(
            f"{args.xml_out.name}.regen-backup-{stamp}"
        )
        shutil.copy2(args.xml_out, backup)
        print(f"backed up previous XML -> {backup.name}")

    xml_path = generate_cue_xml(
        xml_tracks,
        config=config.rekordbox,
        output_path=args.xml_out,
    )
    print(f"wrote {xml_path}")


if __name__ == "__main__":
    main()
