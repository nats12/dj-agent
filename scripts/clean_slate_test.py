"""Clean-slate test: delete all cues for a single track from both
DjmdCue + ContentCue, then regenerate the XML with TotalTime + no-RGB
fixes. The user then imports from XML to add fresh cues.

Usage:
    python scripts/clean_slate_test.py --track 45771899
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
from pyrekordbox.db6 import tables

from dj_agent.config import get_config
from dj_agent.sync import generate_cue_xml, match_memory_to_db

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def basename(p: str) -> str:
    return (p or "").replace("\\", "/").rsplit("/", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=str, required=True, help="TrackID to clean")
    parser.add_argument(
        "--xml-out", type=Path, default=None,
        help="XML output path (default: timestamped in config xml_output_dir)",
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

    db = Rekordbox6Database()
    c = db.get_content(ID=args.track)
    if c is None:
        print(f"Track {args.track} not found")
        return

    content_id = str(c.ID)
    print(f"Track: {c.Title!r} by {c.ArtistName!r}  ID={content_id}")
    print(f"  File: {c.FolderPath}")

    # Step 1: Delete existing cues (clean slate)
    cue_rows = db.session.query(tables.DjmdCue).filter_by(ContentID=content_id).all()
    cc_rows = db.session.query(tables.ContentCue).filter_by(ContentID=content_id).all()
    print(f"\nDeleting {len(cue_rows)} DjmdCue + {len(cc_rows)} ContentCue rows...")
    for r in cue_rows:
        db.session.delete(r)
    for r in cc_rows:
        db.session.delete(r)
    db.flush()
    db.commit()
    print("  done (DB committed, cues = 0)")

    # Verify
    remaining = db.session.query(tables.DjmdCue).filter_by(ContentID=content_id).count()
    remaining_cc = db.session.query(tables.ContentCue).filter_by(ContentID=content_id).count()
    print(f"  verify: DjmdCue={remaining}, ContentCue={remaining_cc}")

    # Step 2: Regenerate XML with TotalTime + no-RGB-on-memory-cues
    all_content = list(db.get_content())
    matched = match_memory_to_db(mem_by_fn, all_content)

    xml_tracks: list[dict] = []
    for content, mem in matched:
        if not mem.get("cues"):
            continue
        bpm = (content.BPM or 0) / 100.0 if content.BPM else None
        key_obj = content.Key
        tonality = key_obj.ScaleName if key_obj else None
        total_time = content.Length

        xml_tracks.append({
            "db_content_id": str(content.ID),
            "title": content.Title or "",
            "artist": content.ArtistName or "",
            "path": (content.FolderPath or "").replace("\\", "/"),
            "cues": mem["cues"],
            "total_time": total_time,
            "bpm": bpm,
            "tonality": tonality,
        })

    print(f"\nXML: {len(xml_tracks)} tracks total (including {args.track})")

    xml_out = args.xml_out
    if xml_out and xml_out.exists():
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup = xml_out.with_name(f"{xml_out.name}.clean-slate-{stamp}")
        shutil.copy2(xml_out, backup)
        print(f"  backed up XML -> {backup.name}")

    xml_path = generate_cue_xml(
        xml_tracks,
        config=config.rekordbox,
        output_path=xml_out,
    )
    print(f"  wrote {xml_path}")

    # Verify the target track is in the XML
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(xml_path))
    for tr in tree.getroot().find("COLLECTION"):
        if tr.get("TrackID") == args.track:
            marks = tr.findall("POSITION_MARK")
            hot = [m for m in marks if m.get("Num") != "-1"]
            mem_marks = [m for m in marks if m.get("Num") == "-1"]
            has_tt = tr.get("TotalTime")
            has_rgb_on_mem = any(m.get("Red") for m in mem_marks)
            print(f"\n  Target track in XML:")
            print(f"    TotalTime={has_tt!r}")
            print(f"    Hot cues: {len(hot)}, Memory cues: {len(mem_marks)}")
            print(f"    RGB on memory cues: {has_rgb_on_mem}")
            print(f"    Sample hot cue: {dict(hot[0].attrib) if hot else 'none'}")
            print(f"    Sample mem cue: {dict(mem_marks[0].attrib) if mem_marks else 'none'}")
            break

    print("\n=== Next steps ===")
    print("1. Open Rekordbox")
    print("2. Preferences -> Advanced -> Database -> rekordbox xml")
    print("3. Browse -> select a DIFFERENT xml -> OK (cache defeat)")
    print("4. Browse -> select the real xml -> OK")
    print(f"5. Sidebar -> expand rekordbox xml -> find 'dj-agent sync'")
    print(f"6. Find '{c.Title}' in the playlist -> right-click -> Import to Collection")


if __name__ == "__main__":
    main()
