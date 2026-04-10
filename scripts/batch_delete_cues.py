"""Delete all DjmdCue + ContentCue rows for every matched track.
Clean slate so XML import can write fresh cues."""
import io
import json
import sys
from pathlib import Path

from pyrekordbox import Rekordbox6Database
from pyrekordbox.db6 import tables

from dj_agent.config import get_config
from dj_agent.sync import match_memory_to_db

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def basename(p: str) -> str:
    return (p or "").replace("\\", "/").rsplit("/", 1)[-1]


def main() -> None:
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
    all_content = list(db.get_content())
    matched = match_memory_to_db(mem_by_fn, all_content)
    print(f"Matched tracks: {len(matched)}")

    total_cue = 0
    total_cc = 0
    for c, mem in matched:
        cid = str(c.ID)
        n1 = db.session.query(tables.DjmdCue).filter_by(ContentID=cid).delete()
        n2 = db.session.query(tables.ContentCue).filter_by(ContentID=cid).delete()
        total_cue += n1
        total_cc += n2

    print(f"Deleted: {total_cue} DjmdCue rows + {total_cc} ContentCue rows")
    print("Committing...")
    db.flush()
    db.commit()
    print("Done.")

    # Verify
    sample = db.session.query(tables.DjmdCue).filter_by(ContentID="45771899").count()
    print(f"Verify I Wanna Go (45771899): DjmdCue={sample}")


if __name__ == "__main__":
    main()
