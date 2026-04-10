"""Read Rekordbox's PSSI (Phrase Structure) from ANLZ files and update
memory.json with beat-grid-aligned cue points.

Delegates to ``dj_agent.cues.detect_cue_points_from_pssi`` — the same
function ``pipeline.py`` uses — so PSSI→cue logic is shared. When
running on a machine with the Rekordbox library (ANLZ files available),
this gives much better cues than the librosa-based fallback, because
Rekordbox uses an ML model trained specifically on DJ music structure.

Usage:
    python scripts/pssi_to_cues.py --track 22156096 --dry-run
    python scripts/pssi_to_cues.py --all
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from pyrekordbox import Rekordbox6Database

from dj_agent.config import get_config
from dj_agent.cues import detect_cue_points_from_pssi
from dj_agent.sync import match_memory_to_db

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ANLZ_BASE = Path(
    r"C:/Users/Antonio Fortes/AppData/Roaming/Pioneer/rekordbox/share"
)


def basename(p: str) -> str:
    return (p or "").replace("\\", "/").rsplit("/", 1)[-1]


def read_pssi_cues(content: object, verbose: bool = False) -> list[dict] | None:
    """Read PSSI for a track and return cue dicts, or None."""
    ap = content.AnalysisDataPath
    if not ap:
        return None
    anlz_path = ANLZ_BASE / ap.lstrip("/")
    bpm = (content.BPM or 0) / 100.0
    if bpm <= 0:
        return None

    cue_points = detect_cue_points_from_pssi(anlz_path, bpm=bpm)
    if cue_points is None:
        return None

    cue_dicts = [
        {"name": c.name, "position_ms": c.position_ms, "colour": c.colour,
         "confidence": c.confidence, "memory_only": c.memory_only}
        for c in cue_points
    ]
    if verbose:
        print(f"  PSSI: {len(cue_dicts)} cues, BPM={bpm}")
        for c in cue_dicts:
            m, s = divmod(c["position_ms"] / 1000, 60)
            mo = " MEM_ONLY" if c.get("memory_only") else ""
            print(f"    {int(m)}:{s:05.2f}  {c['name']:<14}{mo}")
    return cue_dicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--track", type=str, help="Single TrackID")
    g.add_argument("--all", action="store_true", help="All matched tracks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = get_config()
    memory_path = Path(config.memory.path).expanduser()
    with open(memory_path, encoding="utf-8") as f:
        memory = json.load(f)

    mem_by_fn: dict[str, dict] = {}
    fn_to_hash: dict[str, str] = {}
    for h, entry in memory["processed_tracks"].items():
        fn = entry.get("filename") or basename(entry.get("path", ""))
        if fn not in mem_by_fn or entry.get("analysed_at", "") > mem_by_fn[fn].get("analysed_at", ""):
            mem_by_fn[fn] = entry
            fn_to_hash[fn] = h

    db = Rekordbox6Database()

    if args.track:
        c = db.get_content(ID=args.track)
        if not c:
            print(f"Track {args.track} not found"); return
        fn = basename(c.FolderPath or "")
        mem = mem_by_fn.get(fn)
        if not mem:
            print(f"No memory entry for {fn}"); return
        targets = [(c, mem, fn)]
    else:
        matched = match_memory_to_db(mem_by_fn, list(db.get_content()))
        targets = [(c, m, m.get("filename") or basename(m.get("path", "")))
                    for c, m in matched]

    print(f"Targets: {len(targets)}")
    updated = 0
    no_pssi = 0

    for c, mem, fn in targets:
        cues = read_pssi_cues(c, verbose=(args.track is not None))
        if cues is None:
            no_pssi += 1
            continue
        if not args.dry_run:
            h = fn_to_hash.get(fn)
            if h and h in memory["processed_tracks"]:
                memory["processed_tracks"][h]["cues"] = cues
                updated += 1

    print(f"\nUpdated: {updated}  No PSSI: {no_pssi}")
    if not args.dry_run and updated > 0:
        print(f"Saving memory to {memory_path}...")
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print("Done.")
    elif args.dry_run:
        print("DRY RUN — no changes saved.")


if __name__ == "__main__":
    main()
