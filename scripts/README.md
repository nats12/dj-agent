# scripts/

Standalone sync utilities. All three read `config.yaml` for the memory
file path and XML output directory, and use `pyrekordbox`'s default
location for `master.db`.

| Script | What it does |
|---|---|
| `sync_preview.py` | Dry-run preview of `/sync` — shows exactly what would be written to Rekordbox. No DB changes, no XML written. Safe to run anytime. |
| `run_sync.py` | Execute the sync. Writes energy My Tags, colours, title/artist cleanups to `master.db`; generates a cues XML for you to import in Rekordbox. Backs up `master.db` before any write. |
| `regen_xml.py` | Regenerate only the cues XML from the current DB + memory, without re-running any DB writes. Use after a bug fix in `generate_cue_xml`. |

## Usage

```bash
# 1. See what would change
python scripts/sync_preview.py

# 2. Do it — writes to master.db and a timestamped XML file
python scripts/run_sync.py

# 3. Or overwrite an existing Imported Library file
python scripts/run_sync.py --xml-out ~/Documents/DJ/rekordbox.xml
```

**Before running `run_sync.py`**, close Rekordbox — it holds a lock on
`master.db` while running and any writes will be silently overwritten
when it exits.

## Matching strategy

Memory entries are matched to DB tracks by **filename**. Built-in
Rekordbox content (`rekordbox/Sampler/…`, `PioneerDJ/Demo…`) is skipped
to avoid collisions with generic filenames like `House1.wav`.

Title and artist fields are cleaned **in place** by running
`cleanup_title` / `cleanup_artist` on the current DB value. If the
cleaner produces no change, the field is left alone — so
user-curated metadata is never overwritten with filename-parsed values.

Hot cues are exported via XML (never direct DB write) because cue
points live in linked tables that require Rekordbox's own import to
stay consistent. The XML uses the **cleaned** title/artist for each
track so Rekordbox's XML import cannot clobber the values we just
wrote to the DB.
