---
name: sync
description: Write results back to Rekordbox - direct DB for most metadata, XML only for hot cues.
---

# Sync to Rekordbox

Most metadata goes direct to DB. XML is only needed for hot cues.

## Why two methods?

Hot cues in Rekordbox's DB are stored across multiple linked tables with complex relationships, beat grid references, and internal checksums. Writing them directly risks corruption. XML import is Rekordbox's official supported way to bring in cues — it handles all the internal linking safely.

Everything else (title, artist, genre, energy tags, comments) is simple field writes on `DjmdContent` or related tables and can go direct to DB.

## What goes where

| Data | Method | Why |
|------|--------|-----|
| Track title | **DB direct** | Simple field: `content.Title` |
| Track artist | **DB direct** | Simple field: `content.ArtistName` |
| Genre | **DB direct** | Simple field: `content.GenreName` / `content.Genre` |
| Energy tags (My Tag) | **DB direct** | `DjmdMyTag` / `DjmdSongMyTag` tables |
| Comments cleanup | **DB direct** | Simple field: `content.Commnt` |
| Rating & colour | **DB direct** | Simple fields: `content.Rating`, `content.ColorID` |
| Hot cue points | **XML import** | Complex DB structure, XML is the safe route |

## Step 1: Direct DB Write (with Rekordbox closed)

**CRITICAL: Always ask the user to confirm Rekordbox is closed before proceeding.** Do NOT assume it is closed. Rekordbox overwrites the DB while running, so any writes will be lost silently. Use AskUserQuestion to confirm.

Writes:
- Energy ratings → My Tag system (`DjmdMyTag` / `DjmdSongMyTag` tables)
- Title cleanup → `content.Title`
- Artist cleanup → `content.ArtistName`
- Genre casing fixes → `content.GenreName`
- Clear old agent tags from Comments → `content.Commnt`

```python
from pyrekordbox import Rekordbox6Database
import html
db = Rekordbox6Database()

content = db.get_content(ID=content_id)

# Title/artist cleanup — unescape HTML entities, split artist from title
content.Title = html.unescape(original_title).strip()
content.ArtistName = extracted_artist

# Update timestamp
from datetime import datetime
content.updated_at = datetime.now()

db.flush()
db.commit()
```

## Step 2: XML Export (only if hot cues were generated)

Only generate XML when there are new hot cues to write. Skip this step entirely for cleanup-only or energy-only runs.

- Filename: `rekordbox_YYYY-MM-DD_HHMMSS.xml`
- Output directory: `~/Documents/DJ/dj-agent/`
- **Never write energy or tags to Comments or Label in XML**
- Pass through BPM and key unchanged

### CRITICAL: Do NOT use pyrekordbox's RekordboxXml for XML generation

`RekordboxXml.add_track()` double-encodes Location paths, breaking Rekordbox import (tracks won't match). Always build XML manually with `xml.etree.ElementTree`.

Location path format must match Rekordbox's own export exactly:
- Prefix: `file://localhost` (single, no double)
- Only encode: spaces → `%20`, `&` → `%26`
- Leave everything else literal (commas, parentheses, `#`, `;`, etc.)

```python
import xml.etree.ElementTree as ET

def rb_url_encode(path):
    """Encode a file path the way Rekordbox does in its XML."""
    return 'file://localhost' + path.replace(' ', '%20').replace('&', '%26')

root = ET.Element('DJ_PLAYLISTS', Version='1.0.0')
ET.SubElement(root, 'PRODUCT', Name='rekordbox', Version='7.2.8', Company='AlphaTheta')
collection = ET.SubElement(root, 'COLLECTION', Entries=str(n))

track = ET.SubElement(collection, 'TRACK',
    TrackID=str(content_id),
    Name=title,
    Artist=artist,
    Location=rb_url_encode(raw_path),
)

# Add POSITION_MARK elements for each hot cue
colour_map = {
    'green': {'Red': '40', 'Green': '226', 'Blue': '20'},
    'red': {'Red': '230', 'Green': '50', 'Blue': '50'},
    'blue': {'Red': '50', 'Green': '100', 'Blue': '230'},
    'yellow': {'Red': '230', 'Green': '200', 'Blue': '30'},
}
for idx, cue in enumerate(cues[:8]):
    rgb = colour_map.get(cue['colour'], colour_map['green'])
    ET.SubElement(track, 'POSITION_MARK',
        Name=cue['name'], Type='cue',
        Start=f"{cue['position_ms'] / 1000.0:.3f}",
        Num=str(idx), **rgb,
    )

tree = ET.ElementTree(root)
ET.indent(tree, space='  ')
tree.write(xml_path, encoding='utf-8', xml_declaration=True)
```

Note: HTML entities in filenames (e.g. `&amp;` in `The Obsessed &amp; DBF`) are **literal characters on disk**. Do NOT `html.unescape()` file paths — only unescape title/artist display fields.

## Step 2b: Clean-slate cue deletion (REQUIRED before XML import)

**Rekordbox 7 refuses to overwrite existing cues via XML import.** "Import to
Collection" silently no-ops on any track that already has cue data in the DB.
To write new cues, you MUST delete the existing cues first:

```python
from pyrekordbox import Rekordbox6Database
from pyrekordbox.db6 import tables

db = Rekordbox6Database()
for content_id in track_ids_to_update:
    db.session.query(tables.DjmdCue).filter_by(ContentID=content_id).delete()
    db.session.query(tables.ContentCue).filter_by(ContentID=content_id).delete()
db.flush()
db.commit()
```

Both `DjmdCue` AND `ContentCue` must be cleared. In Rekordbox 7, `ContentCue.Cues`
(a JSON blob) is the authoritative cue store — `DjmdCue` rows are a secondary index
rebuilt from the JSON on startup. Deleting only `DjmdCue` has no effect; Rekordbox
regenerates it from `ContentCue.Cues`.

## Step 3: XML Import (user does this manually, only when cues were generated)

### XML format requirements (critical for Rekordbox 7)

- **TRACK elements MUST include `TotalTime`** (seconds) — without it, Rekordbox may
  skip the track on import.
- **Hot cue POSITION_MARK** (`Num="0".."7"`) includes `Red`, `Green`, `Blue` attrs.
- **Memory cue POSITION_MARK** (`Num="-1"`) must **NOT** include RGB attributes —
  Rekordbox's own export omits them from memory cues.
- **A `PLAYLISTS` section** with at least one playlist node is required for the
  sidebar right-click → "Import to Collection" target.

### Import steps

1. Open Rekordbox
2. Preferences → Advanced → Database → rekordbox xml
3. **Cache defeat**: Browse → select a DIFFERENT xml first → OK → then Browse → select
   the real XML → OK (forces Rekordbox to re-read the file)
4. In sidebar, expand "rekordbox xml"
5. Find the "dj-agent sync" playlist → right-click → Import to Collection

### What gets updated on import
- Hot cue points (POSITION_MARK elements)

### What Rekordbox preserves
- BPM and key (passed through unchanged)
- Playlists and playlist order

### What gets REPLACED on import
- Existing cue points on tracks where we deleted them in Step 2b

## Agent Workflow

1. Ask user to confirm Rekordbox is closed
2. DB write: energy tags, title/artist cleanup, genre, comments
3. If new hot cues were generated:
   a. Delete existing DjmdCue + ContentCue rows for target tracks (clean slate)
   b. Generate XML with TotalTime, RGB on hot cues only, PLAYLISTS section
   c. Tell user to open Rekordbox, cache-defeat, and import
4. If no cues: done — no XML needed, user can just open Rekordbox
5. Save memory file
