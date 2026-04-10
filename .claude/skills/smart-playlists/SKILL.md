---
name: smart-playlists
description: Generate Rekordbox playlists using boolean algebra rules on metadata and tags.
---

# Smart Playlists

Create dynamic playlists using rules — fills the gap between Serato Smart Crates (powerful) and Rekordbox Intelligent Playlists (limited).

## Rule Syntax

```
genre:Techno AND energy:8+
genre:House OR genre:Techno AND NOT vocal
bpm:125-135 AND key:8B
artist:Bicep
genre:Techno AND energy:7+ AND mood:dark
```

## Operators

| Operator | Example | Meaning |
|----------|---------|---------|
| `AND` | `genre:Techno AND energy:8+` | Both must match |
| `OR` | `key:8B OR key:9B` | Either can match |
| `NOT` | `NOT vocal` | Must not match |
| `:value` | `genre:Techno` | Exact match |
| `:N+` | `energy:8+` | Greater or equal |
| `:N-` | `energy:5-` | Less or equal |
| `:lo-hi` | `bpm:125-135` | Range match |

## Available Fields

TrackInfo fields: `genre`, `artist`, `title`, `bpm`, `key`, `duration`, `bitrate`
Tag fields: `energy`, `mood`, `vocal`, `hardness`, `commercial` (from My Tag analysis)

## Usage

```python
from dj_agent.smartlists import filter_tracks
result = filter_tracks(library, "genre:Techno AND energy:8+ AND NOT vocal", all_tags)
```

## Workflow

1. `smart playlist [rule]` — create a playlist matching the rule
2. Results shown for review
3. Export to Rekordbox as a new playlist
