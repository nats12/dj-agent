---
name: enrich-metadata
description: Enrich track metadata from online sources — MusicBrainz, Last.fm, Discogs, Beatport.
---

# Enrich Metadata

Look up tracks on multiple online databases to fill missing genre, mood, style, year, and label information.

## Sources

| Source | What it provides | Requires |
|--------|-----------------|----------|
| MusicBrainz | Track ID, release info | `pip install musicbrainzngs` |
| Last.fm | Genre tags, mood tags | `LASTFM_API_KEY` env var |
| Discogs | Genre, subgenre/style, year, label | `DISCOGS_USER_TOKEN` env var |
| Beatport | EDM genre taxonomy (most granular) | Web scraping (future) |

Spotify Audio Features API was **deprecated November 2024** and is not used.

## Usage

```python
from dj_agent.metadata import enrich_track_metadata

result = enrich_track_metadata("Charlotte de Witte", "Overdrive")
print(f"Genres: {result['genre_tags']}")
print(f"Styles: {result['style_tags']}")
print(f"Year: {result['year']}, Label: {result['label']}")
```

## Features

- Each source is independently optional — if a library isn't installed or an API key is missing, that source is skipped
- Rate limiting with backoff to avoid API bans
- Results cached in `memory.json` to avoid repeated lookups
- Graceful degradation: partial results are still useful

## Workflow

1. `enrich metadata` — look up untagged tracks
2. Review proposed genre/style fills
3. Accept or correct — corrections saved to memory
4. Write accepted metadata to Rekordbox DB
