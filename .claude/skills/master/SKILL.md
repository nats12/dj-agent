---
name: master
description: Platinum Notes-style audio mastering — multiband dynamics, clip repair, EQ, limiting. Creates new files, never modifies originals.
---

# Master — Audio Processing

One-click audio improvement for your DJ library. Replicates Platinum Notes functionality: volume standardization, clipped peak repair, multiband dynamics, shelving EQ, and final limiting.

**WARNING:** This actively reshapes dynamics and frequency balance. Well-mastered tracks may sound worse. Always preview before batch processing.

## Templates

| Template | Target LUFS | Character | Best For |
|----------|-------------|-----------|----------|
| **Official** | -8 | Gentle, preserves dynamics | Well-mastered tracks, general use |
| **Festival** | -7 | Punchy bass, bright presence | Festival/mainstage, needs to compete loud |
| **Big Boost** | -6 | Maximum loudness, aggressive | Quiet tracks, older tracks |
| **Gentle** | -9 | Minimal processing | Tracks that just need consistency |

## Processing Chain

1. **Clip repair** — spline interpolation reconstructs clipped peaks
2. **Multiband compression** — 4-band (0-200Hz, 200-2kHz, 2k-8kHz, 8kHz+) via Linkwitz-Riley crossovers
3. **Shelving EQ** — bass/treble shaping per template
4. **LUFS gain** — adjust to target loudness
5. **Brick-wall limiter** — final peak safety

## Usage

```python
from dj_agent.master import master_track, format_comparison

result = master_track("track.flac", template="official")
print(format_comparison(result))
```

## Before/After Metrics

Every processed track gets a comparison report:
- LUFS (perceived loudness)
- Peak dBFS
- Dynamic range (dB)
- Spectral centroid (brightness)

## Rules

- **Never modifies originals** — creates `_mastered` copies
- **Matches input format** — FLAC→FLAC, WAV→WAV, MP3→MP3 (no lossy re-encode)
- **Not part of the `magic` pipeline** — must be explicitly requested
- **Not idempotent** — don't process already-mastered files

## Workflow

1. `master [playlist]` or `master [track]` — process with Official template
2. `master [playlist] with festival template` — use a specific template
3. Review before/after metrics
4. Preview in Rekordbox before committing to the processed versions
