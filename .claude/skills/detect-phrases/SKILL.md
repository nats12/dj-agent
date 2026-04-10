---
name: detect-phrases
description: Detect musical phrases (8/16/32 bar sections) and classify as intro, build, drop, breakdown, outro.
---

# Detect Phrases

Identify musical phrase boundaries and classify sections. Feeds into cue detection for more accurate placement.

## Usage

```python
from dj_agent.phrases import detect_phrases
phrases = detect_phrases("/path/to/track.flac", bpm=128.0)
for p in phrases:
    print(f"{p.label:<12} {p.start_ms/1000:.1f}s-{p.end_ms/1000:.1f}s ({p.bar_count} bars, energy: {p.energy:.2f})")
```

## How It Works

1. **madmom** neural downbeat detection (preferred) or librosa beat tracking (fallback)
2. Group downbeats into bars (4 beats per bar for 4/4 time)
3. Group bars into phrases (default: 8 bars per phrase)
4. Classify by position and energy: intro, build, drop, breakdown, outro

## Key Finding

73.6% of DJ cue points fall within 8 bars, 86.2% within 16 bars.

## Workflow

1. `detect phrases` — analyse library or playlist
2. Results stored in memory for use by cue detection and set building
