---
name: harmonic-mix
description: Suggest harmonically compatible tracks for mixing using the Camelot wheel, BPM, and energy.
---

# Harmonic Mix

Suggest next tracks that will mix harmonically with the current track.

## Camelot Wheel Compatibility

| Move | Relationship | Effect |
|------|-------------|--------|
| Same key | Identical | Safe, no clash |
| ±1 number | Adjacent | Smooth transition |
| Switch A↔B | Relative major/minor | Mood shift |
| +2 numbers | Energy boost | Bold, exciting |

## Transition Scoring

```
score = 0.35 × harmonic + 0.25 × BPM + 0.20 × energy + 0.20 × genre
```

## Usage

```python
from dj_agent.harmonic import suggest_harmonic_transitions
suggestions = suggest_harmonic_transitions(current_track, library, top_k=10)
for s in suggestions:
    print(f"{s.track.artist} - {s.track.title} ({s.key_relation}, score: {s.score:.2f})")
```

## Workflow

1. `suggest mix for [track]` — show compatible next tracks
2. `harmonic order [playlist]` — reorder a playlist for harmonic flow
