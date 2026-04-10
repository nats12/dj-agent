---
name: detect-vocals
description: Classify tracks as vocal, instrumental, or partial-vocal. Results written to My Tag system.
---

# Detect Vocals

Classify tracks as **vocal**, **instrumental**, or **partial vocal**. Results are written to the Rekordbox My Tag system — filterable on CDJs/XDJs during live performance.

## Two Tiers

### Fast (Essentia, ~2-5s/track)
Binary voice/instrumental classifier using Essentia TF models. Falls back to a librosa harmonic-percussive heuristic if Essentia is unavailable.

```python
from dj_agent.vocals import detect_vocals_fast
result = detect_vocals_fast("/path/to/track.flac")
print(f"{result.classification} (confidence: {result.vocal_probability:.0%})")
```

### Thorough (Demucs, ~30s/track GPU)
Separates stems with Demucs htdemucs, computes vocal RMS / mix RMS ratio.

```python
from dj_agent.vocals import detect_vocals_thorough
result = detect_vocals_thorough("/path/to/track.flac")
```

## Classification Thresholds

| Probability | Classification |
|-------------|---------------|
| < 0.25 | Instrumental |
| 0.25 - 0.60 | Partial Vocal |
| > 0.60 | Vocal |

## Why This Matters

Mixing two vocal tracks together sounds muddy. Knowing which tracks have vocals lets you avoid vocal clashes in transitions and build sets with intentional vocal/instrumental alternation.

## Important

**Do NOT use inaSpeechSegmenter** for this — it classifies singing as "music", not "speech". It only detects spoken word (MC intros, radio drops).

## Workflow

1. `detect vocals` — scan library or playlist
2. Results written to My Tag: "Vocal", "Instrumental", "Partial Vocal"
3. Filter on CDJs by My Tag during live performance
