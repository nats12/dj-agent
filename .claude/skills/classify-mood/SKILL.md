---
name: classify-mood
description: Classify track mood/vibe (dark, euphoric, aggressive, chill, melancholic) plus commercial vs underground and hardness scoring.
---

# Classify Mood

Tag tracks by emotional character — not just energy (intensity) but the *vibe*. Results written to My Tag system.

## Two Tiers

### Essentia Models (fast, ~3s/track)
Five binary mood classifiers: aggressive, happy, party, relaxed, sad. Plus arousal/valence regression for a continuous emotion space.

```python
from dj_agent.mood import classify_mood_essentia
result = classify_mood_essentia("/path/to/track.flac")
print(f"Mood: {result.primary_mood}, Arousal: {result.arousal:.2f}, Valence: {result.valence:.2f}")
```

### CLAP Zero-Shot (flexible, ~5s/track)
Define your own mood labels in `config.yaml` — no retraining needed:

```yaml
mood_labels:
  - "Dark and Hypnotic warehouse techno"
  - "Happy and Euphoric festival anthem"
  - "Aggressive and Hard industrial"
  - "Chill and Relaxed deep house"
  - "Melancholic and Emotional progressive"
```

```python
from dj_agent.mood import classify_mood_clap
result = classify_mood_clap("/path/to/track.flac")
```

## Additional Classifications

### Commercial vs Underground
```python
from dj_agent.mood import classify_commercial
score = classify_commercial("/path/to/track.flac")  # 0.0=underground, 1.0=commercial
```

### Hardness Score (1-10)
Combines energy, BPM, and mood. Adapted from normalization suite.
```python
from dj_agent.mood import calculate_hardness
hardness = calculate_hardness(energy=0.8, bpm=140, mood="aggressive")  # → 10
```

## My Tag Structure

```
Mood (parent)
├── Mood:Dark
├── Mood:Euphoric
├── Mood:Aggressive
├── Mood:Chill
└── Mood:Melancholic
```

## Workflow

1. `classify mood` — scan library or playlist
2. Review results (especially border cases)
3. Correct any misclassifications — agent learns from corrections
4. Results written to My Tag system
