---
name: build-set
description: Optimise track ordering for DJ sets using transition scoring and energy arc constraints.
---

# Build Set

Order tracks for optimal transitions with an energy arc.

## How It Works

1. Score every pair of tracks by transition compatibility (harmonic, BPM, energy, genre, vocal overlap)
2. Find the optimal ordering using greedy nearest-neighbor + 2-opt local search
3. Optionally constrain to an energy arc (warmup → build → peak → cooldown)

## Energy Arcs

| Arc | Pattern |
|-----|---------|
| `warmup_to_peak` | 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 10 → 9 |
| `peak_time` | 7 → 8 → 9 → 10 → 10 → 9 → 10 → 10 → 9 → 8 |
| `chill_set` | 3 → 4 → 5 → 5 → 4 → 5 → 6 → 5 → 4 → 3 |
| `full_night` | 2 → 3 → 5 → 6 → 7 → 8 → 9 → 10 → 10 → 9 → 8 → 7 → 5 → 3 |
| `flat` | No constraint |

## Usage

```python
from dj_agent.setbuilder import build_set
ordered = build_set(tracks, energies=energy_dict, arc="warmup_to_peak")
```

## Transition Scoring (detailed)

```python
from dj_agent.transitions import score_transition, rate_transition
result = score_transition(track_a, track_b, energy_a=7, energy_b=8)
print(f"Score: {result['total']:.2f} ({rate_transition(result['total'])})")
print(f"  Harmonic: {result['harmonic']}, BPM: {result['bpm']}, Energy: {result['energy']}")
```

## Workflow

1. `build set for [playlist]` — optimise track order
2. Choose energy arc based on gig type
3. Review transitions (green/yellow/red ratings)
4. Export reordered playlist to Rekordbox
