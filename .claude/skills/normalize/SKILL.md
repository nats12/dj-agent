---
name: normalize
description: Measure and normalize track loudness (LUFS). Critical for consistent volume on CDJs.
---

# Normalize — Volume Consistency

Measure integrated LUFS per track and optionally normalize to a target level. CDJs ignore Rekordbox Auto Gain metadata — normalizing the actual audio is the only way to get consistent volume on club systems.

## Two Modes

### Measure Only (default)
Show LUFS, true peak, and loudness range for each track. No files are modified.

```python
from dj_agent.normalize import measure_track, measure_batch, format_loudness_report

# Single track
result = measure_track("/path/to/track.flac")
print(f"LUFS: {result.integrated_lufs:.1f}, Peak: {result.true_peak_dbtp:.1f} dBTP, LRA: {result.loudness_range_lu:.1f} LU")

# Batch
results = measure_batch(track_paths)
print(format_loudness_report(results, target_lufs=-8.0))
```

### Normalize (creates copies)
Create normalized copies in a separate directory. **Never modifies originals.**

```python
from dj_agent.normalize import normalize_track

result = normalize_track(
    input_path="/path/to/track.flac",
    output_path="/path/to/Normalized/track.flac",
)
print(f"Normalized: {result['original_lufs']:.1f} → {result['normalized_lufs']:.1f} LUFS")
```

## Target LUFS

| Context | Target | Notes |
|---------|--------|-------|
| Club / CDJ | -8 LUFS | Default — punchy for club systems |
| Streaming | -14 LUFS | Spotify/Apple Music target |
| Broadcast | -23 LUFS | EBU R128 |

Configurable in `config.yaml` under `normalize.target_lufs`.

## Loudness Range (LRA)

Tracks with mismatched LRA sound jarring in transitions. The report shows LRA so you can group tracks by dynamic range:
- Club EDM: typically 3-7 LU
- Chill / ambient: 8-15 LU

## Workflow

1. `normalize` or `measure loudness` — show LUFS report for library or playlist
2. Review outliers (tracks far from target)
3. If normalizing: creates copies in `Normalized/` directory
4. Optionally writes ReplayGain tags for software players
