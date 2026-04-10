---
name: check-beatgrid
description: Verify BPM accuracy, detect half/double errors, flag tempo drift.
---

# Check Beat Grid

Verify Rekordbox BPM values against neural beat detection. Catches common errors.

## What It Detects

| Issue | Description |
|-------|-------------|
| Half BPM | Rekordbox says 70, should be 140 |
| Double BPM | Rekordbox says 260, should be 130 |
| Wrong BPM | Detected BPM differs by >2% |
| Tempo drift | BPM changes within the track (live recordings) |

## Usage

```python
from dj_agent.beatgrid import verify_bpm, detect_tempo_drift

result = verify_bpm("/path/to/track.flac", rekordbox_bpm=128.0, genre="Techno")
if result["issue"]:
    print(f"Issue: {result['issue']}, suggested: {result['suggested_bpm']}")

drift = detect_tempo_drift("/path/to/track.flac")
if drift:
    print(f"Tempo drift at {drift[0]['time']}s: {drift[0]['bpm']} vs median {drift[0]['median_bpm']}")
```

## Workflow

1. `check beatgrid` — scan library for BPM issues
2. Review flagged tracks
3. Fix in Rekordbox or let the agent suggest corrections
