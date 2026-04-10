---
name: check-quality
description: Audio quality validation — fake FLAC detection, clipping, silence, bitrate checks.
---

# Check Quality — Audio Validation

Scan tracks for quality issues: fake lossless files, clipping, excessive silence, and low bitrate.

## What It Checks

| Check | What it detects | Speed |
|-------|----------------|-------|
| Fake lossless | MP3 transcoded to FLAC (spectral cutoff below expected Nyquist) | ~2-5s |
| Clipping | Consecutive max-amplitude samples indicating distortion | ~1s |
| Silence | Leading, trailing, and mid-track dead air > 2 seconds | ~0.5s |
| Bitrate | Files below 320kbps | instant |
| Format info | Codec, sample rate, bits per sample via ffprobe | instant |

## Usage

```python
from dj_agent.quality import check_audio_quality

report = check_audio_quality("/path/to/track.flac")

if report.is_fake_lossless:
    print(f"WARNING: Likely transcoded (confidence: {report.fake_lossless_confidence:.0%})")
if report.clipping_count > 0:
    print(f"WARNING: {report.clipping_count} clipping events")
if report.leading_silence_ms > 3000:
    print(f"Leading silence: {report.leading_silence_ms}ms")
for warning in report.warnings:
    print(f"  ⚠ {warning}")
```

## Output Format

```
QUALITY REPORT — 847 tracks scanned

Fake lossless (likely transcoded):
  ⚠ Track A.flac — spectral cutoff at 16kHz (confidence: 90%)
  ⚠ Track B.flac — spectral cutoff at 18kHz (confidence: 50%)

Clipping detected:
  ⚠ Track C.wav — 12 clipping events

Excessive silence:
  ⚠ Track D.mp3 — 8.2s leading silence
  ⚠ Track E.flac — 3.5s dead air at 2:45

Low bitrate (<320kbps):
  ⚠ Track F.mp3 — 192kbps
```

## Workflow

1. `check quality` — scan full library or playlist
2. Review flagged tracks
3. Replace fake lossless with genuine copies if available
4. Re-encode clipped tracks from better sources
