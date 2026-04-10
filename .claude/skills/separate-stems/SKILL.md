---
name: separate-stems
description: Separate tracks into stems (vocals, drums, bass, other) using BS-RoFormer, Demucs, or SAM Audio.
---

# Separate Stems

Split tracks into individual stems for mashups, acapellas, instrumentals, or creative remixing.

## Engines

### BS-RoFormer (default, SOTA vocals)
2-stem model (vocals + instrumental). SDR 12.97 on vocals — best single-file vocal model as of 2026. Default for acapella/instrumental workflows.

### Mel-Band RoFormer (ensemble partner)
SDR 11.44. Cleaner on exposed pop vocals, fewer metallic artifacts. Used alongside BS-RoFormer in `best` quality preset for ~+0.3 dB improvement.

### HTDemucs (fast multi-stem)
4-stem (vocals/drums/bass/other) or 6-stem (+guitar/piano). Lower vocal quality (SDR ~9) but gives you all stems in one pass. Best for quick splits.

### SAM Audio (text-prompted, flexible)
Meta's foundation model. Isolate **any sound** by describing it: "singing voice", "synth pad", "hi-hats", "crowd noise". One stem per inference pass. Best for non-standard separations that fixed-stem models can't do.

## Quality Presets

| Preset | Engine | Speed (GPU) | Best for |
|--------|--------|-------------|----------|
| `fast` | HTDemucs 4-stem | ~10s/track | Batch processing, quick previews |
| `balanced` | BS-RoFormer 2-stem | ~25s/track | Default — best quality/speed ratio |
| `best` | RoFormer ensemble | ~50s/track | Single-track mashup work |
| `sam` | SAM Audio | ~30s/stem | Non-standard separations, creative use |

## Usage

```python
from dj_agent.stems import (
    separate_stems,          # All stems as numpy arrays
    export_stems,            # All stems saved as WAV files
    create_acapella,         # Vocal-only WAV
    create_instrumental,     # Everything-minus-vocals WAV
    create_acapella_and_instrumental,  # Both in one pass (2x faster)
    separate_with_prompt,    # SAM Audio text-prompted separation
)
```

## Workflow

1. User says "separate stems", "export stems", "make acapella", "make instrumental", or "isolate [sound]"
2. For standard stems (vocals, drums, bass, other): use BS-RoFormer/Demucs via `separate_stems()` or convenience functions
3. For non-standard requests ("isolate the synth", "extract the crowd noise"): use SAM Audio via `separate_with_prompt()`
4. Output as 24-bit WAV files in `{track_stem}_stems/` directory
5. Report what was created and where

## Examples

**Standard separation:**
```
"separate stems for track X"     → export_stems(path)
"make an acapella of track X"    → create_acapella(path)
"make an instrumental"           → create_instrumental(path)
"separate with best quality"     → export_stems(path, quality="best")
```

**SAM Audio (text-prompted):**
```
"isolate the synth pad"          → separate_with_prompt(path, "synthesizer pad")
"extract the hi-hats"            → separate_with_prompt(path, "hi-hat cymbals")
"pull out the crowd noise"       → separate_with_prompt(path, "crowd cheering")
"isolate the piano"              → separate_with_prompt(path, "piano playing")
```

## Notes

- GPU strongly recommended. CPU works but is 5-10x slower.
- SAM Audio requires `pip install sam-audio` and HuggingFace auth (`huggingface-cli login`). Model is gated.
- SAM Audio large (3B params) needs ~16 GB VRAM. Use `sam-base` (1B, ~8 GB) or `sam-small` (500M, ~4 GB) for smaller GPUs.
- BS-RoFormer 2-stem produces a cleaner instrumental than summing 4 Demucs stems (leakage compounds on sum).
- For DJ use: acapella + instrumental from one pass is the most common workflow. Use `create_acapella_and_instrumental()`.
