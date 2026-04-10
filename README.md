# DJ Agent

An AI-powered DJ library enrichment tool — energy ratings, cue points, key detection, metadata cleanup, LUFS normalization, stem separation, mood classification, harmonic mixing, mastering, and more.

Works with **Rekordbox** (DB + XML), exports to **Traktor**, **Serato**, **Engine DJ**, and **VirtualDJ**. Powered by [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) with optional **Gemini** AI reasoning.

> **Fork-friendly.** Fork this repo and make it your own.

---

## Features

### Core Analysis
| Feature | Detail |
|---|---|
| **Energy ratings** | LUFS-based energy scoring (1–10) with configurable weights, written to Rekordbox My Tag system |
| **Key detection** | Essentia EDMA + librosa fallback, Camelot notation, piano chord verification |
| **Hot cue detection** | PSSI-first, phrase-aware cue detection with adaptive segmentation and dual thresholds |
| **Metadata cleanup** | Artist/title splitting, HTML entity fixing, watermark removal, smart title casing |
| **Duplicate detection** | Chunked file hashing + artist-prefix blocking fuzzy match |
| **Beat grid verification** | Half/double BPM fix with genre-aware nearest-octave selection |
| **Library analytics** | Genre/BPM/key distributions, coverage gaps, metadata completeness scoring |

### Audio Processing
| Feature | Detail |
|---|---|
| **LUFS normalization** | Measure + normalize to target LUFS (-8 club, -14 streaming), ReplayGain tags |
| **Audio mastering** | Multiband dynamics (4-band Linkwitz-Riley), clip repair, shelving EQ, limiting |
| **Audio quality** | Fake FLAC detection (brick-wall spectral analysis), clipping, silence detection |
| **Stem separation** | MelBand-Roformer (SOTA) via audio-separator, Demucs fallback |
| **Pitch shifting** | pyrubberband (Ableton engine) + pedalboard fallback, key-aware shifting |

### DJ Intelligence
| Feature | Detail |
|---|---|
| **Harmonic mixing** | Full Camelot wheel, symmetric transition scoring, harmonic suggestions |
| **Set building** | TSP-based track ordering with energy arc constraints |
| **Smart playlists** | Boolean algebra rules engine with parentheses (`genre:Techno AND energy:8+`) |
| **Mashup engine** | Key/BPM/vocal compatibility scoring with transition tips |
| **Mood classification** | Essentia 5-mood + CLAP zero-shot with custom DJ labels |
| **Vocal detection** | Essentia fast pass + Demucs thorough (vocal/instrumental/partial) |

### AI Reasoning (optional)
| Feature | Detail |
|---|---|
| **Vibe analysis** | "Dark hypnotic warehouse techno with rolling bass" — via Gemini or Flamingo |
| **Transition advice** | DJ mixing technique suggestions between two tracks |
| **Nuance tagging** | Bassline type, vocal style, rhythm feel, mood, dancefloor setting |

### Multi-Platform Export
Rekordbox XML, Traktor NML, Serato ID3 (MP3/AIFF), Engine DJ SQLite, VirtualDJ XML.

---

## Prerequisites

### Required Files (Windows)

DJ Agent reads and writes these files. Make sure they exist before running:

| File | Size | Purpose |
|---|---|---|
| `%APPDATA%\Pioneer\rekordbox\master.db` | ~37 MB | Rekordbox database (tracks, playlists, tags, cues) |
| `%APPDATA%\Pioneer\rekordbox\share\PIONEER\USBANLZ\` | ~567 MB | Rekordbox phrase/waveform analysis (PSSI data for cue detection) |
| `%USERPROFILE%\.dj-agent\memory.json` | ~5 MB | Agent memory (processed tracks, corrections, calibration) — created automatically on first run |
| `config.yaml` (project root) | ~1 KB | Agent configuration (energy weights, cue thresholds, etc.) |
| Rekordbox XML export | ~3 MB | Used for cue/playlist sync (path configured in `config.yaml`) |

> **Close Rekordbox before running DJ Agent.** Rekordbox holds an exclusive lock on `master.db` while running — the agent cannot read or write the database until Rekordbox is closed. The agent checks for this automatically and will refuse to proceed if Rekordbox is detected.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/toimfortes/dj-agent.git
cd dj-agent
pip install -e ".[dev]"

# Launch GUI
python -m dj_agent

# Or use via Claude Code
claude
> magic
```

### Optional extras
```bash
pip install -e ".[all]"          # Everything (stems, mood, beats, reasoning, GUI)
pip install -e ".[stems]"        # Roformer stem separation
pip install -e ".[master]"       # Pedalboard mastering
pip install -e ".[mood]"         # Essentia + CLAP mood classification
pip install -e ".[beats]"        # Beat This! transformer beat tracking
pip install -e ".[reasoning]"    # Gemini AI reasoning
pip install -e ".[gui]"          # Gradio web UI
```

### Gemini AI Setup (optional)
Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:
```bash
export GOOGLE_API_KEY="your-key"
```

---

## Commands

| Say this | What it does |
|---|---|
| `magic` | Full pipeline — energy, cues, tags, cleanup, sync |
| `calculate energy` | LUFS-based energy ratings (1–10) |
| `calculate cues` | Detect intro, drop, breakdown, outro |
| `detect key` | Harmonic key detection with Camelot notation |
| `normalize` | Measure/normalize LUFS loudness |
| `master` | Platinum Notes-style mastering |
| `detect vocals` | Vocal/instrumental classification |
| `classify mood` | Mood/vibe tagging |
| `find duplicates` | File hash + fuzzy metadata matching |
| `check quality` | Fake FLAC, clipping, silence detection |
| `check beatgrid` | BPM verification, half/double fix |
| `harmonic mix` | Suggest compatible next tracks |
| `build set` | Optimize track ordering with energy arcs |
| `smart playlist [rule]` | Boolean rule-based playlist generation |
| `find mashups` | Mashup-compatible track finder |
| `separate stems` | Vocal/drum/bass/other separation |
| `shift key` | Pitch shift to target key |
| `analyze vibe` | AI-powered deep vibe analysis |
| `analytics` | Library distribution reports |
| `health` | Library health check |
| `export to traktor/serato/engine` | Multi-platform cue export |

All commands can be scoped: `calculate energy for Disco`, `cleanup Techno/Peak Time`.

---

## Architecture

```
src/dj_agent/           37 Python modules
├── Core:       energy, cues, cleanup, tags, sync, memory, config, types
├── Analysis:   keydetect, quality, duplicates, health, analytics
├── DJ Tools:   harmonic, transitions, setbuilder, mashups, smartlists, phrases
├── V2 Engines: stems, beatgrid, similarity, master, pitchshift
├── AI:         reasoning, mood, vocals, metadata
├── UI:         gui, gpu, export
└── 194 tests across 28 test files
```

V2 model hierarchy (auto-selects best available, graceful fallback):
- **Stems:** MelBand-Roformer → Demucs
- **Beats:** Beat This! → madmom → librosa
- **Phrases:** All-In-One → madmom → librosa
- **Similarity:** CLAP → librosa MFCC
- **Reasoning:** Flamingo → Gemini SDK → Ollama

---

## License

[MIT](LICENSE) — fork it, use it, make it yours.

---

Built with [Claude Code](https://docs.claude.com/en/docs/claude-code/overview). Audio analysis by [librosa](https://librosa.org/), [Essentia](https://essentia.upf.edu/), [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm). Rekordbox integration via [pyrekordbox](https://github.com/dylanljones/pyrekordbox). Mastering by [pedalboard](https://github.com/spotify/pedalboard). Stems by [audio-separator](https://github.com/nomadkaraoke/python-audio-separator).
