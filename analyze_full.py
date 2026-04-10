#!/usr/bin/env python3
"""Full library analysis — energy, key, cues, cleanup, quality, vocals, Gemini vibe."""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ['GOOGLE_API_KEY'] = open(Path(__file__).parent / '.env').read().split('=',1)[1].split('#')[0].strip()

from dj_agent.audio import measure_loudness, load_audio
from dj_agent.energy import calculate_energy
from dj_agent.keydetect import detect_key, detect_tuning
from dj_agent.cues import detect_cue_points
from dj_agent.cleanup import cleanup_title, split_artist_from_title
from dj_agent.quality import check_audio_quality
from dj_agent.vocals import detect_vocals_fast
from dj_agent.reasoning import analyze_vibe, classify_nuance
import librosa, numpy as np

LIBRARY = Path("/home/antoniofortes/Documents/MusicLibrary/Contents")
OUTPUT = Path(__file__).parent / "library_analysis.json"
BATCH = 50  # progress every N tracks
VIBE_EVERY = 25  # Gemini vibe every Nth track

def analyze(path):
    r = {"path": str(path), "file": path.name, "errors": []}
    try:
        loud = measure_loudness(path)
        r["lufs"] = round(loud.integrated_lufs, 1)
        r["peak"] = round(loud.sample_peak_dbfs, 1)
        r["lra"] = round(loud.loudness_range_lu, 1)
    except Exception as e:
        r["errors"].append(f"lufs:{e}")
        r["lufs"] = -100

    try:
        y, sr = load_audio(path, sr=22050, mono=True)
        dur = librosa.get_duration(y=y, sr=sr)
        bpm = float(np.asarray(librosa.beat.beat_track(y=y, sr=sr)[0]).item())
        r["duration"] = round(dur, 1)
        r["bpm"] = round(bpm, 1)
        energy = calculate_energy(y, sr, bpm=bpm, loudness_lufs=r.get("lufs", -20))
        r["energy"] = energy.calibrated_score
        cues = detect_cue_points(y, sr, bpm=bpm, duration=dur)
        r["cues"] = [{"name": c.name, "pos_ms": c.position_ms, "colour": c.colour} for c in cues]
    except Exception as e:
        r["errors"].append(f"audio:{e}")

    try:
        key = detect_key(path)
        r["key"] = key.key
        r["camelot"] = key.camelot
        r["key_conf"] = round(key.confidence, 2)
    except Exception as e:
        r["errors"].append(f"key:{e}")

    try:
        r["tuning"] = round(detect_tuning(path), 3)
    except: pass

    try:
        cleaned, changes = cleanup_title(path.stem)
        artist, title = split_artist_from_title(cleaned)
        r["artist"] = artist or ""
        r["title"] = title
        r["changes"] = changes
    except Exception as e:
        r["errors"].append(f"cleanup:{e}")

    try:
        q = check_audio_quality(path)
        r["format"] = q.format
        r["bitrate"] = q.bitrate
        r["warnings"] = q.warnings
        r["fake_lossless"] = q.is_fake_lossless
    except Exception as e:
        r["errors"].append(f"quality:{e}")

    try:
        v = detect_vocals_fast(path)
        r["vocal"] = v.classification
        r["vocal_prob"] = round(v.vocal_probability, 2)
    except Exception as e:
        r["errors"].append(f"vocal:{e}")

    return r

def main():
    tracks = sorted(LIBRARY.rglob("*.mp3")) + sorted(LIBRARY.rglob("*.wav")) + sorted(LIBRARY.rglob("*.aiff"))
    total = len(tracks)
    print(f"Analyzing {total} tracks...", flush=True)
    
    results = []
    errs = 0
    t0 = time.time()

    for i, t in enumerate(tracks):
        try:
            r = analyze(t)
            
            # Gemini vibe every Nth track
            if (i + 1) % VIBE_EVERY == 1:
                try:
                    r["vibe"] = analyze_vibe(t, backend="gemini")[:200]
                    r["nuance"] = classify_nuance(t, backend="gemini")
                except: pass
            
            results.append(r)
            if r["errors"]: errs += 1
        except Exception as e:
            results.append({"path": str(t), "file": t.name, "errors": [str(e)]})
            errs += 1

        if (i + 1) % BATCH == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {rate:.1f}/s ETA:{eta/60:.0f}m err:{errs}", flush=True)
            # Save incrementally
            OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    elapsed = time.time() - t0
    print(f"\nDone! {total} in {elapsed/60:.1f}m ({elapsed/total:.1f}s/track) {errs} errors", flush=True)
    print(f"Saved: {OUTPUT}", flush=True)

    # Summary
    from collections import Counter
    energies = [r["energy"] for r in results if "energy" in r]
    keys = [r.get("camelot","") for r in results if r.get("camelot")]
    vocals = Counter(r.get("vocal","") for r in results if r.get("vocal"))
    
    print(f"\nEnergy: {min(energies)}-{max(energies)} avg={sum(energies)/len(energies):.1f}")
    print(f"Keys: {len(keys)}/{total}")
    for k,c in Counter(keys).most_common(10): print(f"  {k}: {c}")
    print(f"Vocals: {dict(vocals)}")

if __name__ == "__main__":
    main()
