"""Analyze the full music library — energy, key, cues, cleanup, quality."""

import json
import time
import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dj_agent.audio import measure_loudness, load_audio
from dj_agent.energy import calculate_energy
from dj_agent.keydetect import detect_key, detect_tuning
from dj_agent.cues import detect_cue_points
from dj_agent.cleanup import cleanup_title, split_artist_from_title
from dj_agent.quality import check_audio_quality
from dj_agent.config import EnergyConfig
import librosa
import numpy as np

LIBRARY = Path("/home/antoniofortes/Documents/MusicLibrary/Contents")
OUTPUT = Path("/home/antoniofortes/Projects/dj_agent/library_analysis.json")

def analyze_track(path: Path) -> dict:
    """Full analysis pipeline for one track."""
    result = {"path": str(path), "filename": path.name, "errors": []}
    
    try:
        # LUFS
        loud = measure_loudness(path)
        result["lufs"] = round(loud.integrated_lufs, 1)
        result["peak_dbfs"] = round(loud.sample_peak_dbfs, 1)
        result["lra"] = round(loud.loudness_range_lu, 1)
    except Exception as e:
        result["errors"].append(f"LUFS: {e}")
        result["lufs"] = -100
    
    try:
        # Load audio once for multiple analyses
        y, sr = load_audio(path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        result["duration"] = round(duration, 1)
        
        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.asarray(tempo).item())
        result["bpm"] = round(bpm, 1)
        
        # Energy
        energy = calculate_energy(y, sr, bpm=bpm, loudness_lufs=result.get("lufs", -20))
        result["energy"] = energy.calibrated_score
        
        # Cues
        cues = detect_cue_points(y, sr, bpm=bpm, duration=duration)
        result["cues"] = [{"name": c.name, "position_ms": c.position_ms, "colour": c.colour} for c in cues]
    except Exception as e:
        result["errors"].append(f"Audio: {e}")
    
    try:
        # Key detection
        key = detect_key(path)
        result["key"] = key.key
        result["camelot"] = key.camelot
        result["key_confidence"] = round(key.confidence, 2)
    except Exception as e:
        result["errors"].append(f"Key: {e}")
    
    try:
        # Tuning
        tuning = detect_tuning(path)
        result["tuning_offset"] = round(tuning, 3)
    except Exception as e:
        result["errors"].append(f"Tuning: {e}")
    
    try:
        # Cleanup
        cleaned, changes = cleanup_title(path.stem)
        artist, title = split_artist_from_title(cleaned)
        result["cleaned_title"] = cleaned
        result["artist"] = artist or ""
        result["title"] = title
        result["cleanup_changes"] = changes
    except Exception as e:
        result["errors"].append(f"Cleanup: {e}")
    
    try:
        # Quality
        q = check_audio_quality(path)
        result["format"] = q.format
        result["bitrate"] = q.bitrate
        result["quality_warnings"] = q.warnings
        result["is_fake_lossless"] = q.is_fake_lossless
    except Exception as e:
        result["errors"].append(f"Quality: {e}")
    
    return result


def main():
    tracks = sorted(LIBRARY.rglob("*.mp3")) + sorted(LIBRARY.rglob("*.wav")) + sorted(LIBRARY.rglob("*.aiff"))
    total = len(tracks)
    print(f"Analyzing {total} tracks...")
    
    results = []
    errors = 0
    start = time.time()
    
    for i, t in enumerate(tracks):
        try:
            r = analyze_track(t)
            results.append(r)
            if r["errors"]:
                errors += 1
        except Exception as e:
            results.append({"path": str(t), "filename": t.name, "errors": [str(e)]})
            errors += 1
        
        # Progress every 50 tracks
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {rate:.1f} tracks/s, ETA {eta/60:.0f}min, {errors} errors")
    
    # Save results
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    
    elapsed = time.time() - start
    print(f"\nDone! {total} tracks in {elapsed/60:.1f} minutes ({elapsed/total:.1f}s/track)")
    print(f"Errors: {errors}/{total}")
    print(f"Results saved to: {OUTPUT}")
    
    # Quick stats
    energies = [r["energy"] for r in results if "energy" in r]
    keys = [r.get("camelot", "") for r in results if r.get("camelot")]
    lufs_vals = [r["lufs"] for r in results if r.get("lufs", -100) > -100]
    
    print(f"\n=== Library Stats ===")
    print(f"Energy: min={min(energies)}, max={max(energies)}, avg={sum(energies)/len(energies):.1f}")
    print(f"LUFS: min={min(lufs_vals):.1f}, max={max(lufs_vals):.1f}, avg={sum(lufs_vals)/len(lufs_vals):.1f}")
    print(f"Keys detected: {len(keys)}/{total}")
    
    from collections import Counter
    top_keys = Counter(keys).most_common(10)
    print(f"Top keys: {', '.join(f'{k}({c})' for k,c in top_keys)}")


if __name__ == "__main__":
    main()
