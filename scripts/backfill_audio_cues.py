"""Backfill vocal re-entry cues and per-segment energy for all tracks.

Optimized: does HPSS once per track and windows the harmonic signal,
instead of per-window HPSS (~15x faster than calling detect_vocal_entries).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dj_agent.cues import CuePoint
from dj_agent.pipeline import _compute_segment_energies
from dj_agent.config import get_config
from dj_agent.memory import load_memory, save_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)

BATCH_SIZE = 50
SR = 22050


def detect_vocal_entries_fast(
    y: np.ndarray, sr: int, duration: float, bpm: float,
) -> list[dict]:
    """Fast vocal re-entry detection — single HPSS pass, windowed ratio."""
    if bpm <= 0 or duration <= 0:
        return []

    beat_sec = 60.0 / bpm
    min_gap_sec = 4 * 4 * beat_sec  # 4 bars

    # Single HPSS for the whole track
    y_harm, _ = librosa.effects.hpss(y, margin=1.0)

    # STFT of harmonic component
    n_fft = 2048
    hop = 512
    stft_h = np.abs(librosa.stft(y_harm, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    vocal_mask = (freqs >= 300) & (freqs <= 4000)

    # Frame-level vocal ratio
    vocal_band = stft_h[vocal_mask].mean(axis=0)
    total_harm = stft_h.mean(axis=0) + 1e-8
    frame_vocal = vocal_band / total_harm

    # Convert to time-based windows (~2 beats)
    win_sec = max(2.0, 2 * beat_sec)
    hop_sec = max(1.0, beat_sec)
    frame_dur = hop / sr

    # Threshold from median + 15% of range (same as original)
    sorted_v = np.sort(frame_vocal)
    median_v = sorted_v[len(sorted_v) // 2]
    max_v = sorted_v[-1] + 1e-8
    vocal_threshold = median_v + 0.15 * (max_v - median_v)

    # Window the frame-level ratios
    windows: list[tuple[float, bool]] = []
    t = 0.0
    while t + win_sec <= duration:
        f_start = int(t / frame_dur)
        f_end = int((t + win_sec) / frame_dur)
        f_end = min(f_end, len(frame_vocal))
        if f_end > f_start:
            win_mean = float(frame_vocal[f_start:f_end].mean())
            windows.append((t, win_mean > vocal_threshold))
        t += hop_sec

    if not windows:
        return []

    # Walk windows, emit cue on vocal return after gap
    cues = []
    last_vocal_end = -1e9
    currently_vocal = False
    for t_start, has_vocal in windows:
        if has_vocal:
            if not currently_vocal:
                gap = t_start - last_vocal_end
                if gap >= min_gap_sec and t_start > 0.5:
                    cues.append({
                        "name": "Vocal",
                        "position_ms": int(t_start * 1000),
                        "colour": "green",
                        "confidence": 0.8,
                        "memory_only": True,
                    })
            currently_vocal = True
            last_vocal_end = t_start + win_sec
        else:
            currently_vocal = False

    return cues


def needs_work(track: dict) -> tuple[bool, bool]:
    """Return (needs_seg_energy, needs_vocal)."""
    cues = track.get("cues", [])
    if not cues:
        return False, False
    has_seg_energy = any(c.get("segment_energy") is not None for c in cues)
    vc = track.get("vocal_classification", "")
    has_vocal_cue = any(c.get("name") == "Vocal" for c in cues)
    needs_vocal = vc in ("vocal", "partial_vocal") and not has_vocal_cue
    return not has_seg_energy, needs_vocal


def process_track(track: dict) -> tuple[int, int]:
    """Run vocal re-entry + segment energy. Returns (vocal_added, seg_updated)."""
    path = Path(track["path"])
    if not path.exists():
        return 0, 0

    cues = track.get("cues", [])
    bpm = track.get("bpm", 0)
    duration = track.get("duration_sec", 0)
    lufs = track.get("lufs_integrated", -20.0)
    need_seg, need_vocal = needs_work(track)

    if bpm <= 0 or duration <= 0:
        return 0, 0
    if not need_seg and not need_vocal:
        return 0, 0

    try:
        y, sr = librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        return 0, 0

    vocal_added = 0
    seg_updated = 0

    # Vocal re-entry (fast path)
    if need_vocal:
        try:
            vocal_cues = detect_vocal_entries_fast(y, sr, duration, bpm)
            cues.extend(vocal_cues)
            vocal_added = len(vocal_cues)
        except Exception:
            pass

    # Per-segment energy
    if need_seg:
        try:
            _compute_segment_energies(cues, y, sr, bpm, loudness_lufs=lufs)
            seg_updated = sum(1 for c in cues if c.get("segment_energy") is not None)
        except Exception:
            pass

    track["cues"] = cues
    return vocal_added, seg_updated


def main():
    cfg = get_config()
    memory = load_memory(cfg.memory)
    tracks = memory.get("processed_tracks", {})

    work_list = [
        (h, t) for h, t in tracks.items()
        if any(needs_work(t))
    ]
    _log.info("Tracks needing work: %d / %d", len(work_list), len(tracks))

    if not work_list:
        _log.info("Nothing to do.")
        return

    total_vocal = 0
    total_seg = 0
    done = 0
    t0 = time.time()

    for content_hash, track in work_list:
        v, s = process_track(track)
        total_vocal += v
        total_seg += s
        done += 1

        if done % 10 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(work_list) - done) / rate if rate > 0 else 0
            _log.info(
                "[%d/%d] +%d vocal cues, %d segments energised | %.1f tracks/min | ETA %.0fm",
                done, len(work_list), total_vocal, total_seg,
                rate * 60, eta / 60,
            )

        if done % BATCH_SIZE == 0:
            save_memory(memory, cfg.memory)
            _log.info("Saved memory (batch checkpoint)")

    save_memory(memory, cfg.memory)
    elapsed = time.time() - t0
    _log.info(
        "Done. %d tracks in %.1f min. +%d vocal cues, %d segments energised.",
        done, elapsed / 60, total_vocal, total_seg,
    )


if __name__ == "__main__":
    main()
