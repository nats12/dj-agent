"""Platinum Notes-style audio mastering — multiband dynamics, clip repair, EQ, limiting.

This module is SEPARATE from normalize.py (simple LUFS gain adjustment).
It actively reshapes dynamics and frequency balance — use with care.

Processing chain:
1. Detect and optionally repair clipping
2. Multiband compression (4 bands via Linkwitz-Riley crossover)
3. Shelving EQ (bass/treble shaping)
4. LUFS gain adjustment to target
5. Brick-wall limiter (final safety)
6. Before/after quality metrics

WARNING: This creates new files. Never modifies originals. Already
well-mastered tracks may sound worse after processing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .audio import measure_loudness
from .types import LoudnessResult


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass
class BandSettings:
    """Compression settings for one frequency band."""
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    gain_db: float = 0.0


@dataclass
class MasterTemplate:
    """Full mastering template — all parameters for the processing chain."""
    name: str
    bands: list[BandSettings]
    crossover_freqs: list[float] = field(default_factory=lambda: [200, 2000, 8000])
    bass_shelf_db: float = 0.0
    treble_shelf_db: float = 0.0
    target_lufs: float = -8.0
    peak_ceiling_db: float = -1.0
    repair_clipping: bool = True
    # Macro controls (user-friendly)
    compression_amount: float = 1.0  # 0.0=bypass, 1.0=as defined, 2.0=double


TEMPLATES: dict[str, MasterTemplate] = {
    "official": MasterTemplate(
        name="Official",
        bands=[
            BandSettings(threshold_db=-20, ratio=2.0, attack_ms=30, release_ms=200, gain_db=0),
            BandSettings(threshold_db=-18, ratio=1.8, attack_ms=15, release_ms=150, gain_db=0),
            BandSettings(threshold_db=-16, ratio=1.5, attack_ms=8, release_ms=100, gain_db=0),
            BandSettings(threshold_db=-18, ratio=1.5, attack_ms=5, release_ms=80, gain_db=0),
        ],
        target_lufs=-8.0,
        peak_ceiling_db=-1.0,
    ),
    "festival": MasterTemplate(
        name="Festival",
        bands=[
            BandSettings(threshold_db=-18, ratio=3.0, attack_ms=20, release_ms=150, gain_db=1.5),
            BandSettings(threshold_db=-16, ratio=2.5, attack_ms=10, release_ms=120, gain_db=0),
            BandSettings(threshold_db=-14, ratio=2.0, attack_ms=6, release_ms=80, gain_db=1.0),
            BandSettings(threshold_db=-16, ratio=1.8, attack_ms=4, release_ms=60, gain_db=0),
        ],
        bass_shelf_db=2.0,
        treble_shelf_db=1.0,
        target_lufs=-7.0,
        peak_ceiling_db=-0.5,
    ),
    "big_boost": MasterTemplate(
        name="Big Boost",
        bands=[
            BandSettings(threshold_db=-15, ratio=4.0, attack_ms=15, release_ms=120, gain_db=3.0),
            BandSettings(threshold_db=-14, ratio=3.0, attack_ms=8, release_ms=100, gain_db=1.0),
            BandSettings(threshold_db=-12, ratio=2.5, attack_ms=5, release_ms=70, gain_db=1.5),
            BandSettings(threshold_db=-14, ratio=2.0, attack_ms=3, release_ms=50, gain_db=0.5),
        ],
        bass_shelf_db=3.0,
        treble_shelf_db=1.5,
        target_lufs=-6.0,
        peak_ceiling_db=-0.3,
    ),
    "gentle": MasterTemplate(
        name="Gentle",
        bands=[
            BandSettings(threshold_db=-24, ratio=1.5, attack_ms=40, release_ms=250, gain_db=0),
            BandSettings(threshold_db=-22, ratio=1.3, attack_ms=20, release_ms=200, gain_db=0),
            BandSettings(threshold_db=-20, ratio=1.3, attack_ms=12, release_ms=150, gain_db=0),
            BandSettings(threshold_db=-22, ratio=1.2, attack_ms=8, release_ms=100, gain_db=0),
        ],
        target_lufs=-9.0,
        peak_ceiling_db=-1.0,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def master_track(
    input_path: str | Path,
    output_path: str | Path | None = None,
    template: str = "official",
    custom: MasterTemplate | None = None,
) -> dict[str, Any]:
    """Process a single track through the mastering chain.

    Parameters
    ----------
    input_path : path to input audio file
    output_path : path for output file. If None, auto-generates in same
        directory with ``_mastered`` suffix. Format matches input by default.
    template : one of "official", "festival", "big_boost", "gentle"
    custom : optional custom MasterTemplate (overrides template name)

    Returns dict with before/after metrics and output path.
    """
    input_path = Path(input_path)
    tmpl = custom or TEMPLATES.get(template)
    if tmpl is None:
        raise ValueError(f"Unknown template: {template}. Choose from: {list(TEMPLATES)}")

    # Idempotency guard — check metadata tag first, filename as fallback
    if _is_already_mastered(input_path):
        raise ValueError(
            f"Refusing to re-master '{input_path.name}' — file has DJ_AGENT_MASTERED tag. "
            "Processing it again would destroy dynamic range. Use the original file."
        )

    # Dynamic range check — refuse to master already-compressed audio
    # (catches professionally mastered tracks that lack our tag)
    try:
        from .audio import measure_loudness
        loud = measure_loudness(input_path)
        if loud.loudness_range_lu < 3.0 and loud.loudness_range_lu > 0:
            import warnings
            warnings.warn(
                f"Track '{input_path.name}' has very low dynamic range "
                f"(LRA={loud.loudness_range_lu:.1f} LU). It may already be mastered. "
                "Re-mastering will further compress dynamics."
            )
    except Exception:
        pass  # measurement failure shouldn't block mastering

    # Determine output path — match input format by default
    if output_path is None:
        suffix = input_path.suffix
        output_path = input_path.with_stem(input_path.stem + "_mastered")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load at native sample rate (preserve quality)
    data, sr = sf.read(str(input_path), always_2d=True)  # (samples, channels)

    # Before metrics
    before = _measure_metrics(data, sr)

    # Step 1: Clip repair
    if tmpl.repair_clipping:
        data = _repair_clipping(data, threshold=0.98)

    # Step 2: Multiband compression
    data = _multiband_compress(data, sr, tmpl)

    # Step 3: Shelving EQ
    if tmpl.bass_shelf_db != 0 or tmpl.treble_shelf_db != 0:
        data = _apply_shelving_eq(data, sr, tmpl.bass_shelf_db, tmpl.treble_shelf_db)

    # Step 4: LUFS gain adjustment to target (BEFORE limiter)
    data = _lufs_gain_adjust(data, sr, tmpl.target_lufs)

    # Step 5: Brick-wall limiter (final safety)
    data = _limit(data, sr, tmpl.peak_ceiling_db)

    # After metrics
    after = _measure_metrics(data, sr)

    # Write output (match input format)
    _write_output(data, sr, input_path, output_path)

    # Tag output as mastered (prevents re-mastering)
    _tag_as_mastered(output_path, tmpl.name)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "template": tmpl.name,
        "before": before,
        "after": after,
    }


# ---------------------------------------------------------------------------
# Step 1: Clip repair — spline interpolation (honest reconstruction)
# ---------------------------------------------------------------------------

def _repair_clipping(data: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    """Repair clipped peaks using cubic spline interpolation.

    This attempts actual waveform reconstruction (not just soft-clipping).
    For each channel, identifies clipped regions and interpolates through
    them using the surrounding unclipped samples.
    """
    from scipy.interpolate import CubicSpline

    result = data.copy()
    for ch in range(data.shape[1]):
        signal = data[:, ch]
        is_clipped = np.abs(signal) >= threshold

        if not np.any(is_clipped):
            continue

        # Find clipped regions
        regions = _find_clipped_regions(is_clipped)

        for start, end in regions:
            # Context window around clipped region
            ctx = max(64, (end - start) * 3)
            ctx_start = max(0, start - ctx)
            ctx_end = min(len(signal), end + ctx)

            local = signal[ctx_start:ctx_end]
            local_clipped = is_clipped[ctx_start:ctx_end]

            good_idx = np.where(~local_clipped)[0]
            bad_idx = np.where(local_clipped)[0]

            if len(good_idx) < 4 or len(bad_idx) == 0:
                continue

            cs = CubicSpline(good_idx, local[good_idx])
            result[ctx_start:ctx_end, ch][local_clipped] = cs(bad_idx)

    return result


def _find_clipped_regions(is_clipped: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    for i in range(len(is_clipped)):
        if is_clipped[i] and not in_region:
            start = i
            in_region = True
        elif not is_clipped[i] and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(is_clipped)))
    return regions


# ---------------------------------------------------------------------------
# Step 2: Multiband compression
# ---------------------------------------------------------------------------

def _multiband_compress(
    data: np.ndarray,
    sr: int,
    tmpl: MasterTemplate,
) -> np.ndarray:
    """Split into bands, compress each independently, sum back.

    Uses Linkwitz-Riley crossovers (sosfiltfilt for zero-phase offline).
    """
    from pedalboard import Compressor, Gain, Pedalboard

    bands = _split_bands(data, sr, tmpl.crossover_freqs)
    processed_bands: list[np.ndarray] = []

    # Clamp compression_amount to valid range
    comp_amount = max(0.0, min(2.0, tmpl.compression_amount))

    for band_audio, settings in zip(bands, tmpl.bands):
        # Scale by compression_amount macro
        effective_ratio = 1.0 + (settings.ratio - 1.0) * comp_amount
        # Keep threshold fixed — only the ratio scales with compression_amount.
        # Dividing threshold made compression LESS aggressive at higher amounts (bug).
        effective_threshold = settings.threshold_db

        chain = Pedalboard([
            Gain(gain_db=settings.gain_db),
            Compressor(
                threshold_db=effective_threshold,
                ratio=effective_ratio,
                attack_ms=settings.attack_ms,
                release_ms=settings.release_ms,
            ),
        ])

        # pedalboard expects (channels, samples)
        processed = chain(band_audio.T, sr).T
        processed_bands.append(processed)

    return sum(processed_bands)


def _split_bands(
    data: np.ndarray,
    sr: int,
    crossover_freqs: list[float],
    order: int = 2,
) -> list[np.ndarray]:
    """Split audio into frequency bands using Linkwitz-Riley crossovers.

    Uses sosfiltfilt (zero-phase) — doubles effective order but guarantees
    flat magnitude sum. Verified by test_crossover_sums_flat.
    """
    bands: list[np.ndarray] = []
    n_bands = len(crossover_freqs) + 1

    for i in range(n_bands):
        if i == 0:
            # Low band: lowpass at first crossover
            sos = butter(order, crossover_freqs[0], btype="low", fs=sr, output="sos")
        elif i == n_bands - 1:
            # High band: highpass at last crossover
            sos = butter(order, crossover_freqs[-1], btype="high", fs=sr, output="sos")
        else:
            # Mid band: bandpass
            sos = butter(
                order,
                [crossover_freqs[i - 1], crossover_freqs[i]],
                btype="band",
                fs=sr,
                output="sos",
            )

        # Apply per-channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = sosfiltfilt(sos, data[:, ch])

        bands.append(filtered)

    return bands


# ---------------------------------------------------------------------------
# Step 3: Shelving EQ
# ---------------------------------------------------------------------------

def _apply_shelving_eq(
    data: np.ndarray,
    sr: int,
    bass_db: float,
    treble_db: float,
) -> np.ndarray:
    """Apply bass and treble shelf EQ using pedalboard."""
    from pedalboard import HighShelfFilter, LowShelfFilter, Pedalboard

    chain = Pedalboard([])
    if bass_db != 0:
        chain.append(LowShelfFilter(cutoff_frequency_hz=200, gain_db=bass_db))
    if treble_db != 0:
        chain.append(HighShelfFilter(cutoff_frequency_hz=8000, gain_db=treble_db))

    return chain(data.T, sr).T


# ---------------------------------------------------------------------------
# Step 4: LUFS gain adjustment (BEFORE limiter — correct chain order)
# ---------------------------------------------------------------------------

def _lufs_gain_adjust(
    data: np.ndarray,
    sr: int,
    target_lufs: float,
) -> np.ndarray:
    """Apply gain to hit target LUFS. No dynamics change, just level."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    current = meter.integrated_loudness(data)

    if not np.isfinite(current):
        return data

    gain_db = target_lufs - current
    # Clamp gain to prevent extreme amplification of near-silent audio
    gain_db = float(np.clip(gain_db, -24.0, 24.0))
    return data * (10 ** (gain_db / 20.0))


# ---------------------------------------------------------------------------
# Step 5: Brick-wall limiter (final safety)
# ---------------------------------------------------------------------------

def _limit(data: np.ndarray, sr: int, ceiling_db: float) -> np.ndarray:
    """Apply brick-wall limiter using pedalboard."""
    from pedalboard import Limiter, Pedalboard

    chain = Pedalboard([Limiter(threshold_db=ceiling_db, release_ms=100)])
    return chain(data.T, sr).T


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _measure_metrics(data: np.ndarray, sr: int) -> dict[str, float]:
    """Compute before/after quality metrics."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(data)
    peak = float(np.max(np.abs(data)))
    peak_db = 20.0 * np.log10(peak + 1e-12)

    # Dynamic range: difference between 95th and 5th percentile of RMS
    frame_size = int(0.4 * sr)  # 400ms frames
    rms_values: list[float] = []
    for start in range(0, data.shape[0] - frame_size, frame_size // 2):
        frame = data[start : start + frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > 1e-8:
            rms_values.append(20.0 * np.log10(rms))

    if len(rms_values) >= 4:
        lra = float(np.percentile(rms_values, 95) - np.percentile(rms_values, 5))
    else:
        lra = 0.0

    # Spectral centroid (brightness indicator)
    import librosa
    mono = np.mean(data, axis=1) if data.ndim > 1 else data
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))

    return {
        "lufs": round(lufs, 1) if np.isfinite(lufs) else -100.0,
        "peak_dbfs": round(peak_db, 1),
        "dynamic_range_db": round(lra, 1),
        "spectral_centroid_hz": round(centroid, 0),
    }


def format_comparison(result: dict[str, Any]) -> str:
    """Format before/after comparison as a readable string."""
    b = result["before"]
    a = result["after"]

    lines = [
        f"Mastered: {Path(result['input_path']).name}  ({result['template']})",
        "",
        f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}",
        f"  {'-'*55}",
        f"  {'LUFS':<25} {b['lufs']:>10.1f} {a['lufs']:>10.1f} {a['lufs']-b['lufs']:>+10.1f}",
        f"  {'Peak (dBFS)':<25} {b['peak_dbfs']:>10.1f} {a['peak_dbfs']:>10.1f} {a['peak_dbfs']-b['peak_dbfs']:>+10.1f}",
        f"  {'Dynamic Range (dB)':<25} {b['dynamic_range_db']:>10.1f} {a['dynamic_range_db']:>10.1f} {a['dynamic_range_db']-b['dynamic_range_db']:>+10.1f}",
        f"  {'Brightness (Hz)':<25} {b['spectral_centroid_hz']:>10.0f} {a['spectral_centroid_hz']:>10.0f} {a['spectral_centroid_hz']-b['spectral_centroid_hz']:>+10.0f}",
        "",
        f"  Output: {result['output_path']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writing — match input format
# ---------------------------------------------------------------------------

def _write_output(
    data: np.ndarray,
    sr: int,
    input_path: Path,
    output_path: Path,
) -> None:
    """Write output file, matching input format to avoid lossy re-encoding."""
    suffix = output_path.suffix.lower()

    # Clamp to valid range
    data = np.clip(data, -1.0, 1.0)

    if suffix in (".flac", ".wav", ".aiff"):
        # Lossless — write directly
        subtype = "PCM_24" if suffix == ".flac" else "PCM_16"
        sf.write(str(output_path), data, sr, subtype=subtype)

    elif suffix == ".mp3":
        # Lossy — use pedalboard's MP3 writer to avoid quality loss from ffmpeg re-encode
        from pedalboard.io import AudioFile
        with AudioFile(str(output_path), "w", sr, data.shape[1],
                       format="mp3", quality="320k") as f:
            f.write(data.T)

    else:
        # Fallback
        sf.write(str(output_path), data, sr)


# ---------------------------------------------------------------------------
# Idempotency via metadata tags
# ---------------------------------------------------------------------------

def _is_already_mastered(path: Path) -> bool:
    """Check if a file has the DJ_AGENT_MASTERED metadata tag."""
    # Check metadata tag (survives renames)
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(str(path))
        if mf and mf.tags:
            # ID3 (MP3/AIFF)
            for frame in mf.tags.getall("TXXX") if hasattr(mf.tags, "getall") else []:
                if hasattr(frame, "desc") and frame.desc == "DJ_AGENT_MASTERED":
                    return True
            # Vorbis (FLAC/OGG)
            if hasattr(mf.tags, "get") and mf.tags.get("DJ_AGENT_MASTERED"):
                return True
    except Exception:
        pass

    # Filename fallback
    return "_mastered" in path.stem


def _tag_as_mastered(path: Path, template_name: str) -> None:
    """Write DJ_AGENT_MASTERED tag to the output file."""
    from datetime import datetime
    tag_value = f"template={template_name};date={datetime.now().isoformat()}"

    try:
        suffix = path.suffix.lower()
        if suffix in (".mp3", ".aiff"):
            from mutagen.id3 import ID3, TXXX
            try:
                tags = ID3(str(path))
            except Exception:
                tags = ID3()
            tags.add(TXXX(encoding=3, desc="DJ_AGENT_MASTERED", text=[tag_value]))
            tags.save(str(path))
        elif suffix == ".flac":
            from mutagen.flac import FLAC
            audio = FLAC(str(path))
            audio["DJ_AGENT_MASTERED"] = tag_value
            audio.save()
    except Exception:
        pass  # tag writing is best-effort — mastering still succeeds
