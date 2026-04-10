"""LUFS normalization — measure and optionally normalize audio files.

Adapted from AI-Music-Library-Normalization-Suite engineer.py.
Uses pyloudnorm for measurement and ffmpeg-normalize for file processing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .audio import measure_loudness
from .config import NormalizeConfig
from .types import LoudnessResult


# ---------------------------------------------------------------------------
# Measurement (pure Python, fast)
# ---------------------------------------------------------------------------

def measure_track(path: str | Path) -> LoudnessResult:
    """Measure LUFS, true peak, and loudness range for a single track."""
    return measure_loudness(path)


def measure_batch(paths: list[str | Path]) -> list[tuple[Path, LoudnessResult]]:
    """Measure LUFS for multiple tracks. Returns list of (path, result)."""
    results: list[tuple[Path, LoudnessResult]] = []
    for p in paths:
        p = Path(p)
        try:
            lr = measure_loudness(p)
            results.append((p, lr))
        except Exception as exc:
            results.append((p, LoudnessResult(
                integrated_lufs=float("-inf"), sample_peak_dbfs=float("-inf"),
                loudness_range_lu=0.0, short_term_max_lufs=float("-inf"),
            )))
    return results


# ---------------------------------------------------------------------------
# Normalization (requires ffmpeg-normalize)
# ---------------------------------------------------------------------------

_CODEC_MAP = {
    "mp3": "libmp3lame",
    "flac": "flac",
    "wav": "pcm_s16le",
    "aac": "aac",
    "ogg": "libvorbis",
    "aiff": "pcm_s16be",
}


def normalize_track(
    input_path: str | Path,
    output_path: str | Path,
    config: NormalizeConfig | None = None,
) -> dict[str, Any]:
    """Normalize a single track to the target LUFS.

    Requires ``ffmpeg-normalize`` to be installed (``pip install ffmpeg-normalize``).

    Returns a dict with original/normalized loudness and output path.
    """
    if config is None:
        from .config import get_config
        config = get_config().normalize

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Measure original
    original = measure_loudness(input_path)

    # Find ffmpeg-normalize
    exe = shutil.which("ffmpeg-normalize")
    if exe is None:
        raise FileNotFoundError(
            "ffmpeg-normalize not found. Install: pip install ffmpeg-normalize"
        )

    codec = _CODEC_MAP.get(config.output_format, "libmp3lame")

    cmd = [
        exe, str(input_path),
        "-o", str(output_path),
        "-c:a", codec,
        "--target-level", str(config.target_lufs),
        "--true-peak", str(config.true_peak_db),
    ]

    if config.output_format == "mp3":
        cmd.extend(["-b:a", config.output_bitrate])
    elif config.output_format == "flac":
        cmd.extend(["-e", "-compression_level", "8"])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    if not output_path.exists():
        raise RuntimeError(f"ffmpeg-normalize did not produce output: {output_path}")

    # Measure normalized
    normalized = measure_loudness(output_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_lufs": original.integrated_lufs,
        "normalized_lufs": normalized.integrated_lufs,
        "true_peak": normalized.sample_peak_dbfs,
        "gain_applied": normalized.integrated_lufs - original.integrated_lufs,
    }


# ---------------------------------------------------------------------------
# ReplayGain tag writing
# ---------------------------------------------------------------------------

def write_replaygain_tags(
    path: str | Path,
    track_gain_db: float,
    track_peak: float,
) -> None:
    """Write ReplayGain tags to an audio file using mutagen.

    Supports FLAC (Vorbis Comments) and MP3 (ID3 TXXX frames).
    """
    from mutagen import File
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3, TXXX

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".flac":
        audio = FLAC(str(path))
        audio["REPLAYGAIN_TRACK_GAIN"] = f"{track_gain_db:.2f} dB"
        audio["REPLAYGAIN_TRACK_PEAK"] = f"{track_peak:.6f}"
        audio.save()

    elif suffix == ".mp3":
        try:
            audio = ID3(str(path))
        except Exception:
            audio = ID3()
        audio.add(TXXX(encoding=3, desc="replaygain_track_gain",
                        text=[f"{track_gain_db:.2f} dB"]))
        audio.add(TXXX(encoding=3, desc="replaygain_track_peak",
                        text=[f"{track_peak:.6f}"]))
        audio.save(str(path))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_loudness_report(results: list[tuple[Path, LoudnessResult]],
                           target_lufs: float = -8.0) -> str:
    """Format LUFS measurements as a human-readable table."""
    lines = [
        f"{'Track':<50} {'LUFS':>7} {'Peak':>7} {'LRA':>5} {'Δ Target':>8}",
        "-" * 80,
    ]
    for path, lr in results:
        name = path.stem[:48]
        delta = lr.integrated_lufs - target_lufs
        sign = "+" if delta > 0 else ""
        lines.append(
            f"{name:<50} {lr.integrated_lufs:>7.1f} "
            f"{lr.sample_peak_dbfs:>7.1f} "
            f"{lr.loudness_range_lu:>5.1f} "
            f"{sign}{delta:>7.1f}"
        )
    return "\n".join(lines)
