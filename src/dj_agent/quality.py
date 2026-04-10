"""Audio quality validation — fake FLAC detection, clipping, silence."""

from __future__ import annotations
import logging

import subprocess
import json
from pathlib import Path

import numpy as np

from .types import QualityReport, SilenceRegion


# ---------------------------------------------------------------------------
# Fake lossless detection
# ---------------------------------------------------------------------------

def detect_fake_lossless(path: str | Path) -> tuple[bool, float]:
    """Detect transcoded files by checking spectral frequency cutoff.

    Returns ``(is_fake, confidence)`` where confidence is 0-1.

    Uses flac-detective if installed, otherwise falls back to
    librosa spectral rolloff analysis.
    """
    path = Path(path)

    # Try flac-detective first
    try:
        from flac_detective import FLACAnalyzer  # type: ignore[import-untyped]
        analyzer = FLACAnalyzer()
        result = analyzer.analyze_file(str(path))
        score = result.get("score", 0)
        is_fake = score > 60
        return is_fake, min(score / 100.0, 1.0)
    except ImportError:
        pass

    # Fallback: spectral rolloff analysis
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        nyquist = sr / 2.0
        if nyquist <= 20000:
            return False, 0.0  # low sample rate — can't assess

        # Look for a "brick wall" frequency cutoff (sharp dropoff)
        # characteristic of lossy encoding, not just low average energy.
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        # Average spectrum energy per frequency bin
        mean_spectrum = np.mean(S, axis=1)
        if mean_spectrum.max() == 0:
            return False, 0.0

        # Normalize
        mean_spectrum = mean_spectrum / mean_spectrum.max()

        # Find the frequency where energy drops below 1% of peak
        # (brick wall = sharp cutoff, not gradual rolloff)
        above_threshold = freqs[mean_spectrum > 0.01]
        if len(above_threshold) == 0:
            return False, 0.0

        cutoff_freq = float(above_threshold[-1])

        # Known lossy cutoffs (brick wall style)
        if cutoff_freq < 16500:
            return True, 0.85  # likely 128kbps transcode
        elif cutoff_freq < 19000:
            return True, 0.5   # possibly 192-256kbps transcode
        else:
            return False, 0.1

    except Exception:
        return False, 0.0


# ---------------------------------------------------------------------------
# Clipping detection
# ---------------------------------------------------------------------------

def detect_clipping(path: str | Path, threshold: float = 0.99,
                    chunk_frames: int = 65536) -> int:
    """Count clipping events (3+ consecutive samples at max amplitude).

    Reads in chunks to avoid loading entire files into memory.
    Returns the number of clipping events detected.
    """
    import soundfile as sf

    clip_count = 0
    run_length = 0  # carry across chunk boundaries

    with sf.SoundFile(str(path)) as f:
        while f.tell() < f.frames:
            data = f.read(chunk_frames, dtype="float32")
            if len(data) == 0:
                break

            # Collapse stereo to max absolute per sample
            if data.ndim > 1:
                data = np.max(np.abs(data), axis=1)
            else:
                data = np.abs(data)

            for sample in data:
                if sample > threshold:
                    run_length += 1
                else:
                    if run_length >= 3:
                        clip_count += 1
                    run_length = 0

    # Check final run
    if run_length >= 3:
        clip_count += 1

    return clip_count


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def detect_silence(
    path: str | Path,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 1000,
) -> list[SilenceRegion]:
    """Detect silent regions using pydub.

    Returns a list of :class:`SilenceRegion` for silence longer than
    *min_silence_ms* milliseconds.
    """
    from pydub import AudioSegment
    from pydub.silence import detect_silence as _detect_silence

    audio = AudioSegment.from_file(str(path))
    silent_ranges = _detect_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,  # 10ms steps (faster than default 1ms)
    )

    return [SilenceRegion(start_ms=s, end_ms=e) for s, e in silent_ranges]


def get_leading_trailing_silence(
    regions: list[SilenceRegion],
    track_duration_ms: int,
) -> tuple[int, int]:
    """Extract leading and trailing silence from a list of silence regions.

    Returns ``(leading_ms, trailing_ms)``.
    """
    leading = 0
    trailing = 0

    if regions and regions[0].start_ms == 0:
        leading = regions[0].duration_ms

    if regions and regions[-1].end_ms >= track_duration_ms - 100:
        trailing = regions[-1].duration_ms

    return leading, trailing


# ---------------------------------------------------------------------------
# Format info via ffprobe
# ---------------------------------------------------------------------------

def get_format_info(path: str | Path) -> dict[str, str | int]:
    """Get audio format info via ffprobe (if available)."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        return {
            "codec": stream.get("codec_name", "unknown"),
            "sample_rate": int(stream.get("sample_rate", 0)),
            "bits_per_sample": int(stream.get("bits_per_raw_sample", 0)),
            "bitrate": int(fmt.get("bit_rate", 0)) // 1000,
            "duration": float(fmt.get("duration", 0)),
            "format_name": fmt.get("format_name", "unknown"),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Full quality check
# ---------------------------------------------------------------------------

def check_audio_quality(path: str | Path) -> QualityReport:
    """Run all quality checks on a single track."""
    path = Path(path)
    warnings: list[str] = []

    # Format info
    info = get_format_info(path)

    # Fake lossless
    is_fake = False
    fake_conf = 0.0
    if path.suffix.lower() in (".flac", ".wav", ".aiff"):
        is_fake, fake_conf = detect_fake_lossless(path)
        if is_fake:
            warnings.append(f"Possibly transcoded from lossy source (confidence: {fake_conf:.0%})")

    # Clipping
    clip_count = 0
    try:
        clip_count = detect_clipping(path)
        if clip_count > 0:
            warnings.append(f"{clip_count} clipping events detected")
    except Exception:
        pass

    # Silence (pydub dependency — guard against import/decoder failures)
    silence_regions: list[SilenceRegion] = []
    try:
        silence_regions = detect_silence(path)
    except Exception:
        warnings.append("Silence detection unavailable (pydub/ffmpeg missing)")
    duration_ms = int(info.get("duration", 0) * 1000)
    leading, trailing = get_leading_trailing_silence(silence_regions, duration_ms)

    mid_silence = [
        (r.start_ms, r.end_ms) for r in silence_regions
        if r.start_ms > 1000 and r.end_ms < duration_ms - 1000
        and r.duration_ms > 2000
    ]
    if leading > 3000:
        warnings.append(f"Excessive leading silence: {leading}ms")
    if trailing > 5000:
        warnings.append(f"Excessive trailing silence: {trailing}ms")
    if mid_silence:
        warnings.append(f"{len(mid_silence)} mid-track silence region(s) > 2s")

    # Bitrate
    bitrate = int(info.get("bitrate", 0))
    if 0 < bitrate < 320:
        warnings.append(f"Low bitrate: {bitrate}kbps")

    return QualityReport(
        path=str(path),
        format=info.get("codec", path.suffix.lstrip(".")),
        sample_rate=int(info.get("sample_rate", 0)),
        bits_per_sample=int(info.get("bits_per_sample", 0)),
        bitrate=bitrate,
        is_fake_lossless=is_fake,
        fake_lossless_confidence=fake_conf,
        clipping_count=clip_count,
        leading_silence_ms=leading,
        trailing_silence_ms=trailing,
        mid_silence_regions=mid_silence,
        warnings=warnings,
    )
