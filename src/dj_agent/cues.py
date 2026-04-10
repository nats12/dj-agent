"""Phrase-aware cue point detection with multi-feature labelling.

Detects structural segment boundaries via agglomerative clustering on a
mel spectrogram, then extracts per-segment features (bass RMS, vocal-band
harmonic ratio, percussive ratio, overall RMS, centroid) and labels each
segment with a DJ-relevant name (Intro, Drop, Breakdown, Build, Vocal,
Groove, Outro).

Also detects vocal re-entry points: if vocals return after ≥4 bars of
instrumental silence (BPM-aware), a "Vocal In" cue is emitted. 4 bars at
120 BPM ≈ 8 s, scaling with tempo to match musical phrasing.

Sources:
- PSSI-first: reads Rekordbox song structure if available
- Adaptive k based on track duration
- Phrase-level snapping (8-bar boundaries)
- Research: 4-bar threshold matches Mixed In Key's vocal-cue heuristic
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .config import CueConfig
from .types import CuePoint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cue_points(
    audio: np.ndarray,
    sr: int,
    bpm: float,
    duration: float,
    config: CueConfig | None = None,
    anlz_path: str | Path | None = None,
    has_vocals: bool | None = None,
) -> list[CuePoint]:
    """Detect structural cue points from audio.

    If *anlz_path* is provided and ``config.use_pssi`` is True, tries to
    read Rekordbox's PSSI (song structure) tag first.  Falls back to
    audio-based detection if PSSI is unavailable.

    If *has_vocals* is False, vocal re-entry detection is skipped (it uses
    a librosa heuristic that false-positives on instrumental tracks with
    prominent synth leads in the 300-4000 Hz band). Pass the track-level
    Essentia vocal classification if available.

    Returns a list of :class:`CuePoint` sorted by position.
    """
    if config is None:
        from .config import get_config
        config = get_config().cues

    # PSSI-first: read Rekordbox song structure if available
    if config.use_pssi and anlz_path:
        pssi_cues = detect_cue_points_from_pssi(anlz_path)
        if pssi_cues:
            return pssi_cues[:8]

    k = _adaptive_k(duration, config.min_segments, config.max_segments)

    # Mel spectrogram for segmentation
    S = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)

    # RMS energy per frame
    rms = librosa.feature.rms(y=audio, hop_length=512)[0]

    # Agglomerative clustering
    try:
        bounds = librosa.segment.agglomerative(S_db, k=k)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        try:
            bounds = librosa.segment.agglomerative(S_db, k=max(4, k // 2))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return []

    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=512)

    # Snap to phrase boundaries
    bpm = bpm if bpm and bpm > 0 else 120.0  # default to 120 if missing
    beat_sec = 60.0 / bpm
    phrase_sec = config.phrase_length_bars * beat_sec * 4  # 4 beats/bar
    bound_times = np.array([_snap_to_phrase(t, phrase_sec) for t in bound_times])

    # Build segments with multi-feature extraction
    segments = _build_segments_with_features(bound_times, audio, sr, duration)
    if not segments:
        return []

    # Normalise features across segments (relative scoring per track)
    _normalise_features(segments)

    # Label each segment boundary using bass/vocal/RMS features.
    # If the track is instrumental, suppress "Vocal" labels in segment
    # classification — the vocal-band heuristic triggers on synth leads.
    cues = _classify_segments(
        segments, duration, config,
        allow_vocal_label=(has_vocals is not False),
    )

    # Detect vocal re-entry points (BPM-aware: ≥4 bars of instrumental gap).
    # Skip entirely if the track is known to be instrumental.
    if has_vocals is not False:
        vocal_cues = detect_vocal_entries(audio, sr, duration, bpm, segments)
        cues.extend(vocal_cues)

    # Deduplicate cues that are too close
    cues = _deduplicate(cues, config.min_cue_distance_sec)

    # Cap at 8 (Rekordbox hot-cue max) — keep cues that span the whole
    # track. Prefer Drops, Breakdowns, Vocal entries; always keep Intro/Outro.
    return _select_top_cues(cues, max_cues=8)


def detect_cue_points_from_pssi(
    anlz_path: str | Path,
    bpm: float = 0,
) -> list[CuePoint] | None:
    """Read cue points from Rekordbox's PSSI (Phrase Structure) tag.

    Uses Rekordbox's own ML-based phrase analysis — much more accurate
    than librosa segmentation because it uses a purpose-trained model
    and the track's own beat grid.

    The ``kind`` field meaning depends on the track's ``mood`` value:

      - High mood (1): Intro(1), Up/Build(2), Down/Breakdown(3),
        Chorus/Drop(5), Outro(6)
      - Mid mood (2): Intro(1), Verse 1-6(2-7), Bridge(8), Chorus(9),
        Outro(10)
      - Low mood (3): same as Mid but fewer verse variants

    PSSI spec: https://djl-analysis.deepsymmetry.org/rekordbox-export-analysis/anlz.html

    Returns ``None`` if the PSSI tag is not present.
    """
    try:
        from pyrekordbox.anlz import AnlzFile  # type: ignore[import-untyped]

        ext_path = Path(anlz_path).with_suffix(".EXT")
        if not ext_path.exists():
            return None

        anlz = AnlzFile.parse_file(str(ext_path))
        pssi_tags = anlz.getall("PSSI")
        if not pssi_tags or not pssi_tags[0].entries:
            return None

        pssi = pssi_tags[0]
        entries = pssi.entries

        # Select kind mapping based on mood
        mood = getattr(pssi, "mood", None)
        mood_val = mood if isinstance(mood, int) else getattr(mood, "value", None)
        kind_map = _PSSI_MOOD_MAPS.get(mood_val, _PSSI_KIND_HIGH)

        # Convert beat indices to ms
        if bpm <= 0:
            bpm = 120.0  # safe fallback

        # Emit a cue at each phrase TRANSITION (where label changes).
        # Consecutive same-label phrases are a sustained section — only
        # the first gets a hot cue.
        cues: list[CuePoint] = []
        prev_label = None
        for entry in entries:
            kind = entry.kind
            kind_val = kind if isinstance(kind, int) else getattr(kind, "value", kind)

            label = kind_map.get(kind_val, "Section")
            colour = _PSSI_LABEL_COLOURS.get(label, "green")

            # k1/k2 variants for high mood (e.g. "Drop" vs "Drop 2")
            k1 = getattr(entry, "k1", 0)
            k2 = getattr(entry, "k2", 0)
            variant = ""
            if kind_map is _PSSI_KIND_HIGH and label in ("Drop", "Build", "Intro"):
                if k2:
                    variant = " 2"
            full_label = f"{label}{variant}"

            if full_label == prev_label:
                continue
            prev_label = full_label

            pos_ms = int((entry.beat - 1) * 60000.0 / bpm)

            cues.append(CuePoint(
                position_ms=pos_ms,
                name=full_label,
                colour=colour,
                confidence=1.0,
            ))

        if not cues:
            return None

        # Mark overflow beyond 8 hot cues as memory-only
        if len(cues) > 8:
            _priority = {"Intro": 0, "Outro": 1, "Drop": 2, "Chorus": 2,
                         "Breakdown": 3, "Bridge": 3, "Build": 4, "Verse": 5}
            hot = {0, len(cues) - 1}
            middle = sorted(
                range(1, len(cues) - 1),
                key=lambda i: _priority.get(cues[i].name.split()[0], 9),
            )
            for i in middle:
                if len(hot) >= 8:
                    break
                hot.add(i)
            for i, c in enumerate(cues):
                if i not in hot:
                    c.memory_only = True

        return cues

    except Exception:
        return None


# Kind mapping tables per PSSI mood type
_PSSI_KIND_HIGH: dict[int, str] = {
    1: "Intro", 2: "Build", 3: "Breakdown", 5: "Drop", 6: "Outro",
}
_PSSI_KIND_MID: dict[int, str] = {
    1: "Intro", 2: "Verse", 3: "Verse", 4: "Verse", 5: "Verse",
    6: "Verse", 7: "Verse", 8: "Bridge", 9: "Chorus", 10: "Outro",
}
_PSSI_KIND_LOW: dict[int, str] = _PSSI_KIND_MID  # same structure
_PSSI_MOOD_MAPS = {1: _PSSI_KIND_HIGH, 2: _PSSI_KIND_MID, 3: _PSSI_KIND_LOW}
_PSSI_LABEL_COLOURS: dict[str, str] = {
    "Intro": "green", "Build": "yellow", "Breakdown": "blue",
    "Drop": "red", "Outro": "green", "Verse": "green",
    "Bridge": "blue", "Chorus": "red",
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _adaptive_k(duration: float, min_k: int, max_k: int) -> int:
    """Scale segment count with duration (~1 segment per 30 s)."""
    k = int(duration / 30)
    return max(min_k, min(max_k, k))


def _snap_to_phrase(time_sec: float, phrase_sec: float) -> float:
    """Snap a time to the nearest phrase boundary."""
    if phrase_sec <= 0:
        return time_sec
    return round(time_sec / phrase_sec) * phrase_sec


def _build_segments_with_features(
    bound_times: np.ndarray,
    audio: np.ndarray,
    sr: int,
    duration: float,
) -> list[dict]:
    """Build segment records with per-segment audio features.

    Each segment carries: start, end, rms, bass, vocal (vocal-band harmonic
    ratio), perc (percussive ratio), centroid. Features are raw (not
    normalised) — normalisation happens afterward per track.
    """
    segments: list[dict] = []
    for i in range(len(bound_times)):
        start = float(bound_times[i])
        end = float(bound_times[i + 1]) if i + 1 < len(bound_times) else duration
        if end - start < 2.0:  # skip tiny segments
            continue
        feats = _extract_features(audio, sr, start, end)
        if feats is None:
            continue
        feats["start"] = start
        feats["end"] = end
        segments.append(feats)
    return segments


def _extract_features(
    audio: np.ndarray, sr: int, start_sec: float, end_sec: float
) -> dict | None:
    """Extract DJ-relevant features for a single segment window."""
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    if e - s < sr:  # need at least 1 s
        return None
    seg = audio[s:e]

    # Overall RMS
    rms = float(np.sqrt(np.mean(seg ** 2)))

    # STFT for band-limited features
    try:
        stft = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512))
    except Exception:
        return None
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Bass energy (20-150 Hz) — tells us if kick+bass are present
    bass_mask = (freqs >= 20) & (freqs <= 150)
    bass = float(np.mean(stft[bass_mask])) if bass_mask.any() else 0.0

    # Vocal-band harmonic ratio (300-4000 Hz) — proxy for vocal presence
    try:
        y_harm, y_perc = librosa.effects.hpss(seg, margin=1.0)
        stft_h = np.abs(librosa.stft(y_harm, n_fft=2048, hop_length=512))
        vocal_mask = (freqs >= 300) & (freqs <= 4000)
        vocal_band = float(np.mean(stft_h[vocal_mask])) if vocal_mask.any() else 0.0
        total_harm = float(np.mean(stft_h)) + 1e-8
        vocal = vocal_band / total_harm
        # Percussive vs total energy (drumless intro/break → low perc ratio)
        perc_e = float(np.sqrt(np.mean(y_perc ** 2)))
        harm_e = float(np.sqrt(np.mean(y_harm ** 2)))
        perc_ratio = perc_e / (perc_e + harm_e + 1e-8)
    except Exception:
        vocal = 0.0
        perc_ratio = 0.5

    # Spectral centroid — brightness indicator (filter sweeps, rolloff)
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
    except Exception:
        centroid = 0.0

    return {
        "rms": rms,
        "bass": bass,
        "vocal": vocal,
        "perc_ratio": perc_ratio,
        "centroid": centroid,
    }


def _normalise_features(segments: list[dict]) -> None:
    """Normalise rms/bass/vocal/centroid to [0,1] per track (in-place)."""
    for key in ("rms", "bass", "vocal", "centroid"):
        values = [s[key] for s in segments]
        max_v = max(values) + 1e-8
        for s in segments:
            s[f"{key}_n"] = s[key] / max_v
    # Keep a legacy "energy" alias for downstream code
    for s in segments:
        s["energy"] = s["rms_n"]


def _classify_segments(
    segments: list[dict],
    duration: float,
    config: CueConfig,
    allow_vocal_label: bool = True,
) -> list[CuePoint]:
    """Label each segment boundary by its multi-feature signature.

    Uses bass + rms + vocal + percussive ratio + position to distinguish:
    - Intro: start of track, low bass/rms
    - Drop: high bass + high rms, especially after a low-bass section
    - Breakdown: low bass + mid-low rms in mid-track
    - Build: rising rms, bass still developing (often pre-drop)
    - Vocal: high vocal-band harmonic ratio + moderate+ rms
    - Groove: sustained bass+rms, no vocal, between drops
    - Outro: final segment with falling rms
    """
    cues: list[CuePoint] = []
    if not segments:
        return cues

    # First segment: Intro
    first_pos = int(segments[0]["start"] * 1000)
    cues.append(CuePoint(position_ms=first_pos, name="Intro", colour="green"))

    for i in range(1, len(segments)):
        s = segments[i]
        prev = segments[i - 1]
        pos = int(s["start"] * 1000)
        rel = s["start"] / duration if duration > 0 else 0

        rms = s["rms_n"]
        bass = s["bass_n"]
        vocal = s["vocal_n"]
        prev_bass = prev["bass_n"]
        prev_rms = prev["rms_n"]

        name, colour = _label_segment(
            rms=rms, bass=bass, vocal=vocal,
            prev_rms=prev_rms, prev_bass=prev_bass,
            rel_pos=rel, is_last=(i == len(segments) - 1),
            allow_vocal=allow_vocal_label,
        )
        cues.append(CuePoint(position_ms=pos, name=name, colour=colour))

    return cues


def _label_segment(
    *,
    rms: float, bass: float, vocal: float,
    prev_rms: float, prev_bass: float,
    rel_pos: float, is_last: bool,
    allow_vocal: bool = True,
) -> tuple[str, str]:
    """Pick a DJ-relevant label from normalised segment features.

    All inputs are in [0, 1] (track-relative). Decision order matters —
    specific cases first, generic fallbacks last.
    """
    # Outro: final segment, RMS falling
    if is_last and rel_pos > 0.60 and rms < 0.55:
        return "Outro", "yellow"

    # Drop: big bass + big RMS, ideally after a low-bass section
    if bass >= 0.70 and rms >= 0.70:
        if prev_bass < 0.45:
            return "Drop", "red"  # classic drop entry
        return "Peak", "red"      # sustained high energy

    # Breakdown: low bass + dropped RMS in mid-track
    if bass < 0.40 and rms < 0.60 and 0.15 < rel_pos < 0.85:
        return "Breakdown", "blue"

    # Build-up: RMS rising + bass still developing (common pre-drop pattern)
    if rms > prev_rms + 0.12 and bass < 0.65:
        return "Build", "yellow"

    # Vocal section: strong vocal band + sustained RMS (only if track has vocals)
    if allow_vocal and vocal >= 0.75 and rms >= 0.50:
        return "Vocal", "green"

    # Groove / main section: beat + bass present, no big change
    if bass >= 0.50 and rms >= 0.50:
        return "Groove", "red"

    # Low-energy sustain (ambient, quiet section)
    return "Break", "blue"


def detect_vocal_entries(
    audio: np.ndarray,
    sr: int,
    duration: float,
    bpm: float,
    segments: list[dict] | None = None,
) -> list[CuePoint]:
    """Detect vocal re-entry points as hot cues.

    Slides a short analysis window across the track, tracks whether each
    window contains vocals (harmonic energy in 300-4000 Hz band), and
    emits a "Vocal" cue whenever vocals return after ≥4 bars of
    instrumental silence. BPM-aware: 4 bars at 120 BPM ≈ 8 s.

    Works with both librosa segments (if already computed) and standalone
    (e.g. after PSSI cue detection) — when ``segments`` is None, a quick
    first pass over the track establishes the vocal threshold directly.
    """
    if bpm <= 0 or duration <= 0:
        return []

    # 4 bars of instrumental silence = phrase unit DJs treat as a section
    beat_sec = 60.0 / bpm
    min_gap_sec = 4 * 4 * beat_sec  # 4 bars × 4 beats

    # Analysis window ≈ 2 beats, hop ≈ 1 beat — fine-grained but cheap
    win_sec = max(2.0, 2 * beat_sec)
    hop_sec = max(1.0, beat_sec)

    # Establish vocal threshold from segments (if available) or from
    # a quick scan of the track at ~30s intervals
    if segments and all("vocal" in s for s in segments):
        seg_vocals = sorted(s["vocal"] for s in segments)
    else:
        seg_vocals = []
        for t_scan in np.arange(0, duration - win_sec, 30.0):
            feats = _extract_features(audio, sr, t_scan, t_scan + win_sec)
            if feats is not None:
                seg_vocals.append(feats["vocal"])
        seg_vocals.sort()

    if not seg_vocals:
        return []
    median_vocal = seg_vocals[len(seg_vocals) // 2]
    max_vocal = seg_vocals[-1] + 1e-8
    # A window "has vocals" when its vocal ratio is well above the median
    vocal_threshold = median_vocal + 0.15 * (max_vocal - median_vocal)

    # Slide windows across the track
    windows: list[tuple[float, bool]] = []
    t = 0.0
    while t + win_sec <= duration:
        feats = _extract_features(audio, sr, t, t + win_sec)
        if feats is not None:
            windows.append((t, feats["vocal"] > vocal_threshold))
        t += hop_sec

    if not windows:
        return []

    # Walk windows, emit a cue when vocals return after a gap of ≥ min_gap_sec
    cues: list[CuePoint] = []
    last_vocal_end = -1e9
    currently_vocal = False
    for t_start, has_vocal in windows:
        if has_vocal:
            if not currently_vocal:
                gap = t_start - last_vocal_end
                if gap >= min_gap_sec and t_start > 0.5:  # not the very start
                    cues.append(CuePoint(
                        position_ms=int(t_start * 1000),
                        name="Vocal",
                        colour="green",
                        confidence=0.8,
                    ))
            currently_vocal = True
            last_vocal_end = t_start + win_sec
        else:
            currently_vocal = False

    return cues


def _deduplicate(cues: list[CuePoint], min_dist_sec: float) -> list[CuePoint]:
    """Remove cues that are too close together or that repeat the same label.

    Two rules:
    1. Cues within *min_dist_sec* of each other are merged (keep the first).
    2. Consecutive cues with the same name are collapsed — the repeated
       ones are demoted rather than dropped, so structural markers win
       and we still get varied labels. A chain of "Breakdown"s becomes a
       single "Breakdown" followed by (nothing) — downstream selection
       then fills remaining slots with other cue types.
    """
    if not cues:
        return cues
    cues.sort(key=lambda c: c.position_ms)

    # Pass 1: distance dedup
    result: list[CuePoint] = [cues[0]]
    for cue in cues[1:]:
        if (cue.position_ms - result[-1].position_ms) >= min_dist_sec * 1000:
            result.append(cue)

    # Pass 2: collapse consecutive identical labels
    collapsed: list[CuePoint] = []
    for cue in result:
        if collapsed and collapsed[-1].name == cue.name:
            continue
        collapsed.append(cue)
    return collapsed


# Structural-importance ranking for cue selection when we have more than 8
#
# Hot cues = structural (for NAVIGATION — press a letter to jump).
# Memory cues = vocal entries (for AWARENESS — visible on waveform).
#
# This gives the DJ 8 structural navigation points as hot cues (Intro,
# Drop, Breakdown, Build, Outro) PLUS vocal entry markers as memory
# cues visible on the waveform without consuming hot cue slots.
_CUE_PRIORITY = {
    "Intro": 0,     # always keep
    "Outro": 1,     # always keep
    "Drop": 2,      # DJ mix-in/out points
    "Breakdown": 3, # energy dip markers
    "Build": 4,
    "Chorus": 5,
    "Bridge": 5,
    "Verse": 6,
    "Peak": 7,
    "Groove": 8,
    "Break": 9,
    "Vocal": 10,    # memory cue — visible on waveform, doesn't take hot slots
}


def _select_top_cues(cues: list[CuePoint], max_cues: int = 8) -> list[CuePoint]:
    """Rank cues and mark overflow as memory-only.

    Strategy: always keep Intro (first) and Outro (last) as hot cues;
    from the middle, pick cues by structural priority (Drops, Breakdowns,
    Vocal entries first) until we hit the hot-cue limit. Any cue beyond
    *max_cues* is kept but tagged with ``memory_only=True`` so sync.py
    writes it as a memory cue only (Num=-1, no hot cue slot).

    Returns ALL cues in chronological order — the first *max_cues* as
    hot cues, the rest as memory-only.
    """
    if len(cues) <= max_cues:
        return cues

    # Sort by position so first=Intro, last=Outro
    cues = sorted(cues, key=lambda c: c.position_ms)
    hot_cue_indices: set[int] = {0, len(cues) - 1}  # always keep first + last

    # Rank middle cues by priority, then by position (earlier first on ties)
    middle_indices = list(range(1, len(cues) - 1))
    middle_indices.sort(
        key=lambda i: (_CUE_PRIORITY.get(cues[i].name, 99), cues[i].position_ms)
    )

    for i in middle_indices:
        if len(hot_cue_indices) >= max_cues:
            break
        hot_cue_indices.add(i)

    # Tag overflow cues as memory-only
    for i, cue in enumerate(cues):
        if i not in hot_cue_indices:
            cue.memory_only = True

    return cues
