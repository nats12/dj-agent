"""Shared dataclasses used across all dj_agent modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackInfo:
    """Core track metadata read from Rekordbox."""

    db_content_id: str
    path: str
    artist: str
    title: str
    genre: str
    bpm: float
    key: str
    duration: float
    bitrate: int = 0
    content_hash: str = ""


@dataclass
class EnergyResult:
    """Output of the energy analysis pipeline."""

    integrated_lufs: float
    short_term_max_lufs: float
    spectral_centroid_mean: float
    onset_density: float
    bass_ratio: float
    dynamic_range: float
    raw_score: float
    calibrated_score: int  # 1-10
    vibe_description: Optional[str] = None
    texture_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class CuePoint:
    """A single hot-cue or memory-cue point.

    If ``memory_only`` is True, this cue is written as a Rekordbox memory
    cue only (Num="-1") with no hot cue slot. Used for cues beyond the
    8-slot hot cue limit so the user still sees them in Rekordbox.
    """

    position_ms: int
    name: str  # "Intro", "Drop", "Breakdown", "Outro"
    colour: str  # "green", "red", "blue", "yellow"
    confidence: float = 1.0
    memory_only: bool = False


@dataclass
class LoudnessResult:
    """Output of LUFS measurement."""

    integrated_lufs: float
    sample_peak_dbfs: float
    loudness_range_lu: float
    short_term_max_lufs: float = 0.0


@dataclass
class QualityReport:
    """Audio quality validation results for a single track."""

    path: str
    format: str
    sample_rate: int
    bits_per_sample: int
    bitrate: int
    is_fake_lossless: bool = False
    fake_lossless_confidence: float = 0.0
    clipping_count: int = 0
    leading_silence_ms: int = 0
    trailing_silence_ms: int = 0
    mid_silence_regions: list[tuple[int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class VocalResult:
    """Output of vocal detection."""

    has_vocals: bool
    vocal_probability: float
    method: str  # "essentia", "demucs"
    classification: str = "unknown"  # "vocal", "instrumental", "partial_vocal"


@dataclass
class MoodResult:
    """Output of mood classification."""

    primary_mood: str
    mood_scores: dict[str, float] = field(default_factory=dict)
    arousal: float = 0.0  # 0-1
    valence: float = 0.0  # 0-1
    is_commercial: float = 0.5  # 0=underground, 1=commercial
    hardness: int = 5  # 1-10
    method: str = "essentia"


@dataclass
class SilenceRegion:
    """A detected region of silence."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class Phrase:
    """A detected musical phrase (e.g., 8 or 16 bars)."""

    start_ms: int
    end_ms: int
    bar_count: int
    label: str  # DJ label: "intro", "build", "drop", "breakdown", "outro"
    energy: float = 0.0
    confidence: float = 1.0  # how confident are we in the label
    source_label: str = ""   # original model label (e.g., "chorus", "verse")


@dataclass
class HarmonicSuggestion:
    """A suggested next track for harmonic mixing."""

    track: TrackInfo
    score: float
    key_relation: str  # "same", "adjacent", "relative", "energy_boost"
    bpm_diff: float
    energy_diff: int
