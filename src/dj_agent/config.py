"""YAML-based configuration system.

Adapted from AI-Music-Library-Normalization-Suite config pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class EnergyConfig:
    """Weights and thresholds for energy calculation."""

    lufs_weight: float = 0.30
    spectral_centroid_weight: float = 0.12
    onset_density_weight: float = 0.20
    bpm_weight: float = 0.13
    dynamic_range_weight: float = 0.10
    bass_energy_weight: float = 0.15

    def __post_init__(self) -> None:
        total = (self.lufs_weight + self.spectral_centroid_weight
                 + self.onset_density_weight + self.bpm_weight
                 + self.dynamic_range_weight + self.bass_energy_weight)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Energy weights must sum to 1.0, got {total:.4f}")

    # BPM ranges per genre (lo, hi) for BPM normalisation
    genre_bpm_ranges: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "ambient": (60, 100),
        "downtempo": (80, 115),
        "deep house": (118, 125),
        "house": (120, 130),
        "tech house": (124, 132),
        "techno": (128, 150),
        "hard techno": (140, 160),
        "trance": (128, 145),
        "drum and bass": (160, 180),
        "bouncy": (140, 155),
        "acid": (130, 150),
        "afrohouse": (118, 128),
        "amapiano": (110, 120),
        "disco": (115, 130),
        "psytrance": (140, 150),
        "melodic techno": (120, 132),
    })
    default_bpm_range: tuple[float, float] = (100, 150)


@dataclass
class CueConfig:
    """Cue point detection parameters."""

    min_segments: int = 4
    max_segments: int = 12
    energy_threshold_low: float = 0.45
    energy_threshold_high: float = 0.65
    min_cue_distance_sec: float = 8.0
    phrase_length_bars: int = 8
    use_pssi: bool = True  # Read Rekordbox PSSI tag first if available


@dataclass
class MemoryConfig:
    """Memory file location and backup settings."""

    path: str = "~/.dj-agent/memory.json"
    backup_count: int = 5
    auto_backup: bool = True


@dataclass
class RekordboxConfig:
    """Rekordbox integration settings."""

    check_process: bool = True
    backup_before_write: bool = True
    xml_output_dir: str = "~/Documents/DJ/dj-agent/"


@dataclass
class DuplicateConfig:
    """Duplicate detection parameters."""

    hash_chunk_size: int = 65536  # 64KB
    fuzzy_threshold: int = 85
    duration_tolerance_sec: float = 10.0


@dataclass
class NormalizeConfig:
    """LUFS normalization settings."""

    target_lufs: float = -8.0  # Club standard
    true_peak_db: float = -1.0
    output_format: str = "flac"
    output_bitrate: str = "320k"


@dataclass
class ReasoningConfig:
    """Settings for AI musical reasoning."""

    backend: str = "auto"  # auto, flamingo, ollama, gemini
    model_id: str = "nvidia/audio-flamingo-3-hf"
    ollama_model: str = "qwen3.5:27b"
    gemini_tier: str = "flash"
    quantization: str = "4bit"  # 4bit, 8bit, none
    max_new_tokens: int = 150
    snippet_duration_sec: float = 30.0
    trust_remote_code: bool = False  # HuggingFace trust_remote_code — opt-in only


@dataclass
class Config:
    """Main configuration."""

    energy: EnergyConfig = field(default_factory=EnergyConfig)
    cues: CueConfig = field(default_factory=CueConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rekordbox: RekordboxConfig = field(default_factory=RekordboxConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    processing_tier: str = "fast"  # fast, medium, full

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Filter out fields that don't belong to each dataclass
        energy_data = {k: v for k, v in data.get("energy", {}).items()
                       if k in EnergyConfig.__dataclass_fields__}
        cues_data = {k: v for k, v in data.get("cues", {}).items()
                     if k in CueConfig.__dataclass_fields__}
        reasoning_data = {k: v for k, v in data.get("reasoning", {}).items()
                          if k in ReasoningConfig.__dataclass_fields__}
        def _filter(dc_cls, section: str) -> dict:
            raw = data.get(section, {})
            return {k: v for k, v in raw.items() if k in dc_cls.__dataclass_fields__}

        return cls(
            energy=EnergyConfig(**energy_data),
            cues=CueConfig(**cues_data),
            reasoning=ReasoningConfig(**reasoning_data),
            memory=MemoryConfig(**_filter(MemoryConfig, "memory")),
            rekordbox=RekordboxConfig(**_filter(RekordboxConfig, "rekordbox")),
            duplicates=DuplicateConfig(**_filter(DuplicateConfig, "duplicates")),
            normalize=NormalizeConfig(**_filter(NormalizeConfig, "normalize")),
            processing_tier=data.get("processing_tier", "fast"),
        )

    @classmethod
    def default(cls) -> Config:
        """Create configuration with all defaults."""
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "energy": {k: v for k, v in self.energy.__dict__.items()
                       if k != "genre_bpm_ranges"},
            "cues": self.cues.__dict__,
            "reasoning": self.reasoning.__dict__,
            "memory": self.memory.__dict__,
            "rekordbox": self.rekordbox.__dict__,
            "duplicates": self.duplicates.__dict__,
            "normalize": self.normalize.__dict__,
            "processing_tier": self.processing_tier,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Global config singleton (thread-safe)
# ---------------------------------------------------------------------------

import threading

_config: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Return the global config, loading from config.yaml if present."""
    global _config
    if _config is not None:
        return _config
    with _config_lock:
        if _config is not None:  # double-check after acquiring lock
            return _config
        candidates = [
            Path("config.yaml"),
            Path(__file__).resolve().parent.parent.parent / "config.yaml",
        ]
        for p in candidates:
            if p.exists():
                _config = Config.from_yaml(p)
                return _config
        _config = Config.default()
        return _config


def set_config(config: Config) -> None:
    """Override the global config (useful for tests)."""
    global _config
    with _config_lock:
        _config = config
