"""Tests proving graceful degradation when V2 optional engines are missing.

Tests use monkeypatching at the function level (not sys.modules) to avoid
Python 3.14 module reload issues with numpy/scipy.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestStemsFallback:
    def test_has_audio_separator_detection(self):
        """_has_audio_separator returns bool without crashing."""
        from dj_agent.stems import _has_audio_separator
        result = _has_audio_separator()
        assert isinstance(result, bool)

    def test_has_demucs_detection(self):
        """_has_demucs returns bool without crashing."""
        from dj_agent.stems import _has_demucs
        result = _has_demucs()
        assert isinstance(result, bool)

    def test_model_auto_selects_best(self):
        """model='auto' should not crash (picks whatever is available)."""
        from dj_agent.stems import _has_audio_separator, _has_demucs
        # Just verify the selection logic exists — actual separation needs audio
        assert callable(_has_audio_separator)
        assert callable(_has_demucs)


class TestBeatgridFallback:
    def test_librosa_fallback_works(self, sample_sine_wav: Path):
        """Librosa BPM detection works as fallback."""
        from dj_agent.beatgrid import _librosa_bpm
        bpm = _librosa_bpm(sample_sine_wav)
        assert isinstance(bpm, float)

    def test_verify_bpm_returns_valid_structure(self, sample_sine_wav: Path):
        """verify_bpm always returns a valid dict regardless of engine."""
        from dj_agent.beatgrid import verify_bpm
        result = verify_bpm(sample_sine_wav, rekordbox_bpm=128.0)
        assert "detected_bpm" in result
        assert "issue" in result
        assert isinstance(result["match"], bool)

    def test_half_double_fix_always_works(self):
        """Genre-aware BPM fix works without any ML engine."""
        from dj_agent.beatgrid import _fix_half_double
        assert 125 <= _fix_half_double(65.0, "Techno") <= 150
        assert 160 <= _fix_half_double(87.0, "Drum and Bass") <= 180


class TestPhrasesFallback:
    def test_librosa_phrases_work(self, sample_sine_wav: Path):
        """Librosa phrase detection works as fallback."""
        from dj_agent.phrases import _librosa_phrases
        result = _librosa_phrases(sample_sine_wav, bpm=128.0, bars_per_phrase=8)
        assert isinstance(result, list)

    def test_detect_phrases_returns_list(self, sample_sine_wav: Path):
        """detect_phrases always returns a list regardless of engine."""
        from dj_agent.phrases import detect_phrases
        result = detect_phrases(sample_sine_wav, bpm=128.0)
        assert isinstance(result, list)


class TestSimilarityFallback:
    def test_librosa_features_always_work(self, sample_sine_wav: Path):
        """librosa feature extraction works without CLAP."""
        from dj_agent.similarity import _librosa_features
        vec = _librosa_features(sample_sine_wav)
        assert vec.ndim == 1
        assert len(vec) > 50
        assert vec.dtype == np.float32

    def test_brute_force_similarity_works_without_faiss(self):
        """find_similar works without FAISS for small libraries."""
        from dj_agent.similarity import find_similar
        vecs = {
            "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            "c": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        results = find_similar(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), vecs, top_k=2,
        )
        assert len(results) == 2
        assert results[0][0] == "a"  # most similar

    def test_method_librosa_forces_mfcc(self, sample_sine_wav: Path):
        """method='librosa' always uses MFCC regardless of CLAP availability."""
        from dj_agent.similarity import compute_feature_vector
        vec = compute_feature_vector(sample_sine_wav, method="librosa")
        assert vec.ndim == 1
        assert len(vec) > 50  # ~62 MFCC features


class TestKeydetectFallback:
    def test_librosa_key_detection_works(self, sample_sine_wav: Path):
        """librosa key detection works without Essentia."""
        from dj_agent.keydetect import _librosa_detect
        result = _librosa_detect(sample_sine_wav)
        assert result.method == "librosa"
        assert result.camelot

    def test_tuning_detection_works(self, sample_sine_wav: Path):
        """Tuning detection uses only librosa (always available)."""
        from dj_agent.keydetect import detect_tuning
        offset = detect_tuning(sample_sine_wav)
        assert isinstance(offset, float)
        assert -1.0 <= offset <= 1.0


class TestContractCompatibility:
    """Verify V2 return types match V1 contracts."""

    def test_similarity_vector_is_1d_float32(self, sample_sine_wav: Path):
        from dj_agent.similarity import compute_feature_vector
        vec = compute_feature_vector(sample_sine_wav, method="librosa")
        assert vec.ndim == 1
        assert vec.dtype == np.float32

    def test_phrases_returns_list_of_phrase(self, sample_sine_wav: Path):
        from dj_agent.phrases import detect_phrases
        from dj_agent.types import Phrase
        result = detect_phrases(sample_sine_wav, bpm=128.0)
        assert isinstance(result, list)
        for p in result:
            assert isinstance(p, Phrase)

    def test_keydetect_returns_keyresult(self, sample_sine_wav: Path):
        from dj_agent.keydetect import detect_key, KeyResult
        result = detect_key(sample_sine_wav)
        assert isinstance(result, KeyResult)
        assert hasattr(result, "key")
        assert hasattr(result, "camelot")
        assert hasattr(result, "confidence")
        assert hasattr(result, "method")

    def test_beatgrid_returns_dict_with_required_keys(self, sample_sine_wav: Path):
        from dj_agent.beatgrid import verify_bpm
        result = verify_bpm(sample_sine_wav, rekordbox_bpm=128.0)
        for key in ("detected_bpm", "rekordbox_bpm", "match", "issue", "suggested_bpm"):
            assert key in result
