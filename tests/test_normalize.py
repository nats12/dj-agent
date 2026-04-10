"""Tests for LUFS normalization."""

from pathlib import Path

import numpy as np
import pytest

from dj_agent.normalize import measure_track, format_loudness_report


def test_measure_track(sample_sine_wav: Path):
    result = measure_track(sample_sine_wav)
    # A 0.5 amplitude sine should be well below 0 dBFS
    assert result.integrated_lufs < 0
    assert result.sample_peak_dbfs < 0
    assert isinstance(result.loudness_range_lu, float)


def test_measure_returns_finite(sample_sine_wav: Path):
    result = measure_track(sample_sine_wav)
    assert np.isfinite(result.integrated_lufs)
    assert np.isfinite(result.sample_peak_dbfs)


def test_format_report(sample_sine_wav: Path):
    result = measure_track(sample_sine_wav)
    report = format_loudness_report([(sample_sine_wav, result)], target_lufs=-8.0)
    assert "LUFS" in report
    assert "sine" in report
