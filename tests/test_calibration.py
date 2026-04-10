"""Tests for calibration — no double-counting, median-based."""

from dj_agent.calibration import apply_calibration, recalculate_calibration


def test_genre_offset_used_when_available():
    cal = {"global_offset": 1.0, "genre_offsets": {"techno": -0.5}}
    result = apply_calibration(5.0, "Techno", cal)
    # Should use genre offset (-0.5), NOT global + genre
    # 5.0 + (-0.5) = 4.5 → np.round rounds to 4 (banker's rounding)
    assert result == 4


def test_global_offset_used_when_no_genre():
    cal = {"global_offset": 1.0, "genre_offsets": {"techno": -0.5}}
    result = apply_calibration(5.0, "House", cal)
    # House not in genre_offsets → use global
    assert result == 6  # 5.0 + 1.0 = 6


def test_never_sums_global_and_genre():
    cal = {"global_offset": 2.0, "genre_offsets": {"techno": 1.0}}
    result = apply_calibration(5.0, "Techno", cal)
    # Must be 5+1=6, NOT 5+2+1=8
    assert result == 6


def test_clamp_to_range():
    cal = {"global_offset": 5.0, "genre_offsets": {}}
    assert apply_calibration(8.0, "Techno", cal) == 10  # clamped


def test_recalculate_with_genre_threshold():
    corrections = [
        {"original_energy": 5, "corrected_energy": 7, "genre": "Techno"},
        {"original_energy": 6, "corrected_energy": 8, "genre": "Techno"},
        {"original_energy": 4, "corrected_energy": 6, "genre": "Techno"},
        {"original_energy": 5, "corrected_energy": 6, "genre": "House"},
    ]
    result = recalculate_calibration(corrections, min_for_genre=3)

    # Techno has 3+ corrections → gets its own offset (median of [2, 2, 2] = 2)
    assert "techno" in result["genre_offsets"]
    assert result["genre_offsets"]["techno"] == 2.0

    # House has only 1 → goes to ungrouped → global = median of [1] = 1
    assert result["global_offset"] == 1.0


def test_recalculate_uses_median():
    corrections = [
        {"original_energy": 5, "corrected_energy": 7, "genre": "Techno"},
        {"original_energy": 5, "corrected_energy": 7, "genre": "Techno"},
        {"original_energy": 5, "corrected_energy": 7, "genre": "Techno"},
        {"original_energy": 5, "corrected_energy": 2, "genre": "Techno"},  # outlier
    ]
    result = recalculate_calibration(corrections, min_for_genre=3)
    # Median of [2, 2, 2, -3] = 2.0 (not mean which would be 0.75)
    assert result["genre_offsets"]["techno"] == 2.0
