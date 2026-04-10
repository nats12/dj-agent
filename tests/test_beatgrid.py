"""Tests for beat grid verification."""

from dj_agent.beatgrid import _fix_half_double, verify_bpm


class TestHalfDoubleFix:
    def test_half_bpm_doubled(self):
        # 65 BPM detected for a Techno track (should be 130)
        result = _fix_half_double(65.0, "Techno")
        assert 125 <= result <= 150

    def test_double_bpm_halved(self):
        # 260 BPM for house (should be 130)
        result = _fix_half_double(260.0, "House")
        assert 118 <= result <= 132

    def test_correct_bpm_unchanged(self):
        result = _fix_half_double(128.0, "Techno")
        assert result == 128.0

    def test_dnb_range(self):
        result = _fix_half_double(87.0, "Drum and Bass")
        assert 160 <= result <= 180

    def test_no_genre_uses_default(self):
        result = _fix_half_double(65.0, None)
        assert 80 <= result <= 180


class TestVerifyBpm:
    def test_matching_bpm(self, sample_sine_wav):
        """Even a sine wave should return a valid result dict."""
        result = verify_bpm(sample_sine_wav, rekordbox_bpm=128.0)
        assert "detected_bpm" in result
        assert "issue" in result
        assert isinstance(result["match"], bool)

    def test_result_structure(self, sample_sine_wav):
        result = verify_bpm(sample_sine_wav, rekordbox_bpm=128.0, genre="Techno")
        assert "suggested_bpm" in result
        assert result["rekordbox_bpm"] == 128.0
