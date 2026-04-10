"""Tests for mashup idea engine."""

from dj_agent.mashups import find_mashup_candidates, score_mashup
from dj_agent.types import TrackInfo


def _t(bpm: float, key: str, genre: str = "Techno", cid: str = "1") -> TrackInfo:
    return TrackInfo(cid, "/a", "Art", "Song", genre, bpm, key, 300)


def test_same_key_scores_high():
    ms = score_mashup(_t(128, "8B"), _t(128, "8B"))
    assert ms.harmonic == 1.0
    assert ms.total > 0.7


def test_distant_key_scores_low():
    ms = score_mashup(_t(128, "8B"), _t(128, "2A"))
    assert ms.harmonic <= 0.1


def test_vocal_instrumental_complement():
    """One vocal + one instrumental should score higher than two vocals."""
    tags_vocal = {"vocal": True, "energy": 7}
    tags_instr = {"vocal": False, "energy": 7}

    ms_complement = score_mashup(_t(128, "8B"), _t(128, "8B"), tags_vocal, tags_instr)
    ms_both_vocal = score_mashup(_t(128, "8B"), _t(128, "8B"), tags_vocal, tags_vocal)

    assert ms_complement.vocal_complement > ms_both_vocal.vocal_complement


def test_bpm_tolerance():
    """BPM within 3% should score well."""
    ms_close = score_mashup(_t(128, "8B"), _t(130, "8B"))
    ms_far = score_mashup(_t(128, "8B"), _t(160, "8B"))
    assert ms_close.bpm > ms_far.bpm


def test_half_tempo_compatible():
    """128 BPM and 64 BPM should be compatible (half tempo)."""
    ms = score_mashup(_t(128, "8B"), _t(64, "8B"))
    assert ms.bpm > 0.3  # half tempo should be recognized


def test_find_candidates():
    library = [
        _t(128, "8B", cid="1"),
        _t(129, "9B", cid="2"),  # good match
        _t(128, "8B", cid="3"),  # perfect match
        _t(90, "2A", cid="4"),   # bad match
    ]
    target = _t(128, "8B", cid="target")
    results = find_mashup_candidates(target, library, top_k=5)
    assert len(results) >= 2
    # Best match should be one of the compatible tracks
    ids = [s.track.db_content_id for s in results[:2]]
    assert "3" in ids or "2" in ids


def test_tip_generation():
    library = [_t(128, "3B", cid="2")]
    target = _t(128, "8B", cid="1")
    results = find_mashup_candidates(target, library, top_k=1)
    if results:
        assert isinstance(results[0].tip, str)
