"""Library analytics — distributions, coverage gaps, metadata completeness."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .types import TrackInfo


def analyse_library(tracks: list[TrackInfo]) -> dict[str, Any]:
    """Compute analytics for a full track library.

    Returns a dict with distribution data, coverage gaps, and
    metadata completeness scores.
    """
    total = len(tracks)
    if total == 0:
        return {"total": 0}

    # Genre distribution
    genres = Counter(t.genre for t in tracks if t.genre)
    top_genres = genres.most_common(20)

    # BPM distribution (5-BPM buckets)
    bpm_buckets: Counter[int] = Counter()
    bpms: list[float] = []
    for t in tracks:
        if t.bpm > 0:
            bucket = int(t.bpm // 5) * 5
            bpm_buckets[bucket] += 1
            bpms.append(t.bpm)

    # Key distribution
    keys = Counter(t.key for t in tracks if t.key)

    # Metadata completeness
    missing_genre = sum(1 for t in tracks if not t.genre)
    missing_key = sum(1 for t in tracks if not t.key)
    missing_artist = sum(1 for t in tracks if not t.artist)
    missing_bpm = sum(1 for t in tracks if not t.bpm or t.bpm <= 0)

    completeness = 1.0 - (
        (missing_genre + missing_key + missing_artist + missing_bpm) / (total * 4)
    )

    # BPM gaps (ranges with no tracks)
    bpm_range = range(80, 185, 5)
    bpm_gaps = [
        b for b in bpm_range
        if bpm_buckets.get(b, 0) == 0
        and (bpm_buckets.get(b - 5, 0) > 0 or bpm_buckets.get(b + 5, 0) > 0)
    ]

    # Key coverage (Camelot wheel)
    all_camelot = [f"{n}{l}" for n in range(1, 13) for l in ("A", "B")]
    key_coverage = sum(1 for k in all_camelot if keys.get(k, 0) > 0)

    return {
        "total": total,
        "genre_distribution": top_genres,
        "bpm_distribution": sorted(bpm_buckets.items()),
        "bpm_min": min(bpms) if bpms else 0,
        "bpm_max": max(bpms) if bpms else 0,
        "bpm_avg": round(sum(bpms) / len(bpms), 1) if bpms else 0,
        "bpm_gaps": bpm_gaps,
        "key_distribution": keys.most_common(24),
        "key_coverage": f"{key_coverage}/24",
        "missing_genre": missing_genre,
        "missing_key": missing_key,
        "missing_artist": missing_artist,
        "missing_bpm": missing_bpm,
        "metadata_completeness": round(completeness * 100, 1),
    }


def format_analytics(report: dict[str, Any]) -> str:
    """Format analytics as a human-readable string."""
    if report.get("total", 0) == 0:
        return "No tracks to analyse."

    lines = [
        "=" * 55,
        f"  LIBRARY ANALYTICS — {report['total']} tracks",
        "=" * 55,
        "",
        f"Metadata completeness: {report['metadata_completeness']}%",
        f"  Missing genre: {report['missing_genre']}",
        f"  Missing key: {report['missing_key']}",
        f"  Missing artist: {report['missing_artist']}",
        f"  Missing BPM: {report['missing_bpm']}",
        "",
        f"BPM range: {report['bpm_min']}-{report['bpm_max']}, avg: {report['bpm_avg']}",
        f"Key coverage: {report['key_coverage']} Camelot positions",
        "",
    ]

    # Genre chart
    if report["genre_distribution"]:
        lines.append("Top genres:")
        max_count = report["genre_distribution"][0][1]
        for genre, count in report["genre_distribution"][:15]:
            bar_len = int((count / max(max_count, 1)) * 25)
            bar = "\u2588" * bar_len
            lines.append(f"  {genre:<25} {bar} {count}")
        lines.append("")

    # BPM chart
    if report["bpm_distribution"]:
        lines.append("BPM distribution:")
        max_bpm_count = max(c for _, c in report["bpm_distribution"])
        for bucket, count in report["bpm_distribution"]:
            bar_len = int((count / max(max_bpm_count, 1)) * 20)
            bar = "\u2588" * bar_len
            lines.append(f"  {bucket:>3}-{bucket+4:<3} {bar} {count}")
        lines.append("")

    # Gaps
    if report["bpm_gaps"]:
        lines.append(f"BPM gaps (no tracks): {', '.join(f'{b}-{b+4}' for b in report['bpm_gaps'][:10])}")

    return "\n".join(lines)
