"""Library health report generation (read-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def generate_health_report(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a health report from a list of track dicts.

    Each track dict should have: path, artist, title, genre, bpm, key, bitrate.
    """
    total = len(tracks)
    missing = 0
    no_genre = 0
    no_key = 0
    no_artist = 0
    low_quality = 0
    weird_bpm = 0
    genres: dict[str, int] = {}
    bpms: list[float] = []

    for t in tracks:
        # File existence
        raw_path = t.get("path", "")
        if raw_path.startswith("file://"):
            raw_path = unquote(urlparse(raw_path).path)
        if not Path(raw_path).exists():
            missing += 1

        # Metadata completeness
        if not t.get("genre"):
            no_genre += 1
        else:
            g = t["genre"]
            genres[g] = genres.get(g, 0) + 1

        if not t.get("key"):
            no_key += 1
        if not t.get("artist"):
            no_artist += 1

        # Quality
        br = t.get("bitrate", 0)
        if br and br < 320:
            low_quality += 1

        # BPM
        bpm = t.get("bpm", 0)
        if bpm:
            bpms.append(bpm)
            if bpm < 60 or bpm > 200:
                weird_bpm += 1

    # Genre distribution (top 10)
    top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10]

    # BPM stats
    avg_bpm = sum(bpms) / len(bpms) if bpms else 0
    min_bpm = min(bpms) if bpms else 0
    max_bpm = max(bpms) if bpms else 0

    return {
        "total_tracks": total,
        "files_found": total - missing,
        "files_missing": missing,
        "missing_genre": no_genre,
        "missing_key": no_key,
        "missing_artist": no_artist,
        "below_320kbps": low_quality,
        "suspicious_bpm": weird_bpm,
        "top_genres": top_genres,
        "bpm_min": min_bpm,
        "bpm_max": max_bpm,
        "bpm_avg": round(avg_bpm, 1),
    }


def format_health_report(report: dict[str, Any]) -> str:
    """Format a health report dict as a human-readable string."""
    lines = [
        "=" * 50,
        f"  LIBRARY HEALTH REPORT — {report['total_tracks']} tracks",
        "=" * 50,
        "",
        f"Files:  {report['files_found']} found, {report['files_missing']} missing",
        f"Missing genre: {report['missing_genre']}",
        f"Missing key: {report['missing_key']} — analyse these in Rekordbox",
        f"Missing artist: {report['missing_artist']}",
        "",
        f"Below 320kbps: {report['below_320kbps']}",
        f"Suspicious BPM (<60 or >200): {report['suspicious_bpm']} — check in Rekordbox",
        "",
        "Top genres:",
    ]

    if report["top_genres"]:
        max_count = report["top_genres"][0][1]
        for genre, count in report["top_genres"]:
            bar_len = int((count / max(max_count, 1)) * 20)
            bar = "\u2588" * bar_len
            lines.append(f"   {genre:<25} {bar} {count}")
    else:
        lines.append("   (no genres found)")

    lines.extend([
        "",
        f"BPM range: {report['bpm_min']}-{report['bpm_max']}, Average: {report['bpm_avg']}",
    ])

    return "\n".join(lines)
