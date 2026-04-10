"""Multi-source metadata enrichment.

Adapted from AI-Music-Library-Normalization-Suite metadata_enrichment.py.
Sources: MusicBrainz, Last.fm, Discogs.
Spotify Audio Features removed (deprecated Nov 2024).
AcousticBrainz removed (shut down 2022).
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import musicbrainzngs  # type: ignore[import-untyped]
    _MB_AVAILABLE = True
except ImportError:
    _MB_AVAILABLE = False

try:
    import pylast  # type: ignore[import-untyped]
    _LASTFM_AVAILABLE = True
except ImportError:
    _LASTFM_AVAILABLE = False

try:
    import discogs_client  # type: ignore[import-untyped]
    _DISCOGS_AVAILABLE = True
except ImportError:
    _DISCOGS_AVAILABLE = False


class MetadataEnricher:
    """Query multiple online sources to fill genre, mood, style, year, label.

    Each source is independently optional and degrades gracefully.
    Results are cached in memory.json to avoid repeated API calls.
    """

    def __init__(
        self,
        lastfm_api_key: str | None = None,
        discogs_user_token: str | None = None,
        rate_limit_sec: float = 1.0,
    ):
        self._rate_limit = rate_limit_sec
        self._last_request: float = 0.0

        # MusicBrainz
        if _MB_AVAILABLE:
            musicbrainzngs.set_useragent("dj-agent", "0.2.0", "https://github.com/nats12/dj-agent")
            musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)

        # Last.fm
        self._lastfm_key = lastfm_api_key or os.getenv("LASTFM_API_KEY")
        self._lastfm: Any = None
        if _LASTFM_AVAILABLE and self._lastfm_key:
            self._lastfm = pylast.LastFMNetwork(api_key=self._lastfm_key)

        # Discogs
        self._discogs_token = discogs_user_token or os.getenv("DISCOGS_USER_TOKEN")
        self._discogs: Any = None
        if _DISCOGS_AVAILABLE and self._discogs_token:
            self._discogs = discogs_client.Client(
                "dj-agent/0.2.0", user_token=self._discogs_token,
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def enrich(
        self,
        artist: str,
        title: str,
        album: str | None = None,
    ) -> dict[str, Any]:
        """Query all configured sources and merge results."""
        result: dict[str, Any] = {
            "artist": artist,
            "title": title,
            "genre_tags": [],
            "mood_tags": [],
            "style_tags": [],
            "year": None,
            "label": None,
            "musicbrainz_id": None,
            "sources": [],
        }

        # 1. MusicBrainz
        mbid = self._musicbrainz_lookup(artist, title, album)
        if mbid:
            result["musicbrainz_id"] = mbid
            result["sources"].append("musicbrainz")

        # 2. Last.fm tags
        lastfm = self._lastfm_lookup(artist, title)
        if lastfm:
            result["genre_tags"].extend(lastfm.get("genre_tags", []))
            result["mood_tags"].extend(lastfm.get("mood_tags", []))
            result["sources"].append("lastfm")

        # 3. Discogs
        discogs = self._discogs_lookup(artist, title)
        if discogs:
            result["genre_tags"].extend(discogs.get("genres", []))
            result["style_tags"].extend(discogs.get("styles", []))
            if discogs.get("year"):
                result["year"] = discogs["year"]
            if discogs.get("label"):
                result["label"] = discogs["label"]
            result["sources"].append("discogs")

        # Deduplicate
        result["genre_tags"] = list(dict.fromkeys(result["genre_tags"]))
        result["mood_tags"] = list(dict.fromkeys(result["mood_tags"]))
        result["style_tags"] = list(dict.fromkeys(result["style_tags"]))

        return result

    # ------------------------------------------------------------------
    # MusicBrainz
    # ------------------------------------------------------------------

    def _musicbrainz_lookup(
        self, artist: str, title: str, album: str | None,
    ) -> str | None:
        if not _MB_AVAILABLE:
            return None
        try:
            self._throttle()
            res = musicbrainzngs.search_recordings(
                artist=artist, recording=title,
                release=album or "", limit=3, strict=False,
            )
            recs = res.get("recording-list", [])
            return recs[0]["id"] if recs else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Last.fm
    # ------------------------------------------------------------------

    _GENRE_KEYWORDS = frozenset({
        "electronic", "techno", "house", "trance", "dubstep", "drum and bass",
        "dnb", "ambient", "dance", "edm", "electronica", "idm", "breaks",
        "garage", "progressive", "disco", "acid", "minimal", "deep house",
    })

    _MOOD_KEYWORDS = frozenset({
        "dark", "chill", "energetic", "happy", "sad", "aggressive",
        "euphoric", "melancholic", "uplifting", "atmospheric", "hypnotic",
    })

    def _lastfm_lookup(self, artist: str, title: str) -> dict[str, Any] | None:
        if not self._lastfm:
            return None
        try:
            self._throttle()
            track = self._lastfm.get_track(artist, title)
            top_tags = track.get_top_tags(limit=20)
            all_tags = [t.item.name.lower() for t in top_tags]

            genre_tags = [t for t in all_tags if any(k in t for k in self._GENRE_KEYWORDS)]
            mood_tags = [t for t in all_tags if any(k in t for k in self._MOOD_KEYWORDS)]

            return {"genre_tags": genre_tags[:5], "mood_tags": mood_tags[:3]}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Discogs
    # ------------------------------------------------------------------

    def _discogs_lookup(self, artist: str, title: str) -> dict[str, Any] | None:
        if not self._discogs:
            return None
        try:
            self._throttle()
            results = self._discogs.search(f"{artist} {title}", type="release")
            if not results:
                return None
            release = results[0]
            return {
                "genres": getattr(release, "genres", []) or [],
                "styles": getattr(release, "styles", []) or [],
                "year": getattr(release, "year", None),
                "label": (
                    release.labels[0].name
                    if hasattr(release, "labels") and release.labels
                    else None
                ),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.monotonic()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def enrich_track_metadata(
    artist: str,
    title: str,
    album: str | None = None,
) -> dict[str, Any]:
    """One-shot enrichment with default config."""
    return MetadataEnricher().enrich(artist, title, album)
