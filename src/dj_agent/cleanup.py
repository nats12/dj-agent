"""Title / artist metadata cleanup.

Fixes over the original:
- Rule 10 (years): won't eat legitimate years like "Levels 2013"
- Rule 11 (codes): requires 2 uppercase + 4+ digits, not the overly broad pattern
- Artist splitting: requires spaces around delimiter, handles hyphenated artists
- feat. / ft. / vs. / b2b parsing
"""

from __future__ import annotations

import html
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Known hyphenated artists (extensible via config)
# ---------------------------------------------------------------------------

DEFAULT_HYPHENATED_ARTISTS: set[str] = {
    "jay-z", "k-klass", "mk-ultra", "x-press 2", "t-bone",
    "a-trak", "sub-zero", "k-os", "j-kwon", "ki/ki",
    "will-i-am", "deadmau5",  # not hyphenated but included for safety
}


# ---------------------------------------------------------------------------
# Title cleanup
# ---------------------------------------------------------------------------

def normalize_for_split(raw: str) -> str:
    """Pre-split normalization of a raw filename stem.

    Only converts filename-style underscores to spaces so that ``" - "``
    delimiters become visible to :func:`split_artist_from_title`. Does NOT
    apply title-specific rules (year stripping, store IDs, watermarks);
    those must run after the split on the title half only, otherwise they
    damage short numeric titles like ``"Binary Finary - 1998"``.
    """
    if raw.count("_") >= 3 and "-" in raw:
        # Keep dashes literal; convert underscores to spaces.
        normalized = raw.replace("_", " ")
        normalized = re.sub(r"  +", " ", normalized).strip(" -")
        return normalized
    return raw


def cleanup_artist(artist: str) -> tuple[str, list[str]]:
    """Minimal artist-side cleanup: HTML entities and whitespace only.

    Title-specific rules (trailing year, store IDs, watermarks, MASTER suffix,
    etc.) must not run on the artist half — they'd mangle legitimate artist
    names. Keep this intentionally narrow.
    """
    changes: list[str] = []
    decoded = html.unescape(artist)
    if decoded != artist:
        changes.append("decoded HTML entities in artist")
        artist = decoded
    stripped = re.sub(r"  +", " ", artist.strip())
    if stripped != artist:
        changes.append("trimmed artist whitespace")
    return stripped, changes


def cleanup_title(title: str, artist: str = "") -> tuple[str, list[str]]:
    """Clean a track title, returning ``(cleaned, list_of_changes)``."""
    original = title
    changes: list[str] = []

    # 1. Decode HTML entities
    decoded = html.unescape(title)
    if decoded != title:
        changes.append("decoded HTML entities")
        title = decoded

    # 2. Strip whitespace
    title = title.strip()

    # 3. Collapse double spaces
    title = re.sub(r"  +", " ", title)

    # 3b. Remove trailing "_pn" (Platinum Notes-mastered duplicate marker).
    # Must run BEFORE the underscore-conversion rule below, otherwise
    # "_MST_V4_pn" gets converted to " MST V4 pn" and the trailing "pn"
    # is no longer an underscore-delimited suffix we can safely strip.
    cleaned = re.sub(r"_pn\s*$", "", title).strip()
    if cleaned != title:
        changes.append("removed _pn suffix")
        title = cleaned

    # 4. Remove website watermarks
    watermark_patterns = [
        r"\s*-?\s*www\.\S+",
        r"\s*\[?\w+\.com\]?\s*$",
        r"\s*\((?:promodj|djsoundtop|electronicfresh|zipdj)\.\w+\)",
        r"\s+djsoundtop\.\w+",
        r"\s+electronicfresh\.\w+",
        r"\s*#VKUSMUZ\s*",
    ]
    for pat in watermark_patterns:
        cleaned = re.sub(pat, "", title, flags=re.IGNORECASE).strip()
        if cleaned != title:
            changes.append("removed website watermark")
            title = cleaned

    # 5. Remove file extensions
    title = re.sub(
        r"\.(mp3|flac|wav|aiff|m4a|aac|ogg)$", "", title, flags=re.IGNORECASE
    ).strip()

    # 6. Remove trailing " - KEY - BPM"
    title = re.sub(r"\s*-\s*\d{1,2}[AB]\s*-\s*\d{2,3}\s*$", "", title).strip()

    # 7. Remove trailing Beatport/store IDs (8+ digits)
    title = re.sub(r"-\d{8,}\s*$", "", title).strip()

    # 8. Convert filename-style underscores (3+ underscores present)
    if title.count("_") >= 3 and "-" in title:
        parts = title.split("-")
        cleaned_parts = [p.strip().replace("_", " ").strip() for p in parts]
        cleaned_parts = [p for p in cleaned_parts if not re.match(r"^\d{6,}$", p)]
        if (
            artist
            and cleaned_parts
            and cleaned_parts[0].lower().replace(" ", "")
            == artist.lower().replace(" ", "")
        ):
            cleaned_parts = cleaned_parts[1:]
        title = " - ".join(cleaned_parts).strip(" -")

    # 9. Remove [PRO FRONT] etc.
    title = re.sub(r"\[PRO FRONT\]\s*", "", title, flags=re.IGNORECASE).strip()

    # 10. Remove trailing bare year — ONLY if safe
    bare_year = re.search(r"(?<!\()[ ]((?:19|20)\d{2})\s*$", title)
    if bare_year and len(title) > 6:
        before = title[: bare_year.start()].strip()
        if before and not before.lower().endswith(("of", "in", "from", "since", "class")):
            changes.append(f"removed trailing year {bare_year.group(1)}")
            title = before

    # 11. Remove store IDs (exactly 2 uppercase letters + 4+ digits)
    store_id = re.search(r"\s+[A-Z]{2}\d{4,}\s*$", title)
    if store_id:
        changes.append("removed store ID")
        title = title[: store_id.start()].strip()

    # Remove standalone MASTER at end (not part of a word like "Grandmaster")
    cleaned = re.sub(r"(?<=\s)MASTER\s*$", "", title).strip()
    if cleaned != title:
        changes.append("removed MASTER suffix")
        title = cleaned

    # Remove Free DL
    cleaned = re.sub(r"\s+Free\s+DL\s*$", "", title, flags=re.IGNORECASE).strip()
    if cleaned != title:
        changes.append("removed Free DL")
        title = cleaned

    # 12. Final trim
    title = title.strip(" -\u2013")

    return title, changes


# ---------------------------------------------------------------------------
# Smart title case
# ---------------------------------------------------------------------------

_LOWERCASE = frozenset({
    "a", "an", "the", "and", "but", "or", "nor", "for",
    "in", "on", "at", "to", "of", "by", "vs", "x",
    "feat", "feat.", "ft", "ft.",
})

_UPPERCASE = frozenset({
    "dj", "sos", "uk", "usa", "id", "vip", "ep", "lp",
    "og", "hd", "bpm", "ok", "tv", "ii", "iii", "iv",
    "nyc", "la", "sf", "dc",
})


def smart_title_case(title: str) -> str:
    """Apply title-casing that respects DJ-music conventions."""

    def case_word(word: str, is_first: bool) -> str:
        lower = word.lower()
        if "/" in word:
            return word
        if lower in _UPPERCASE:
            return word.upper()
        if lower in _LOWERCASE and not is_first:
            return lower
        if word:
            return word[0].upper() + word[1:]
        return word

    words = title.split(" ")
    return " ".join(case_word(w, i == 0) for i, w in enumerate(words))


# ---------------------------------------------------------------------------
# Artist / title splitting
# ---------------------------------------------------------------------------

def split_artist_from_title(
    title: str,
    existing_artist: str = "",
    known_artists: set[str] | None = None,
    hyphenated_artists: set[str] | None = None,
) -> tuple[Optional[str], str]:
    """Try to extract artist from a combined "Artist - Title" string.

    Returns ``(artist, title)`` where artist is ``None`` if no split was found.
    """
    if existing_artist.strip():
        return None, title

    hyphens = hyphenated_artists or DEFAULT_HYPHENATED_ARTISTS

    # Check known hyphenated artists first
    title_lower = title.lower()
    for ha in sorted(hyphens, key=len, reverse=True):
        for sep in (" - ", " \u2013 "):
            prefix = ha + sep
            if title_lower.startswith(prefix.lower()):
                artist_part = title[: len(ha)]
                title_part = title[len(prefix) :].strip()
                if title_part:
                    return artist_part, title_part

    # Require SPACES around the delimiter (prevents splitting "Jay-Z")
    match = re.match(r"^(.+?)\s+[-\u2013]\s+(.+)$", title)
    if match:
        potential_artist = match.group(1).strip()
        potential_title = match.group(2).strip()
        if (
            len(potential_artist) < 80
            and potential_title
            and not re.search(
                r"(remix|edit|mix|version|bootleg|rework)",
                potential_artist,
                re.IGNORECASE,
            )
        ):
            return potential_artist, potential_title

    # Pass 2: known-artist prefix matching
    if known_artists:
        for artist in sorted(known_artists, key=len, reverse=True):
            if not artist or len(artist) < 3:
                continue
            if title_lower.startswith(artist.lower() + " "):
                remaining = title[len(artist) :].lstrip(" -\u2013").strip()
                if remaining:
                    return artist, remaining

    return None, title


_KNOWN_GROUPS: set[str] = {
    "above & beyond", "chase & status", "aly & fila", "simon & garfunkel",
    "hall & oates", "brooks & dunn", "flogging molly", "bob & weave",
    "gorgon city", "disclosure",  # not duos but often confused
}


def extract_featured_artists(artist: str) -> tuple[str, list[str]]:
    """Split "Main Artist feat. Guest" into (main, [guests]).

    Handles: feat., ft., vs., b2b, &, x (as collaboration separator).
    Protects known duo/group names from being incorrectly split.
    """
    # Protect known duos/groups
    if artist.lower().strip() in _KNOWN_GROUPS:
        return artist, []

    # Patterns that denote a featured / collaborating artist
    feat_patterns = [
        r"\s+feat\.?\s+",
        r"\s+ft\.?\s+",
        r"\s+featuring\s+",
        r"\s+vs\.?\s+",
        r"\s+b2b\s+",
        r"\s+&\s+",
        r"\s+x\s+",  # "Artist x Artist" collab notation
    ]
    for pat in feat_patterns:
        match = re.split(pat, artist, maxsplit=1, flags=re.IGNORECASE)
        if len(match) == 2:
            return match[0].strip(), [match[1].strip()]

    return artist, []
