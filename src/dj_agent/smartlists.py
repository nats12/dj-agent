"""Smart playlist generation — boolean algebra rules engine.

Fills the gap between Serato Smart Crates (powerful rules) and Rekordbox
Intelligent Playlists (limited).  Rules operate on My Tag data and track
metadata to dynamically build playlists.
"""

from __future__ import annotations

import re
from typing import Any

from .types import TrackInfo


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

def parse_rule(rule_str: str) -> dict[str, Any]:
    """Parse a rule string into a structured filter.

    Supported syntax::

        genre:Techno
        genre:Techno AND energy:8+
        genre:Techno AND NOT vocal
        bpm:125-135
        key:8B OR key:9B
        genre:House OR genre:Techno AND energy:7+
        artist:Bicep

    Returns a dict with ``type`` ("and", "or", "not", "field") and children.
    """
    rule_str = rule_str.strip()

    # Strip outer parentheses: "(X)" → "X"
    while rule_str.startswith("(") and rule_str.endswith(")"):
        # Verify the parens actually match (not "(a) AND (b)")
        depth = 0
        matched = True
        for i, ch in enumerate(rule_str):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and i < len(rule_str) - 1:
                matched = False
                break
        if matched:
            rule_str = rule_str[1:-1].strip()
        else:
            break

    # Split by OR (lowest precedence)
    or_parts = _split_preserving_parens(rule_str, " OR ")
    if len(or_parts) > 1:
        return {"type": "or", "children": [parse_rule(p) for p in or_parts]}

    # Split by AND
    and_parts = _split_preserving_parens(rule_str, " AND ")
    if len(and_parts) > 1:
        return {"type": "and", "children": [parse_rule(p) for p in and_parts]}

    # NOT prefix
    if rule_str.upper().startswith("NOT "):
        return {"type": "not", "child": parse_rule(rule_str[4:])}

    # Field condition
    return _parse_field_condition(rule_str)


def _split_preserving_parens(text: str, delimiter: str) -> list[str]:
    """Split by delimiter but not inside parentheses."""
    parts: list[str] = []
    depth = 0
    current = ""
    i = 0
    while i < len(text):
        if text[i] == "(":
            depth += 1
            current += text[i]
        elif text[i] == ")":
            depth -= 1
            current += text[i]
        elif depth == 0 and text[i:].upper().startswith(delimiter.upper()):
            parts.append(current.strip())
            current = ""
            i += len(delimiter)
            continue
        else:
            current += text[i]
        i += 1
    if current.strip():
        parts.append(current.strip())
    return parts


def _parse_field_condition(expr: str) -> dict[str, Any]:
    """Parse a single field:value condition."""
    # Special: bare word like "vocal" or "instrumental"
    if ":" not in expr:
        return {"type": "field", "field": "tag", "op": "eq", "value": expr.strip().lower()}

    field, value = expr.split(":", 1)
    field = field.strip().lower()
    value = value.strip()

    # Range: bpm:125-135
    range_match = re.match(r"^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$", value)
    if range_match:
        return {
            "type": "field", "field": field, "op": "range",
            "lo": float(range_match.group(1)),
            "hi": float(range_match.group(2)),
        }

    # Threshold: energy:8+
    thresh_match = re.match(r"^(\d+)\+$", value)
    if thresh_match:
        return {
            "type": "field", "field": field, "op": "gte",
            "value": int(thresh_match.group(1)),
        }

    # Threshold: energy:8-
    thresh_match = re.match(r"^(\d+)-$", value)
    if thresh_match:
        return {
            "type": "field", "field": field, "op": "lte",
            "value": int(thresh_match.group(1)),
        }

    # Exact match
    return {"type": "field", "field": field, "op": "eq", "value": value}


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def matches_rule(
    track: TrackInfo,
    rule: dict[str, Any],
    tags: dict[str, Any] | None = None,
) -> bool:
    """Test whether a track matches a parsed rule.

    Parameters
    ----------
    track : TrackInfo
    rule : parsed rule dict from parse_rule()
    tags : optional dict of extra tags (energy, mood, vocal, etc.)
    """
    tags = tags or {}
    rtype = rule["type"]

    if rtype == "and":
        return all(matches_rule(track, c, tags) for c in rule["children"])
    if rtype == "or":
        return any(matches_rule(track, c, tags) for c in rule["children"])
    if rtype == "not":
        return not matches_rule(track, rule["child"], tags)

    # Field match
    field = rule["field"]
    op = rule["op"]

    val = _get_field_value(track, field, tags)

    if op == "eq":
        if val is None:
            return False
        return str(val).lower() == str(rule["value"]).lower()

    if op == "gte":
        try:
            return float(val or 0) >= rule["value"]
        except (ValueError, TypeError):
            return False

    if op == "lte":
        try:
            return float(val or 0) <= rule["value"]
        except (ValueError, TypeError):
            return False

    if op == "range":
        try:
            v = float(val or 0)
            return rule["lo"] <= v <= rule["hi"]
        except (ValueError, TypeError):
            return False

    return False


def filter_tracks(
    tracks: list[TrackInfo],
    rule_str: str,
    all_tags: dict[str, dict[str, Any]] | None = None,
) -> list[TrackInfo]:
    """Filter a list of tracks by a rule string.

    Parameters
    ----------
    tracks : list of TrackInfo
    rule_str : rule string (e.g., "genre:Techno AND energy:8+")
    all_tags : dict mapping content_id → tag dict
    """
    rule = parse_rule(rule_str)
    all_tags = all_tags or {}

    return [
        t for t in tracks
        if matches_rule(t, rule, all_tags.get(t.db_content_id, {}))
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field_value(
    track: TrackInfo,
    field: str,
    tags: dict[str, Any],
) -> Any:
    """Get a field value from a track or its tags."""
    # Direct TrackInfo fields
    field_map = {
        "genre": track.genre,
        "artist": track.artist,
        "title": track.title,
        "bpm": track.bpm,
        "key": track.key,
        "duration": track.duration,
        "bitrate": track.bitrate,
    }
    if field in field_map:
        return field_map[field]

    # Check tags dict (energy, mood, vocal, etc.)
    if field in tags:
        return tags[field]

    # Check for "tag" field (bare word matching)
    if field == "tag":
        # Check if the value appears in any tag
        return tags.get("classification", tags.get("mood", ""))

    return None
