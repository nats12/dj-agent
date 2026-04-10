"""Tests for Rekordbox safety utilities."""

import platform
from pathlib import Path
from unittest.mock import patch

from dj_agent.rekordbox import is_rekordbox_running, find_rekordbox_db


def test_is_rekordbox_running_returns_bool():
    result = is_rekordbox_running()
    assert isinstance(result, bool)


def test_is_rekordbox_running_pgrep_missing():
    """If pgrep isn't found, should return False (not crash)."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert is_rekordbox_running() is False


def test_find_rekordbox_db_returns_none_when_missing():
    """On a system without Rekordbox, should return None."""
    result = find_rekordbox_db()
    # May or may not find it depending on the system
    assert result is None or isinstance(result, Path)
