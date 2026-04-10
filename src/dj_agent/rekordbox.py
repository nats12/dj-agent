"""Rekordbox integration — process checks, DB backup, safe sessions."""

from __future__ import annotations

import platform
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from .config import RekordboxConfig


def is_rekordbox_running() -> bool:
    """Return *True* if a Rekordbox process is currently running.

    Supports macOS, Linux, and Windows.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["pgrep", "-x", "rekordbox"],
                capture_output=True,
            )
        elif system == "Windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq rekordbox.exe"],
                capture_output=True, text=True,
            )
            # tasklist always returns 0; case-insensitive check for mixed-case exe names
            return "rekordbox.exe" in (result.stdout or "").lower()
        else:  # Linux / WSL
            result = subprocess.run(
                ["pgrep", "-f", "rekordbox"],
                capture_output=True,
            )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def find_rekordbox_db() -> Path | None:
    """Locate the Rekordbox master.db on the current system."""
    candidates = [
        Path.home() / "Library" / "Pioneer" / "rekordbox" / "master.db",
        Path.home() / ".local" / "share" / "Pioneer" / "rekordbox" / "master.db",
        Path.home() / "AppData" / "Roaming" / "Pioneer" / "rekordbox" / "master.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def backup_database(db_path: Path | None = None) -> Path:
    """Create a timestamped backup of master.db.

    Returns the path to the backup file.
    """
    if db_path is None:
        db_path = find_rekordbox_db()
    if db_path is None or not db_path.exists():
        raise FileNotFoundError("Cannot find Rekordbox master.db to back up")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(f"master.db.backup.{stamp}")
    shutil.copy2(db_path, backup)
    return backup


@contextmanager
def safe_db_session(config: RekordboxConfig) -> Generator:
    """Context manager that guards Rekordbox DB writes.

    1. Checks that Rekordbox is not running (if configured).
    2. Creates a backup of master.db (if configured).
    3. Yields a ``pyrekordbox.Rekordbox6Database`` instance.
    4. Commits on success, rolls back on failure.

    Usage::

        with safe_db_session(config) as db:
            content = db.get_content(ID=some_id)
            content.Title = "Fixed Title"
            db.flush()
    """
    if config.check_process and is_rekordbox_running():
        raise RuntimeError(
            "Rekordbox is running.  Close it before writing to the database."
        )

    if config.backup_before_write:
        backup_database()

    from pyrekordbox import Rekordbox6Database  # type: ignore[import-untyped]

    db = Rekordbox6Database()

    # Attach a heartbeat checker so callers can verify mid-session
    db._dj_agent_check_lock = lambda: _check_lock_heartbeat(config)

    try:
        yield db
        # Re-check before commit — user may have opened Rekordbox during processing
        _check_lock_heartbeat(config)
        try:
            db.commit()
        except Exception:
            db.session.rollback()
            raise
    except BaseException:
        db.session.rollback()
        raise


def _check_lock_heartbeat(config: RekordboxConfig) -> None:
    """Re-check that Rekordbox hasn't started since the session opened.

    Call this before committing or between batch operations.
    """
    if config.check_process and is_rekordbox_running():
        raise RuntimeError(
            "Rekordbox was opened during the session! "
            "Rolling back to prevent database corruption. "
            "Close Rekordbox and try again."
        )
