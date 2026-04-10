"""Tests for memory system — atomic writes, backup, migration."""

import json
from pathlib import Path

from dj_agent.config import MemoryConfig
from dj_agent.memory import load_memory, save_memory, _empty


def test_load_creates_empty_when_missing(tmp_path: Path):
    config = MemoryConfig(path=str(tmp_path / "nonexistent.json"))
    data = load_memory(config)
    assert data["version"] == 2
    assert data["processed_tracks"] == {}


def test_save_and_reload(tmp_path: Path):
    config = MemoryConfig(path=str(tmp_path / "mem.json"), auto_backup=False)
    data = _empty()
    data["processed_tracks"]["abc"] = {"test": True}
    save_memory(data, config)

    loaded = load_memory(config)
    assert loaded["processed_tracks"]["abc"]["test"] is True


def test_backup_rotation(tmp_path: Path):
    config = MemoryConfig(
        path=str(tmp_path / "mem.json"),
        backup_count=3,
        auto_backup=True,
    )
    # Save 4 times → should create mem.json.1, .2, .3
    for i in range(4):
        data = _empty()
        data["processed_tracks"][f"track_{i}"] = {"n": i}
        save_memory(data, config)

    assert (tmp_path / "mem.json").exists()
    assert (tmp_path / "mem.json.1").exists()
    assert (tmp_path / "mem.json.2").exists()
    assert (tmp_path / "mem.json.3").exists()


def test_v1_migration(v1_memory: Path, tmp_path: Path):
    config = MemoryConfig(path=str(v1_memory), auto_backup=False)
    data = load_memory(config)
    assert data["version"] == 2
    # The old key "abc123" should still be present (file doesn't exist,
    # so it can't be re-hashed) with needs_rehash flag.
    found = False
    for key, entry in data["processed_tracks"].items():
        if entry.get("needs_rehash"):
            found = True
    assert found, "v1 entries with missing files should have needs_rehash=True"


def test_atomic_write_leaves_no_tmp(tmp_path: Path):
    config = MemoryConfig(path=str(tmp_path / "mem.json"), auto_backup=False)
    save_memory(_empty(), config)
    # The .tmp file should have been renamed away
    assert not (tmp_path / "mem.json.tmp").exists()
    assert (tmp_path / "mem.json").exists()
