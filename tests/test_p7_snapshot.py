import json
import sqlite3
from pathlib import Path
from runpy import run_path

import pytest

SCRIPT = run_path(str(Path(__file__).resolve().parents[1] / "scripts/p7_snapshot.py"))


def test_snapshot_captures_committed_wal_and_rejects_corrupt_restore(tmp_path):
    root = tmp_path / "project"
    db_path = root / "data/p5/tasks.sqlite3"
    db_path.parent.mkdir(parents=True)
    (root / "config").mkdir()
    (root / "config/models.yaml").write_text("local configuration")
    (root / "uv.lock").write_text("version = 1")
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE tasks(id TEXT)")
        db.execute("INSERT INTO tasks VALUES('committed')")
        db.commit()
        package = tmp_path / "snapshot"
        manifest = SCRIPT["snapshot"](root, package)
        assert manifest["sqlite"]["data/p5/tasks.sqlite3"]["integrity"] == "ok"
        assert "uv.lock" in manifest["files"]
        assert not any(name.endswith(("-wal", "-shm")) for name in manifest["files"])
    target = tmp_path / "restore"
    SCRIPT["restore"](package, target)
    with sqlite3.connect(target / "data/p5/tasks.sqlite3") as db:
        assert db.execute("SELECT id FROM tasks").fetchone()[0] == "committed"
    (package / "payload/config/models.yaml").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        SCRIPT["restore"](package, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_snapshot_refuses_recursive_output_and_symlink_sources(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        SCRIPT["snapshot"](root, root / "nested")
    outside = tmp_path / "outside"
    outside.write_text("external")
    (root / "config/models.yaml").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        SCRIPT["snapshot"](root, tmp_path / "snapshot")


def test_snapshot_manifest_cannot_escape_payload(tmp_path):
    package = tmp_path / "package"
    (package / "payload").mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({"files": {"../outside": {}}}))
    with pytest.raises(ValueError):
        SCRIPT["verify"](package)


def test_snapshot_rejects_symlinked_payload_parent(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "config/models.yaml").write_text("local")
    package = tmp_path / "snapshot"
    SCRIPT["snapshot"](root, package)
    config = package / "payload/config"
    external = tmp_path / "external"
    config.rename(external)
    config.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        SCRIPT["restore"](package, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()
