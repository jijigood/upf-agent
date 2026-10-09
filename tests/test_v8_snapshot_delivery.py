import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from upf_agent.v8.store import Ledger

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "v8_snapshot_delivery", PROJECT / "scripts/v8_snapshot.py"
)
snapshot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot_module)


def test_snapshot_sqlite_tools_private_config_and_restore(tmp_path):
    root = tmp_path / "runtime"
    (root / "data").mkdir(parents=True)
    (root / "config").mkdir()
    db_file = root / "data/example.sqlite3"
    with sqlite3.connect(db_file) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE receipt(value TEXT)")
        connection.execute("INSERT INTO receipt VALUES('committed')")
        connection.commit()
        (root / "config/v8-neo4j.json").write_text('{"password":"private-test"}')
        backup = tmp_path / "snapshot"
        snapshot_module.snapshot(root, backup)
    manifest = json.loads((backup / "manifest.json").read_text())
    assert "project/tools/retrieval_worker.py" in manifest["files"]
    assert "project/config/retrieval.yaml" in manifest["files"]
    assert all("v8-neo4j.json" not in name for name in manifest["files"])
    restored = tmp_path / "restored"
    snapshot_module.restore(backup, restored)
    with sqlite3.connect(restored / "data/example.sqlite3") as connection:
        assert connection.execute("SELECT value FROM receipt").fetchone() == ("committed",)
    assert (restored / "tools").resolve() == PROJECT / "tools"
    with pytest.raises(ValueError, match="existing runtime"):
        snapshot_module.restore(backup, restored)
    # Check all source hashes before creating any new restore destination.
    (backup / "runtime/data/example.sqlite3").write_bytes(b"tampered")
    second = tmp_path / "rejected"
    with pytest.raises(ValueError, match="hash mismatch"):
        snapshot_module.restore(backup, second)
    assert not second.exists()


def test_snapshot_refuses_active_runtime_and_nested_destination(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = Ledger(root)
    try:
        with ledger.execution_lock("f" * 32):
            with pytest.raises(ValueError, match="active run"):
                snapshot_module.snapshot(root, tmp_path / "blocked")
        assert not (tmp_path / "blocked").exists()
        with pytest.raises(ValueError, match="outside the runtime"):
            snapshot_module.snapshot(root, root / "data/recursive-copy")
        assert not (root / "data/recursive-copy").exists()
    finally:
        ledger.close()
