"""Create and verify a same-path data snapshot using consistent SQLite backups."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path

DIRECTORIES = (
    "src",
    "tools",
    "scripts",
    "tests",
    "docs",
    "config",
    "environment-lock",
    "examples",
    "eval",
    "evidence",
    "runs",
    "data/raw",
    "data/p2",
    "data/p3",
    "data/p4/logs",
    "data/p4/templates",
    "data/p4/topology",
    "data/p4/fixtures",
    "data/p5",
)
FILES = ("pyproject.toml", "uv.lock", ".python-version", "README.md")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sqlite_file(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(16) == b"SQLite format 3\x00"


def ignored(path: Path) -> bool:
    if path.parts[:2] == ("evidence", "p7"):
        return True
    return (
        any(
            part.startswith((".building-", ".venv"))
            or part in {"__pycache__", ".pytest_cache", ".ruff_cache", "locks"}
            for part in path.parts
        )
        or path.name.endswith((".pyc", "-wal", "-shm"))
        or (path.name.endswith(".lock") and path.name != "uv.lock")
    )


def snapshot(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Snapshot needs a fresh directory outside the source project")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=output.parent))
    payload = stage / "payload"
    records, databases, aliases = {}, {}, {}
    locks = root / "data/p5/locks"
    lock_paths = set(locks.glob("*.lock"))
    try:
        with ExitStack() as held:
            for path in sorted(lock_paths):
                lock = held.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            paths = {root / name for name in FILES if (root / name).is_file()}
            for name in DIRECTORIES:
                paths.update(p for p in (root / name).rglob("*") if p.is_file())
            for source in sorted(paths):
                relative = source.relative_to(root)
                if ignored(relative):
                    continue
                is_alias = source.is_symlink() or any(
                    parent.is_symlink() for parent in source.parents if parent != root
                )
                if is_alias:
                    resolved = source.resolve(strict=True)
                    if not resolved.is_relative_to(root):
                        raise ValueError(f"Snapshot refuses external symlink: {relative}")
                    aliases[str(relative)] = str(resolved.relative_to(root))
                destination = payload / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if sqlite_file(source):
                    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as live:
                        with closing(sqlite3.connect(destination)) as copy:
                            live.backup(copy)
                            copy.execute("PRAGMA journal_mode=DELETE")
                            integrity = copy.execute("PRAGMA integrity_check").fetchone()[0]
                            if integrity != "ok":
                                raise ValueError(f"SQLite snapshot integrity failed: {relative}")
                            tables = [
                                row[0]
                                for row in copy.execute(
                                    "SELECT name FROM sqlite_master "
                                    "WHERE type='table' ORDER BY name"
                                )
                            ]
                            databases[str(relative)] = {"integrity": integrity, "tables": tables}
                else:
                    before = sha(source)
                    shutil.copy2(source, destination)
                    if before != sha(source) or before != sha(destination):
                        raise ValueError(f"Source changed during snapshot: {relative}")
                records[str(relative)] = {
                    "sha256": sha(destination),
                    "bytes": destination.stat().st_size,
                }
            if lock_paths != set(locks.glob("*.lock")):
                raise ValueError("New agent task started during snapshot; take a new snapshot")
        manifest = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_root": str(root),
            "files": records,
            "sqlite": databases,
            "flattened_internal_aliases": aliases,
            "task_locks_held": len(lock_paths),
            "excluded": [
                "Python environments",
                "model weights",
                "synthetic capacity dataset",
                "live evidence/p7 deployment/service logs",
            ],
            "path_policy": (
                "restore at the same configured absolute project/model paths; "
                "relocation is not verified"
            ),
            "quiescence_requirement": (
                "stop submitting new tasks during snapshot; leave model service running"
            ),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        stage.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


def verify(snapshot_root: Path) -> dict:
    if (snapshot_root / "payload").is_symlink():
        raise ValueError("Snapshot payload must not be a symlink")
    manifest = json.loads((snapshot_root / "manifest.json").read_text())
    base = (snapshot_root / "payload").resolve()
    expected = set(manifest["files"])
    actual = {str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("Snapshot file set differs from manifest")
    for name, record in manifest["files"].items():
        relative = Path(name)
        path = base / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents if parent != base)
            or not path.resolve().is_relative_to(base)
        ):
            raise ValueError(f"Unsafe snapshot entry: {name}")
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
    return manifest


def restore(snapshot_root: Path, output: Path) -> dict:
    manifest = verify(snapshot_root)
    if output.exists() or output.resolve().is_relative_to(snapshot_root.resolve()):
        raise ValueError("Restore needs a fresh destination; existing data is never overwritten")
    shutil.copytree(snapshot_root / "payload", output)
    return {"restored_files": len(manifest["files"]), "output": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["create", "verify", "restore"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "verify":
        result = {"verified_files": len(verify(args.root)["files"])}
    elif args.output is None:
        parser.error("create and restore require --output")
    elif args.operation == "create":
        manifest = snapshot(args.root, args.output)
        result = {"files": len(manifest["files"]), "sqlite": len(manifest["sqlite"])}
    else:
        result = restore(args.root, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
