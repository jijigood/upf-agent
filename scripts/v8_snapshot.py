"""Inspectably back up/restore v8 data and checkpoints; never overwrite a live root."""

import argparse
import fcntl
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        sqlite = stream.read(16) == b"SQLite format 3\x00"
    if sqlite:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
            with sqlite3.connect(target) as dst:
                src.backup(dst)
    else:
        shutil.copy2(source, target)


def snapshot(root: Path, destination: Path):
    root, destination = root.resolve(strict=True), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError("snapshot destination must be outside the runtime")
    locks = root / "data/v8/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / "snapshot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("runtime has an active run; snapshot requires quiescence") from error
        return copy_snapshot(root, destination)


def copy_snapshot(root: Path, destination: Path):
    if destination.exists():
        raise ValueError("destination already exists")
    destination.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    files = {}
    for prefix, base in [("runtime", root), ("project", project)]:
        directories = (
            ["config", "data", "evidence", "runs", "examples", "experiments"]
            if prefix == "runtime"
            else [
                "src",
                "scripts",
                "tools",
                "config",
                "tests",
                "examples/v8",
                "requirements",
                "docs/v8",
                "docs/v8-preparation",
            ]
        )
        paths = [p for name in directories for p in (base / name).rglob("*") if p.is_file()]
        if prefix == "project":
            paths.extend(
                base / name for name in ["pyproject.toml", "uv.lock", "README_v8.md", ".gitignore"]
            )
        for path in sorted(paths):
            if (
                path.is_symlink()
                or "__pycache__" in path.parts
                or "locks" in path.parts
                or path.name.endswith(("-wal", "-shm"))
                or path.name == "v8-neo4j.json"
            ):
                continue
            relative = Path(prefix) / path.relative_to(base)
            target = destination / relative
            copy_file(path, target)
            files[str(relative)] = sha(target)
    manifest = {
        "format_version": 1,
        "files": files,
        "runtime_root": str(root),
        "sqlite_backup_api": True,
        "external_dependencies": [
            "shared model weight directories and recorded SHA256 manifest",
            "local generation service",
            "Python 3.11 and locked environments",
            "same-host retrieval environment and recorded native absolute paths",
            "dedicated Neo4j credentials/container (not bundled)",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "files": len(files),
                "manifest_sha256": sha(destination / "manifest.json"),
            }
        )
    )


def restore(source: Path, destination: Path):
    source, destination = source.resolve(strict=True), destination.resolve()
    if destination.exists():
        raise ValueError("refusing to overwrite an existing runtime")
    manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid snapshot path: {name}")
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source.resolve()) or sha(path) != expected:
            raise ValueError(f"snapshot hash mismatch: {name}")
    shutil.copytree(source / "runtime", destination)
    project = Path(__file__).resolve().parents[1]
    for name in ["tools", ".venv-retrieval"]:
        (destination / name).symlink_to(project / name, target_is_directory=True)
    print(json.dumps({"restored": str(destination), "verified_files": len(manifest["files"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["snapshot", "restore"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.operation == "snapshot":
        snapshot(args.source, args.destination)
    else:
        restore(args.source, args.destination)
