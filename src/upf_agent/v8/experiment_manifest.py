"""Freeze dirty worktrees without Git mutations or dependencies on private corpora."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from upf_agent.v8.models import canonical, digest


def _git(project, *args):
    return subprocess.check_output(["git", *args], cwd=project)


def freeze_worktree(project: Path, output: Path, extra_files=()):
    project, output = project.resolve(), output.resolve()
    if output.exists():
        raise ValueError("freeze output already exists; choose a new experiment ID")
    tracked = set(_git(project, "ls-files", "-z").decode().split("\0")) - {""}
    untracked = set(
        _git(project, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    ) - {""}
    extra = set()
    for name in extra_files:
        path = project / name
        if not path.absolute().is_relative_to(project) or ".." in Path(name).parts:
            raise ValueError("extra freeze file must be inside the project")
        extra.add(str(path.relative_to(project)))
    names = sorted(tracked | untracked | extra)
    # Capture Git state before creating output files, including a backup of the staged index.
    git_state = {
        "git_head": _git(project, "rev-parse", "HEAD").decode().strip(),
        "status.txt": _git(project, "status", "--porcelain=v1", "--untracked-files=all"),
        "unstaged.patch": _git(project, "diff", "--binary"),
        "staged.patch": _git(project, "diff", "--cached", "--binary"),
    }
    output.mkdir(parents=True)
    files = {}
    for name in names:
        source = project / name
        if not source.exists() and not source.is_symlink():
            files[name] = {"kind": "deleted", "tracked": name in tracked}
            continue
        if not source.is_file() and not source.is_symlink():
            continue
        if source.is_relative_to(output):
            continue
        target = output / "worktree" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            link = os.readlink(source)
            target.symlink_to(link)
            body, kind = link.encode(), "symlink"
        else:
            body, kind = source.read_bytes(), "file"
            shutil.copy2(source, target)
            if target.read_bytes() != body:
                raise ValueError(f"file changed during freeze: {name}")
        files[name] = {
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "kind": kind,
            "tracked": name in tracked,
            "untracked": name in untracked,
        }
    for filename in ("status.txt", "unstaged.patch", "staged.patch"):
        (output / filename).write_bytes(git_state[filename])
    index = Path(_git(project, "rev-parse", "--git-path", "index").decode().strip())
    if not index.is_absolute():
        index = project / index
    if index.is_file():
        shutil.copy2(index, output / "git-index.backup")
    manifest = {
        "schema_version": "v8-next-freeze-1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_head": git_state["git_head"],
        "files": files,
        "worktree_content_sha256": digest(files),
        "python": platform.python_version(),
        "packages": {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def source_manifest(project: Path):
    files = [
        *sorted((project / "src/upf_agent").rglob("*.py")),
        project / "pyproject.toml",
        project / "uv.lock",
    ]
    hashes = {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file()
    }
    return {"files": hashes, "sha256": hashlib.sha256(canonical(hashes).encode()).hexdigest()}
