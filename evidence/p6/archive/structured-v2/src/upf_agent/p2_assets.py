"""Hash locally pinned OCR and Docling model assets for offline reproducibility."""

import hashlib
import json
from pathlib import Path


def _manifest(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    tree = hashlib.sha256()
    for file in sorted(x for x in path.rglob("*") if x.is_file()):
        relative = file.relative_to(path).as_posix()
        sha = hashlib.sha256(file.read_bytes()).hexdigest()
        files.append({"path": relative, "bytes": file.stat().st_size, "sha256": sha})
        tree.update(relative.encode())
        tree.update(bytes.fromhex(sha))
    if not files:
        raise ValueError(f"No files in asset directory: {path}")
    return {
        "path": str(path),
        "sha256_tree": tree.hexdigest(),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }


def write_manifest(root: Path, output: Path) -> dict:
    model_root = root.parent / "models"
    assets = {
        "ocr_detection": _manifest(model_root / "ocr/PP-OCRv5_server_det"),
        "ocr_recognition": _manifest(model_root / "ocr/PP-OCRv5_server_rec"),
        "docling": _manifest(model_root / "docling"),
    }
    manifest = {
        "assets": assets,
        "environments": {
            "ocr": "environment-lock/ocr.freeze.txt",
            "docling": "environment-lock/docling.freeze.txt",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        name: {"files": len(item["files"]), "bytes": item["total_bytes"]}
        for name, item in assets.items()
    }
