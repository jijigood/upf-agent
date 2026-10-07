"""Verify every file recorded by a local model SHA-256 manifest."""

import argparse
import json
from pathlib import Path

from hash_models import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checked = 0
    failures: list[str] = []
    for model in manifest["models"]:
        root = Path(model["path"])
        for item in model["files"]:
            path = root / item["name"]
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            if path.stat().st_size != item["size_bytes"]:
                failures.append(f"size mismatch: {path}")
                continue
            if sha256_file(path) != item["sha256"]:
                failures.append(f"SHA-256 mismatch: {path}")
                continue
            checked += 1
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {checked} files across {len(manifest['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
