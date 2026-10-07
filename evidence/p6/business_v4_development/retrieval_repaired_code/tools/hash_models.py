"""Create a reproducible SHA-256 manifest for complete local model assets."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/models.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/p1/model-hashes.json"))
    parser.add_argument("--profiles", nargs="*", default=["baseline", "embedding", "reranker"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    profiles = {
        **config["generation"],
        "embedding": config["embedding"],
        "reranker": config["reranker"],
    }
    results = []
    for name in args.profiles:
        if name not in profiles:
            parser.error(f"unknown profile: {name}")
        entry = profiles[name]
        root = Path(entry["path"])
        index = root / "model.safetensors.index.json"
        if not index.is_file():
            parser.error(f"{name}: index missing; refusing partial assets")
        indexed = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(indexed.get("weight_map", {}).values()))
        if not shards or any(not (root / shard).is_file() for shard in shards):
            parser.error(f"{name}: indexed weight shards incomplete")
        files = sorted(
            [path for path in root.iterdir() if path.is_file() and path.name != "crc32.txt"]
        )
        results.append(
            {
                "profile": name,
                "model_id": entry["id"],
                "path": str(root),
                "indexed_shard_count": len(shards),
                "files": [
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in files
                ],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"created_utc": datetime.now(UTC).isoformat(), "models": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} for {', '.join(args.profiles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
