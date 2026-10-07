"""Compare hashed weight files with Hugging Face local download metadata."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/p1/hf-etag-check.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    models: list[dict[str, object]] = []
    all_pass = True
    for model in manifest["models"]:
        root = Path(model["path"])
        revisions: set[str] = set()
        checked = 0
        problems: list[str] = []
        for file in model["files"]:
            if not file["name"].endswith(".safetensors"):
                continue
            metadata_path = root / ".cache/huggingface/download" / (file["name"] + ".metadata")
            if not metadata_path.is_file():
                problems.append(f"metadata missing: {file['name']}")
                continue
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if len(lines) < 2:
                problems.append(f"metadata malformed: {file['name']}")
                continue
            revision, etag = lines[:2]
            revisions.add(revision)
            if len(etag) != 64 or etag.lower() != file["sha256"]:
                problems.append(f"etag/SHA-256 mismatch: {file['name']}")
            else:
                checked += 1
        if len(revisions) != 1:
            problems.append(f"expected one repository revision, got {sorted(revisions)}")
        passed = not problems
        all_pass &= passed
        models.append(
            {
                "profile": model["profile"],
                "pass": passed,
                "checked_weight_files": checked,
                "repository_revisions": sorted(revisions),
                "problems": problems,
            }
        )
    report = {"checked_utc": datetime.now(UTC).isoformat(), "pass": all_pass, "models": models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
