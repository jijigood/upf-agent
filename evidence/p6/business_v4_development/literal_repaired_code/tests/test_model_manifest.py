"""Model manifest verification must detect a corrupted local file."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_model_manifest_detects_content_change(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "path": str(model_dir),
                        "files": [
                            {
                                "name": weight.name,
                                "size_bytes": weight.stat().st_size,
                                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                ]
            }
        )
    )
    script = Path(__file__).resolve().parents[1] / "tools" / "verify_model_manifest.py"
    good = subprocess.run([sys.executable, str(script), str(manifest)], capture_output=True)
    assert good.returncode == 0
    weight.write_bytes(b"tampered")
    bad = subprocess.run([sys.executable, str(script), str(manifest)], capture_output=True)
    assert bad.returncode == 1
    assert b"SHA-256 mismatch" in bad.stdout
