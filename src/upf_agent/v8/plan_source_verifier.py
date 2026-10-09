"""Independent source-to-binding proof for structured pilot observations."""

import hashlib
import json
from pathlib import Path

from upf_agent.contracts import TimeWindow
from upf_agent.v8.adapters import source_path, verify_binding
from upf_agent.v8.models import canonical


def verify_plan_source(root: Path, binding, task):
    if not verify_binding(root, binding):
        return False
    path = source_path(root, binding.source.relative_path)
    if path.suffix != ".json":
        # Non-JSON manuals require the pinned chunk archive to certify source extraction.
        if binding.category != "K":
            return False
        try:
            from upf_agent.p3_index import load_active_index, load_chunks

            directory, manifest = load_active_index(root)
            if (
                hashlib.sha256((directory / "chunks.jsonl").read_bytes()).hexdigest()
                != manifest["chunks_sha256"]
            ):
                return False
            chunk = load_chunks(directory).get(binding.native_id)
            return bool(
                chunk
                and chunk.source_sha256 == binding.source.sha256
                and chunk.source_id == binding.source.source_id
                and chunk.product == binding.product
                and chunk.version == binding.version
                and binding.source.relative_path == f"data/raw/{chunk.relative_path}"
                and chunk.text == binding.fields.get("instruction")
                and binding.source.locator
                in {
                    ";".join(chunk.locators) or chunk.chunk_id,
                    ", ".join(chunk.locators) or chunk.chunk_id,
                }
            )
        except (OSError, ValueError, KeyError):
            return False
    try:
        body = json.loads(path.read_text())
        record = body
        for component in binding.source.locator.strip("/").split("/"):
            component = component.replace("~1", "/").replace("~0", "~")
            if isinstance(record, list):
                if (
                    not component.isascii()
                    or not component.isdigit()
                    or str(int(component)) != component
                ):
                    return False
                record = record[int(component)]
            else:
                record = record[component]
        if not isinstance(record, dict):
            return False
        if (body.get("synthetic") is True) != binding.synthetic:
            return False
        if (body.get("corroborated") is True) != binding.corroborated:
            return False
        if (
            binding.category in {"R", "P"}
            and record.get("config_version") != binding.config_version
        ):
            return False
        obj = task.aliases.get(record.get("object_id"), record.get("object_id"))
        if (
            obj != binding.object_id
            or record.get("category") != binding.category
            or body.get("source_id") != binding.source.source_id
        ):
            return False
        if any(
            canonical(record.get("fields", {}).get(k)) != canonical(v)
            or k not in record.get("fields", {})
            for k, v in binding.fields.items()
        ):
            return False
        if record.get("version") != binding.version or record.get("product") != binding.product:
            return False
        names = ["valid"]
        if binding.category in {"R", "P"} or binding.observed is not None:
            names.append("observed")
        for name in names:
            raw = record.get(name)
            expected = TimeWindow.model_validate_json(json.dumps(raw)) if raw else None
            if expected != getattr(binding, name):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
