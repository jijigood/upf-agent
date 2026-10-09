"""Explicitly synthetic examples; scenario oracles live separately in evaluation data."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from upf_agent.v8.models import BusinessPath, OrchestrationTask, PathPosition, PermissionScope

FIELDS = [
    "instruction",
    "interface",
    "state",
    "n4_enabled",
    "mtu",
    "event_type",
    "service_id",
    "local_endpoint",
    "peer_endpoint",
    "direction",
    "mapping_version",
]


def example(
    root: Path, name: str = "complete", template: str = "upf_n4_fault"
) -> OrchestrationTask:
    now = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    observed = {
        "start": (now - timedelta(seconds=60)).isoformat(),
        "end": now.isoformat(),
        "timezone_source": "synthetic oracle UTC",
    }
    objects = ["UPF-A", "SMF-B"]
    positions = [
        PathPosition(
            object_id=o,
            domain=f"domain-{i}",
            interface="N3" if template == "upf_n3_fault" else "N4",
        )
        for i, o in enumerate(objects)
    ]
    records = []
    for i, obj in enumerate(objects):
        for kind, fields in (
            ("K", {"instruction": "检查声明接口的只读配置与状态"}),
            (
                "R",
                {
                    "interface": "N4" if template == "upf_n4_fault" else "N3",
                    "state": "up",
                    "n4_enabled": True,
                    "mtu": 1500,
                    "event_type": "observed",
                },
            ),
            (
                "P",
                {
                    "service_id": "service-001",
                    "local_endpoint": f"192.0.2.{i + 1}",
                    "peer_endpoint": f"192.0.2.{2 - i}",
                    "direction": "tx" if i == 0 else "rx",
                    "mapping_version": "map-1",
                },
            ),
        ):
            records.append(
                {
                    "object_id": obj,
                    "category": kind,
                    "fields": fields,
                    "product": "UPF",
                    "purposes": [template],
                    "version": "5.0",
                    "config_version": "cfg-1",
                    "observed": observed,
                }
            )
    body = {
        "synthetic": True,
        "corroborated": True,
        "source_id": f"synthetic-{name}",
        "path": {
            "service_id": "service-001",
            "objects": objects,
            "version": "map-1",
            "domains": [p.domain for p in positions],
        },
        "observations": records,
    }
    path = root / "data/fixtures/v8" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return OrchestrationTask(
        question="检查 UPF-A 与 SMF-B 的接口证据并生成只读建议",
        template=template,
        objects=objects,
        product="UPF",
        version="5.0",
        as_of=now,
        synthetic=True,
        path=BusinessPath(
            service_id="service-001",
            positions=positions,
            source_id=body["source_id"],
            source_sha256=sha,
            source_path=str(path.relative_to(root)),
            version="map-1",
            resolution="resolved",
            synthetic=True,
        ),
        permission=PermissionScope(
            objects=objects, domains=["domain-0", "domain-1"], readable_fields=FIELDS
        ),
        observation_files=[str(path.relative_to(root))],
    )
