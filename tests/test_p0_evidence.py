"""Regression checks for source audit and the historical baseline."""

import json
from pathlib import Path

from upf_agent.p0 import docx_inventory, run_p0

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/huangxiaolin/dify")


def test_p0_audit_preserves_source_identity_and_legacy_counts() -> None:
    result = run_p0(ROOT)
    records = json.loads((ROOT / "data/p0/sources.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "data/p0/legacy_baseline.json").read_text(encoding="utf-8"))

    assert result["sources"] == 7
    assert result["node_counts"]["llm"] == 8
    assert result["edge_count"] == 32
    assert baseline["total_classified_records"] == 32539
    assert baseline["category_counts"]["Unclassified"] == 25979
    assert baseline["category_counts"]["PFCP 节点不可达"] == 37
    assert {entry["relative_path"] for entry in records} == {
        path.name
        for path in SOURCE.iterdir()
        if path.suffix == ".docx"
        or path.name in {"upftest.yml", "upfd_masked.log", "提取自main.pdf"}
    }


def test_textboxes_are_counted_in_emergency_guide() -> None:
    path = SOURCE / "中国联通5G专网UPF紧急故障指引0429(1).docx"
    structure = docx_inventory(path)
    assert structure["textboxes"] == 52
    assert structure["tables"] == 0
