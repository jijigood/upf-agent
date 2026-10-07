from datetime import datetime
from pathlib import Path

import pytest
from openpyxl import Workbook

from upf_agent.log_parser import file_sha
from upf_agent.log_templates import FrozenTemplates, train_templates
from upf_agent.topology import import_topology, lookup_topology


def test_template_snapshot_is_frozen_and_novel_is_not_anomaly(tmp_path: Path) -> None:
    source = tmp_path / "a.log"
    source.write_text("2025-05-14 00:00:00 INFO normal starting service\n")
    report = train_templates(source, tmp_path / "snapshots")
    snapshot = tmp_path / "snapshots" / report["snapshot_id"]
    before = file_sha(snapshot / "templates.json")
    matcher = FrozenTemplates(snapshot)
    original = "2025-05-14 00:00:00 INFO normal starting service"
    assert matcher.match(original) is not None
    assert matcher.match("2025-05-14 00:00:00 INFO brand new alien event") is None
    assert matcher.match(original) == FrozenTemplates(snapshot).match(original)
    assert file_sha(snapshot / "templates.json") == before
    assert report["unknown_is_anomaly"] is False


def test_exact_interface_alias_and_half_open_topology_validity(tmp_path: Path) -> None:
    source = tmp_path / "topology.csv"
    source.write_text(
        "device_id,device_port,peer_device_id,peer_port,interface_type,valid_from,valid_to\n"
        "UPF-A,p1,SW1,x1,N3,2025-05-14T00:00:00+08:00,2025-05-15T00:00:00+08:00\n"
        "UPF-A,p2,SW1,x2,N30,,\n"
        ",p3,SW2,x3,N3,,\n"
    )
    report = import_topology(
        source,
        tmp_path / "topo",
        synthetic=True,
        device_aliases={"alias": "UPF-A"},
        interface_aliases={"N-3": "N3"},
    )
    archive = tmp_path / "topo" / report["generation"]
    result = lookup_topology(
        archive, "alias", "N-3", datetime.fromisoformat("2025-05-14T01:00:00+08:00")
    )
    assert len(result.data["edges"]) == 1
    assert result.data["edges"][0]["synthetic"] is True
    assert (
        lookup_topology(
            archive, "alias", "N3", datetime.fromisoformat("2025-05-15T00:00:00+08:00")
        ).data["edges"]
        == []
    )
    assert report["quarantined_rows"] == 1


def test_topology_tool_explains_expired_synthetic_edge_without_claiming_live_neighbor(tmp_path):
    import json
    from uuid import uuid4

    from upf_agent.agent_contracts import AgentTask, ToolCall
    from upf_agent.agent_tools import execute_tool

    source = tmp_path / "fixture.csv"
    source.write_text(
        "device_id,device_port,peer_device_id,peer_port,interface_type,valid_from,valid_to\n"
        "SYN-UPF,p1,SYN-SW,x1,N3,2025-01-01T00:00:00+00:00,2025-12-31T00:00:00+00:00\n"
        "SYN-UPF,p2,SYN-SW,x2,N30,,\n"
    )
    manifest = import_topology(source, tmp_path / "data/p4/topology", synthetic=True)
    task = AgentTask(
        run_id=uuid4().hex,
        question="检查已过期N3关系",
        task_type="diagnose",
        topology_generation=manifest["generation"],
        device_ids=["SYN-UPF"],
    )
    request = {
        "topology_id": manifest["generation"],
        "device_id": "SYN-UPF",
        "interface_type": "N3",
        "at": "2026-01-01T00:00:00+00:00",
    }
    packet = execute_tool(tmp_path, task, ToolCall(name="lookup_topology", arguments=request), [])
    assert packet.result.status.value == "missing_source"
    assert packet.result.data["edges"] == []
    assert len(packet.result.data["excluded_edges"]) == 1
    assert len(packet.evidence) == len(packet.facts) == 1
    assert "已过期" in packet.facts[0].text
    assert "仅为合成" in packet.facts[0].text
    assert json.loads(packet.evidence[0].excerpt)["synthetic"] is True
    negative = execute_tool(
        tmp_path,
        task,
        ToolCall(name="lookup_topology", arguments={**request, "interface_type": "N30"}),
        [],
    )
    assert negative.error_code == "invalid_arguments"
    assert not negative.facts


def test_merged_cells_fill_only_real_range_not_following_group(tmp_path: Path) -> None:
    book = Workbook()
    sheet = book.active
    sheet.title = "links"
    sheet.append(["网元", "端口", "对端网元", "对端端口", "接口"])
    sheet.append(["UPF-A", "p1", "SW", "x1", "N3"])
    sheet.append([None, "p2", "SW", "x2", "N6"])
    sheet.append([None, "p3", "SW", "x3", "N3"])
    sheet.merge_cells("A2:A3")
    source = tmp_path / "t.xlsx"
    book.save(source)
    report = import_topology(source, tmp_path / "topo", synthetic=True)
    assert report["accepted_edges"] == 2
    assert report["quarantined_rows"] == 1
    result = lookup_topology(tmp_path / "topo" / report["generation"], "UPF-A", "N6")
    assert result.data["edges"][0]["source_row"] == 3
    assert any("A2" in flag for flag in result.data["edges"][0]["quality_flags"])


def test_alias_cycles_naive_validity_and_duplicate_headers_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "t.csv"
    source.write_text(
        "device_id,device_port,peer_device_id,peer_port,valid_from\nA,p1,B,p2,2025-05-14T00:00:00\n"
    )
    with pytest.raises(ValueError, match="cyclic"):
        import_topology(source, tmp_path / "topo", device_aliases={"A": "B", "B": "A"})
    report = import_topology(source, tmp_path / "topo")
    assert report["quarantined_rows"] == 1
    assert report["accepted_edges"] == 0


def test_lru_eviction_is_reported_not_hidden(tmp_path: Path) -> None:
    source = tmp_path / "a.log"
    source.write_text(
        "2025-05-14 00:00:00 INFO apple hello world\n"
        "2025-05-14 00:00:01 INFO brick yellow stone red\n"
        "2025-05-14 00:00:02 INFO canyon blue sky cloud mountain\n"
    )
    report = train_templates(source, tmp_path / "snapshots", max_clusters=2)
    assert report["evicted_clusters"] == 1
    assert len(report["warnings"]) == 1
