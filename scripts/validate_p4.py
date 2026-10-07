"""Replay the real log and synthetic topology; emit auditable development evidence."""

import json
from pathlib import Path

from openpyxl import Workbook

from upf_agent.log_correlation import correlate_events
from upf_agent.log_parser import file_sha
from upf_agent.log_templates import FrozenTemplates, train_templates
from upf_agent.log_tools import LogStore, ingest_log, write_json
from upf_agent.topology import import_topology, lookup_topology

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    evidence = ROOT / "evidence/p4"
    evidence.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/raw/upfd_masked.log"
    template = train_templates(source, ROOT / "data/p4/templates")
    snapshot = ROOT / "data/p4/templates" / str(template["snapshot_id"])
    manifest = ingest_log(source, ROOT / "data/p4/logs", template_snapshot=snapshot)
    store = LogStore(ROOT / "data/p4/logs", str(manifest["generation"]))
    analysis = store.analyze()
    continuity = store._query(
        "SELECT count(*) violations FROM (SELECT byte_offset, line_start, "
        "lag(byte_offset+byte_length,1,0) OVER (ORDER BY byte_offset) previous_end, "
        "lag(line_end+1,1,1) OVER (ORDER BY byte_offset) previous_line FROM events) "
        "WHERE byte_offset!=previous_end OR line_start!=previous_line"
    )[0]
    assert continuity["violations"] == 0
    assert manifest["physical_lines"] == 32539
    assert manifest["source_bytes"] == source.stat().st_size == 6347974
    assert (
        manifest["source_sha256"]
        == "a698d029342259b717db6f0520f4e9b006e59ae6e8ae743d0cc482199b242fcc"
    )
    expected = {
        "pfcp_heartbeat_request:observed": 23,
        "pfcp_node_unreachable:raised": 14,
        "pfcp_node_unreachable:cleared": 12,
        "license_not_installed:raised": 1,
        "license_not_installed:cleared": 11,
        "far_handover:observed": 3238,
        "gtpu_unreachable_detected:raised": 120,
        "gtpu_unreachable_report:raised": 120,
        "pfcp_session_deleted:observed": 5715,
        "gtpu_echo_retry:observed": 382,
    }
    assert all(manifest["state_counts"].get(key) == value for key, value in expected.items())
    examples = {}
    for kind in (
        "far_handover",
        "pfcp_heartbeat_request",
        "pfcp_node_unreachable",
        "license_not_installed",
    ):
        found = store.find(event_type=kind, limit=1)
        event = found.data["events"][0]
        context = store.context(event["event_id"], 0, 0)
        with source.open("rb") as handle:
            handle.seek(event["byte_offset"])
            raw = handle.read(event["byte_length"]).decode()
        assert context.data["raw_text"] == raw
        examples[kind] = context.model_dump(mode="json")
    frozen = FrozenTemplates(snapshot)
    before = file_sha(snapshot / "templates.json")
    one = examples["far_handover"]["data"]["raw_text"]
    repeated = [frozen.match(one) for _ in range(3)]
    assert repeated[0] is not None and len(set(repeated)) == 1
    assert FrozenTemplates(snapshot).match(one) == repeated[0]
    assert file_sha(snapshot / "templates.json") == before
    fixtures = ROOT / "data/p4/fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    csv_source = fixtures / "synthetic_topology.csv"
    csv_source.write_text(
        "device_id,device_port,peer_device_id,peer_port,interface_type,valid_from,valid_to\n"
        "SYNTHETIC-UPF,p1,SYNTHETIC-SW,x1,N3,2025-01-01T00:00:00+00:00,2026-01-01T00:00:00+00:00\n"
        "SYNTHETIC-UPF,p2,SYNTHETIC-SW,x2,N30,,\n"
        ",p3,SYNTHETIC-SW,x3,N3,,\n",
        encoding="utf-8",
    )
    csv_report = import_topology(
        csv_source, ROOT / "data/p4/topology", True, {"DEMO": "SYNTHETIC-UPF"}
    )
    xlsx_source = fixtures / "synthetic_merged_topology.xlsx"
    if not xlsx_source.exists():
        book = Workbook()
        sheet = book.active
        sheet.title = "synthetic_links"
        sheet.append(["网元", "端口", "对端网元", "对端端口", "接口"])
        sheet.append(["SYNTHETIC-UPF", "p1", "SYNTHETIC-SW", "x1", "N3"])
        sheet.append([None, "p2", "SYNTHETIC-SW", "x2", "N6"])
        sheet.append([None, "p3", "SYNTHETIC-SW", "x3", "N3"])
        sheet.merge_cells("A2:A3")
        book.save(xlsx_source)
    xlsx_report = import_topology(xlsx_source, ROOT / "data/p4/topology", True)
    lookup = lookup_topology(ROOT / "data/p4/topology" / csv_report["generation"], "DEMO", "N3")
    assert len(lookup.data["edges"]) == 1
    assert csv_report["quarantined_rows"] == xlsx_report["quarantined_rows"] == 1
    correlation = correlate_events(store)
    assert correlation.data["alarm_pairs"] == []
    report = {
        "kind": "development_deterministic_regression_not_business_gold",
        "log": manifest,
        "analysis": analysis.model_dump(mode="json"),
        "continuity": continuity,
        "independently_counted_semantic_checks": expected,
        "old_baseline": json.loads((ROOT / "evidence/p0/legacy_baseline.json").read_text()),
        "examples": examples,
        "templates": template,
        "template_replay": {"matches": repeated, "snapshot_unchanged": True},
        "template_inventory": store.template_inventory().model_dump(mode="json"),
        "synthetic_csv_topology": csv_report,
        "synthetic_xlsx_topology": xlsx_report,
        "synthetic_lookup": lookup.model_dump(mode="json"),
        "real_log_correlation": correlation.model_dump(mode="json"),
        "human_timezone_confirmed": False,
        "real_topology_validated": False,
        "business_accuracy_validated": False,
    }
    write_json(evidence / "regression.json", report)
    print(
        json.dumps(
            {
                "generation": manifest["generation"],
                "events": manifest["events"],
                "states": manifest["state_counts"],
                "templates": template["clusters"],
                "regression": str(evidence / "regression.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
