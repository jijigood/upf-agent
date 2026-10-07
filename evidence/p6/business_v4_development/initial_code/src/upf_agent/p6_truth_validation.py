"""Resolve reviewed source references and preserve their actual partition boundaries."""

import csv
import re
from pathlib import Path

from upf_agent.log_tools import LogStore
from upf_agent.p3_answer import source_lookup
from upf_agent.p3_index import load_active_index, load_chunks
from upf_agent.p6_contracts import EvalCase


def validate_reviewed_sources(root: Path, cases: list[EvalCase]) -> dict:
    directory, manifest = load_active_index(root)
    chunks = load_chunks(directory)
    source_assignment, native, logs, stores, topologies = {}, set(), set(), {}, set()
    for case in cases:
        if case.category in {"command", "knowledge", "troubleshooting"}:
            if case.index_generation != manifest["index_generation"] or not case.evidence_ids:
                raise ValueError(
                    f"{case.id}: reviewed manual truth requires current native sources"
                )
            for eid in case.evidence_ids:
                if eid not in chunks or chunks[eid].parent_id not in case.source_groups:
                    raise ValueError(
                        f"{case.id}: corrected source exceeds the declared source groups"
                    )
                if eid not in native:
                    source_lookup(root, case.index_generation, eid)
                    native.add(eid)
        elif case.category == "log":
            if (
                not case.log_generation
                or not re.fullmatch(r"[0-9a-f]{20}", case.log_generation)
                or not case.evidence_ids
            ):
                raise ValueError(f"{case.id}: reviewed log truth requires registered event IDs")
            if case.log_generation not in stores:
                stores[case.log_generation] = LogStore(root / "data/p4/logs", case.log_generation)
            for eid in case.evidence_ids:
                if eid not in logs:
                    if not stores[case.log_generation].context(eid).data.get("event"):
                        raise ValueError(f"{case.id}: reviewed event cannot be resolved")
                    logs.add(eid)
                if eid in source_assignment and source_assignment[eid] != case.split:
                    raise ValueError("Actual log evidence crosses dev and heldout")
                source_assignment[eid] = case.split
        elif case.category == "topology":
            if not case.evidence_ids:
                raise ValueError(f"{case.id}: reviewed topology truth requires a fixture row")
            for eid in case.evidence_ids:
                path_text, row_text = eid.rsplit(":row", 1)
                path = (root / path_text).resolve()
                if not path.is_relative_to((root / "eval/p6").resolve()) or not path.is_file():
                    raise ValueError(f"{case.id}: topology truth must use an existing P6 fixture")
                with path.open() as handle:
                    count = sum(1 for _ in csv.DictReader(handle))
                # Match topology.import_topology: row 1 is the CSV header.
                if not 2 <= int(row_text) <= count + 1:
                    raise ValueError(f"{case.id}: topology row is outside the fixture")
                key = str(path)
                if key in source_assignment and source_assignment[key] != case.split:
                    raise ValueError("Actual topology file crosses dev and heldout")
                source_assignment[key] = case.split
                topologies.add(key)
    return {
        "native_chunks_resolved": len(native),
        "log_events_resolved": len(logs),
        "topology_files_resolved": len(topologies),
        "actual_source_partition_pass": True,
        "scope": "source identity and original SHA; human relevance still required",
    }
