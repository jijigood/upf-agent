"""Conservative entity/time associations: observations never imply causality."""

import time
from datetime import datetime
from pathlib import Path
from typing import cast

from upf_agent.contracts import ResultStatus, TimeWindow, ToolResult
from upf_agent.log_tools import LogStore, Row
from upf_agent.topology import lookup_topology


def correlate_events(
    store: LogStore,
    window: TimeWindow | None = None,
    topology_directory: Path | None = None,
    horizon_seconds: int = 86400,
    limit: int = 1000,
) -> ToolResult:
    if not 1 <= horizon_seconds <= 7 * 86400 or not 1 <= limit <= 10000:
        raise ValueError("correlation horizon/limit out of bounds")
    if gap := store._timezone_gap(window):
        return gap
    started = time.monotonic()
    where, parameters = store._where(window, {})
    rows = store._query(
        f"SELECT * FROM events WHERE ({where}) AND event_state IN ('raised','cleared') "
        "ORDER BY timestamp_utc NULLS LAST, byte_offset LIMIT ?",
        [*parameters, limit + 1],
    )
    truncated = len(rows) > limit
    rows = rows[:limit]
    pending: dict[tuple[str, str, str], list[Row]] = {}
    pairs: list[Row] = []
    unpaired: list[Row] = []
    identity_gaps = timezone_gaps = 0
    topology_checks: list[Row] = []
    seen_checks: set[tuple[str, str, str]] = set()
    warnings = [
        "Associations are entity/time observations, not root causes or independent fault counts."
    ]
    for row in rows:
        kind, device = str(row["event_type"]), row["device_id"]
        if not device or (kind != "license_not_installed" and not row["peer_ip"]):
            identity_gaps += 1
            unpaired.append(
                {"event_id": row["event_id"], "reason": "missing_real_device_or_peer_identity"}
            )
            continue
        if row["timestamp_utc"] is None:
            timezone_gaps += 1
            unpaired.append({"event_id": row["event_id"], "reason": "timezone_unresolved"})
            continue
        at = datetime.fromisoformat(str(row["timestamp_utc"]))
        key = kind, str(device), str(row["peer_ip"] or "local_license")
        if row["event_state"] == "raised":
            pending.setdefault(key, []).append(row)
        else:
            raises = pending.pop(key, [])
            if len(raises) == 1:
                raised = raises[0]
                duration = (
                    at - datetime.fromisoformat(str(raised["timestamp_utc"]))
                ).total_seconds()
                if 0 <= duration <= horizon_seconds:
                    pairs.append(
                        {
                            "raised_event_id": raised["event_id"],
                            "cleared_event_id": row["event_id"],
                            "event_type": kind,
                            "device_id": device,
                            "peer_ip": row["peer_ip"],
                            "duration_seconds": duration,
                            "basis": "same_explicit_entity_and_type_within_horizon",
                        }
                    )
                else:
                    unpaired.extend(
                        {"event_id": item["event_id"], "reason": "outside_pairing_horizon"}
                        for item in [raised, row]
                    )
            else:
                unpaired.extend(
                    {
                        "event_id": item["event_id"],
                        "reason": "multiple_raises_ambiguous"
                        if raises
                        else "no_raise_in_selected_scope",
                    }
                    for item in [*raises, row]
                )
        interface = "N4" if row["protocol"] == "PFCP" else None
        # GTP-U may traverse N3/N9; preserve both as candidates, not a claimed mapping.
        interfaces = (
            [interface] if interface else ["N3", "N9"] if row["protocol"] == "GTP-U" else []
        )
        for candidate in interfaces:
            check_key = str(device), str(candidate), at.isoformat()
            if topology_directory and check_key not in seen_checks and len(seen_checks) < 100:
                seen_checks.add(check_key)
                result = lookup_topology(topology_directory, str(device), candidate, at)
                topology_checks.append(
                    {
                        "event_id": row["event_id"],
                        "candidate_interface": candidate,
                        "status": result.status.value,
                        "edges": result.data.get("edges", []),
                        "warnings": result.warnings,
                        "is_causal_path": False,
                    }
                )
    for remaining in pending.values():
        unpaired.extend(
            {"event_id": row["event_id"], "reason": "no_clear_in_selected_scope"}
            for row in remaining
        )
    if not topology_directory:
        warnings.append(
            "No topology source: cannot identify real switch ports or a causal propagation path."
        )
    if identity_gaps:
        warnings.append("Masked/missing entity identifiers prevent alarm pairing.")
    if timezone_gaps:
        warnings.append(
            "Unknown timezone prevents absolute durations and cross-source association."
        )
    if truncated:
        warnings.append("Correlation input capped; pair completeness is unverified.")
    return ToolResult(
        status=ResultStatus.PARTIAL if warnings else ResultStatus.COMPLETED,
        data={
            "alarm_pairs": pairs,
            "unpaired": unpaired,
            "identity_gap_messages": identity_gaps,
            "timezone_gap_messages": timezone_gaps,
            "topology_checks": topology_checks,
            "input_messages": len(rows),
            "truncated": truncated,
            "horizon_seconds": horizon_seconds,
            "count_unit": "messages_not_independent_faults",
            "scope_generation": store.manifest["generation"],
        },
        evidence_ids=cast(list[str], [row["event_id"] for row in rows]),
        warnings=warnings,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )
