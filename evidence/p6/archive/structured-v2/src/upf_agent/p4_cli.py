"""Terminal entry points for P4; no model service is required."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from upf_agent.contracts import TimeWindow
from upf_agent.log_correlation import correlate_events
from upf_agent.log_templates import train_templates
from upf_agent.log_tools import LogStore, ingest_log
from upf_agent.topology import import_topology, lookup_topology

app = typer.Typer(help="Deterministic local log and topology tools", no_args_is_help=True)
console = Console()
ROOT = Path(__file__).resolve().parents[2]


def window(start: str | None, end: str | None) -> TimeWindow | None:
    if bool(start) != bool(end):
        raise typer.BadParameter("both --start and --end are required")
    return (
        TimeWindow(
            start=datetime.fromisoformat(start),
            end=datetime.fromisoformat(end),
            timezone_source="cli_explicit_offset",
        )
        if start and end
        else None
    )


@app.command("import-log")
def import_log_command(
    source: Path,
    timezone: Annotated[str | None, typer.Option()] = None,
    device_id: Annotated[str | None, typer.Option()] = None,
    synthetic: Annotated[bool, typer.Option()] = False,
    snapshot: Annotated[str | None, typer.Option(help="Registered template snapshot ID")] = None,
) -> None:
    snapshot_path = None
    if snapshot:
        if len(snapshot) != 20 or not snapshot.isalnum():
            raise typer.BadParameter("invalid registered snapshot ID")
        snapshot_path = ROOT / "data/p4/templates" / snapshot
    console.print_json(
        data=ingest_log(
            source, ROOT / "data/p4/logs", timezone, device_id, synthetic, snapshot_path
        )
    )


@app.command("train-templates")
def train_command(source: Path, max_clusters: Annotated[int, typer.Option()] = 1024) -> None:
    console.print_json(data=train_templates(source, ROOT / "data/p4/templates", max_clusters))


@app.command("analyze-log")
def analyze_command(
    generation: str,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    event_type: Annotated[str | None, typer.Option()] = None,
    device_id: Annotated[str | None, typer.Option()] = None,
    bucket_minutes: Annotated[int, typer.Option()] = 60,
) -> None:
    result = LogStore(ROOT / "data/p4/logs", generation).analyze(
        window(start, end), event_type, device_id, bucket_minutes
    )
    console.print_json(data=result.model_dump(mode="json"))


@app.command("find-events")
def find_command(
    generation: str,
    event_type: Annotated[str | None, typer.Option()] = None,
    state: Annotated[str | None, typer.Option()] = None,
    protocol: Annotated[str | None, typer.Option()] = None,
    device_id: Annotated[str | None, typer.Option()] = None,
    peer_ip: Annotated[str | None, typer.Option()] = None,
    seid: Annotated[str | None, typer.Option()] = None,
    teid: Annotated[str | None, typer.Option()] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option()] = 20,
) -> None:
    result = LogStore(ROOT / "data/p4/logs", generation).find(
        window(start, end), limit, event_type, state, protocol, device_id, peer_ip, seid, teid
    )
    console.print_json(data=result.model_dump(mode="json"))


@app.command("read-context")
def context_command(
    generation: str,
    event_id: str,
    before: Annotated[int, typer.Option()] = 2,
    after: Annotated[int, typer.Option()] = 2,
) -> None:
    result = LogStore(ROOT / "data/p4/logs", generation).context(event_id, before, after)
    console.print_json(data=result.model_dump(mode="json"))


@app.command("template-inventory")
def template_inventory_command(generation: str) -> None:
    result = LogStore(ROOT / "data/p4/logs", generation).template_inventory()
    console.print_json(data=result.model_dump(mode="json"))


@app.command("import-topology")
def import_topology_command(
    source: Path,
    synthetic: Annotated[bool, typer.Option()] = False,
    mapping: Annotated[Path | None, typer.Option(help="JSON aliases/header mapping")] = None,
) -> None:
    config = json.loads(mapping.read_text()) if mapping else {}
    allowed = {"device_aliases", "header_mapping", "interface_aliases"}
    if set(config) - allowed:
        raise typer.BadParameter("unsupported topology mapping key")
    console.print_json(data=import_topology(source, ROOT / "data/p4/topology", synthetic, **config))


@app.command("lookup-topology")
def lookup_command(
    generation: str,
    device_id: str,
    interface: Annotated[str | None, typer.Option()] = None,
    at: Annotated[str | None, typer.Option()] = None,
) -> None:
    if len(generation) != 20 or not generation.isalnum():
        raise typer.BadParameter("invalid registered topology ID")
    result = lookup_topology(
        ROOT / "data/p4/topology" / generation,
        device_id,
        interface,
        datetime.fromisoformat(at) if at else None,
    )
    console.print_json(data=result.model_dump(mode="json"))


@app.command("correlate-events")
def correlate_command(
    generation: str,
    topology: Annotated[str | None, typer.Option(help="Registered topology ID")] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    horizon_seconds: Annotated[int, typer.Option()] = 86400,
    limit: Annotated[int, typer.Option()] = 1000,
) -> None:
    if topology and (len(topology) != 20 or not topology.isalnum()):
        raise typer.BadParameter("invalid registered topology ID")
    directory = ROOT / "data/p4/topology" / topology if topology else None
    result = correlate_events(
        LogStore(ROOT / "data/p4/logs", generation),
        window(start, end),
        directory,
        horizon_seconds,
        limit,
    )
    console.print_json(data=result.model_dump(mode="json"))
