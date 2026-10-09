"""Independent v8 commands; all artifacts go to the isolated runtime root."""

import json
from pathlib import Path
from typing import Annotated

import typer

from upf_agent.v8.engine import OrchestrationEngine
from upf_agent.v8.graph import save_snapshot
from upf_agent.v8.models import OrchestrationTask
from upf_agent.v8.source_graph import source_graph
from upf_agent.v8.store import Ledger

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "runtime/v8"
app = typer.Typer(help="v8 只读契约编排原型", no_args_is_help=True)
graph_app = typer.Typer(help="三类证据图")
app.add_typer(graph_app, name="graph")
experiment_app = typer.Typer(help="v8-next 冻结、CPU 工程验收与模型方案试点")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("freeze")
def freeze_command(
    output: Annotated[Path, typer.Option(help="新的冻结目录，不得已存在")],
    project: Annotated[Path, typer.Option()] = PROJECT,
):
    from upf_agent.v8.experiment_manifest import freeze_worktree

    manifest = freeze_worktree(project, output)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "files": len(manifest["files"]),
                "worktree_content_sha256": manifest["worktree_content_sha256"],
            }
        )
    )


@experiment_app.command("phase1")
def phase1_command(
    output: Annotated[Path, typer.Option(help="新的实验目录，不得已存在")],
    project: Annotated[Path, typer.Option()] = PROJECT,
    seeds: Annotated[int, typer.Option(min=1)] = 3,
    batches: Annotated[int, typer.Option(min=1)] = 10,
):
    from upf_agent.v8.phase1 import run_phase1

    result = run_phase1(project, output, seeds, batches)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise typer.Exit(2)


@experiment_app.command("phase2")
def phase2_command(
    output: Annotated[Path, typer.Option(help="新的四方法合成开发试点目录")],
    project: Annotated[Path, typer.Option()] = PROJECT,
    cases: Annotated[int, typer.Option(min=1, max=24)] = 24,
    case_id: Annotated[
        list[int] | None, typer.Option(help="重复指定原开发案例编号，1–24；与自定义 --cases 互斥")
    ] = None,
):
    from upf_agent.v8.phase2 import run_phase2

    result = run_phase2(project, output, cases, case_ids=case_id)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["execution_pass"]:
        raise typer.Exit(2)


@app.command("run")
def run_command(
    task: Annotated[Path, typer.Option(help="OrchestrationTask JSON 文件")],
    root: Annotated[Path, typer.Option()] = ROOT,
):
    engine = OrchestrationEngine(root)
    try:
        result = engine.run(OrchestrationTask.model_validate_json(task.read_text()))
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        engine.close()


@app.command("resume")
def resume_command(
    run_id: Annotated[str, typer.Option()], root: Annotated[Path, typer.Option()] = ROOT
):
    engine = OrchestrationEngine(root)
    try:
        typer.echo(json.dumps(engine.resume(run_id), ensure_ascii=False, indent=2))
    finally:
        engine.close()


@app.command("inspect")
def inspect_command(
    run_id: Annotated[str, typer.Option()], root: Annotated[Path, typer.Option()] = ROOT
):
    ledger = Ledger(root)
    try:
        typer.echo(json.dumps(ledger.inspect(run_id), ensure_ascii=False, indent=2))
    finally:
        ledger.close()


@graph_app.command("build")
def build_command(root: Annotated[Path, typer.Option()] = ROOT):
    store = source_graph(root)
    path = root / "data/v8/manual_graph.json"
    sha = save_snapshot(store, path)
    typer.echo(
        json.dumps(
            {
                "nodes": len(store.nodes),
                "edges": len(store.edges),
                "sha256": sha,
                "path": str(path),
            },
            ensure_ascii=False,
        )
    )


@app.command("evaluate")
def evaluate_command(
    suite: Annotated[str, typer.Option(help="dev/local/graph/agent/heldout/all")],
    root: Annotated[Path, typer.Option()] = ROOT,
    batch: Annotated[str | None, typer.Option()] = None,
):
    from upf_agent.v8.evaluation import evaluate_suite

    result = evaluate_suite(root, suite, batch)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("pass") is False:
        raise typer.Exit(2)
