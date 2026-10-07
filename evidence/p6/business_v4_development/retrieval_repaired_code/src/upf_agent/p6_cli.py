"""CLI evaluation entry points; draft development and final acceptance are distinct."""

import json
from pathlib import Path
from typing import Annotated

import typer

from upf_agent.config import load_model_config
from upf_agent.p6_agents import compare_agents
from upf_agent.p6_evaluation import (
    benchmark,
    compare_model,
    parameter_requests,
    prepare_retrieval,
    warm_rag_benchmark,
)
from upf_agent.p6_final import evaluate_reviewed, freeze_configuration
from upf_agent.p6_resources import ResourceSampler
from upf_agent.p6_review import adjudicate_answers, import_review
from upf_agent.quality_repair import write_json

app = typer.Typer(help="P6 development evaluation and acceptance safeguards")
ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "eval/p6/business_v4/candidates.jsonl"


@app.command("import-review")
def review(
    review: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    suite: Annotated[Path, typer.Option()] = SUITE,
    overrides: Annotated[Path | None, typer.Option()] = None,
) -> None:
    typer.echo(json.dumps(import_review(suite, review, output, overrides, ROOT)))


@app.command("cache-retrieval")
def cache_retrieval(
    output: Annotated[Path, typer.Option()],
    suite: Annotated[Path, typer.Option()] = SUITE,
    device: Annotated[str, typer.Option()] = "cuda:1",
) -> None:
    with ResourceSampler() as sampler:
        result = prepare_retrieval(ROOT, suite, output, device)
    result["resources"] = sampler.report()
    write_json(output / "manifest.json", result)
    typer.echo(json.dumps({"cases": len(result["rows"]), "cache": str(output)}))


@app.command("compare-model")
def compare(
    cache: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    suite: Annotated[Path, typer.Option()] = SUITE,
) -> None:
    if model not in {"baseline", "target"}:
        raise typer.BadParameter("model must be baseline or target")
    with ResourceSampler() as sampler:
        result = compare_model(ROOT, suite, cache, model, output)
    result["resources"] = sampler.report()
    write_json(output / "summary.json", result)
    typer.echo(json.dumps({"cases": result["cases"], "machine_pass": result["machine_pass"]}))


@app.command("tool-parameters")
def parameters(
    model: Annotated[str, typer.Option()], output: Annotated[Path, typer.Option()]
) -> None:
    if model not in {"baseline", "target"} or output.exists():
        raise typer.BadParameter("select a known model and a new output path")
    result = parameter_requests(ROOT, model, output)
    typer.echo(
        json.dumps({key: result[key] for key in ("cases", "first_pass", "after_one_repair")})
    )


@app.command("compare-agents")
def agents(
    output: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option()] = "baseline",
    suite: Annotated[Path, typer.Option()] = SUITE,
) -> None:
    if model not in {"baseline", "target"}:
        raise typer.BadParameter("model must be baseline or target")
    result = compare_agents(ROOT, suite, model, output)
    typer.echo(json.dumps({"cases": result["cases"], "rows": len(result["rows"])}))


@app.command("evaluate")
def evaluate(
    suite: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    model: Annotated[str, typer.Option()] = "baseline",
    split: Annotated[str, typer.Option()] = "dev",
    frozen: Annotated[Path | None, typer.Option()] = None,
) -> None:
    if model not in {"baseline", "target"}:
        raise typer.BadParameter("model must be baseline or target")
    result = evaluate_reviewed(ROOT, suite, model, split, output, frozen)
    typer.echo(json.dumps({"cases": result["cases"], "M6_pass": result["M6_pass"]}))


@app.command("freeze-config")
def freeze(
    suite: Annotated[Path, typer.Option()],
    development: Annotated[Path, typer.Option()],
    decision: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    freeze_configuration(ROOT, suite, development, decision, output)
    typer.echo(str(output))


@app.command("benchmark")
def bench(
    model: Annotated[str, typer.Option()],
    context: Annotated[int, typer.Option()],
    concurrency: Annotated[int, typer.Option(min=1, max=2)],
    server_pid: Annotated[int, typer.Option()],
    output: Annotated[Path, typer.Option()],
    thinking_policy: Annotated[str, typer.Option()] = "auto",
) -> None:
    if model not in {"baseline", "target"} or thinking_policy not in {"auto", "off"}:
        raise typer.BadParameter("select a known model and auto/off thinking policy")
    args = (Path("/proc") / str(server_pid) / "cmdline").read_bytes().decode().split("\0")
    config = load_model_config(ROOT / "config/models.yaml").generation[model]
    if str(config.path) not in args or args[args.index("--max-model-len") + 1] != str(context):
        raise typer.BadParameter("provided PID/model/context do not match the live server")
    server_sequences = int(args[args.index("--max-num-seqs") + 1])
    with ResourceSampler(server_pid) as sampler:
        result = benchmark(ROOT, model, context, concurrency, output, thinking_policy)
    result["resources"] = sampler.report()
    result["server_max_num_seqs"] = server_sequences
    write_json(output, result)
    typer.echo(str(output))


@app.command("benchmark-rag")
def rag_bench(
    model: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option()],
    suite: Annotated[Path, typer.Option()] = SUITE,
) -> None:
    if model not in {"baseline", "target"}:
        raise typer.BadParameter("model must be baseline or target")
    with ResourceSampler() as sampler:
        result = warm_rag_benchmark(ROOT, suite, model, output)
    result["resources"] = sampler.report()
    write_json(output, result)
    typer.echo(str(output))


@app.command("adjudicate")
def adjudicate(
    suite: Annotated[Path, typer.Option()],
    summary: Annotated[Path, typer.Option()],
    review: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    typer.echo(json.dumps(adjudicate_answers(suite, summary, review, output)))
