"""Offline migration CLI. Later milestones add diagnosis and agent sessions."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from upf_agent.agent_cli import app as agent_app
from upf_agent.auxiliary import attach_assets, search_auxiliary
from upf_agent.config import load_model_config
from upf_agent.corpus_db import ingest, query_blocks
from upf_agent.doctor import run_doctor
from upf_agent.image_queue import build_queue
from upf_agent.models import LocalModelError, run_smoke
from upf_agent.p0 import run_p0
from upf_agent.p2_verify import verify
from upf_agent.p3_ablation import run_ablation
from upf_agent.p3_answer import ask, render_answer, source_lookup
from upf_agent.p3_index import build_index
from upf_agent.p3_quality import run_quality
from upf_agent.p3_search import search
from upf_agent.p3_verify import verify_index
from upf_agent.p4_cli import app as p4_app
from upf_agent.p6_cli import app as p6_app
from upf_agent.quality_repair import active_ocr_path, rebase_ocr, repair_ocr
from upf_agent.retrieval import RetrievalWorker, RetrievalWorkerError
from upf_agent.review import make_review_packet, write_review_sheet
from upf_agent.serve import ServiceLaunchBlocked, launch_local_model

app = typer.Typer(help="Offline UPF Agent migration CLI", no_args_is_help=True)
app.add_typer(p4_app, name="p4")
app.add_typer(p6_app, name="p6")
for command in agent_app.registered_commands:
    app.command(command.name)(command.callback)
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@app.command("evaluate-p3")
def evaluate_p3_command(
    dataset: Annotated[Path, typer.Option()] = PROJECT_ROOT / "eval/p3_answer_regression.jsonl",
    output: Annotated[Path, typer.Option()] = PROJECT_ROOT / "evidence/p3/quality_regression.json",
    device: Annotated[str, typer.Option()] = "cuda:1",
) -> None:
    """Run inspectable development answer checks, without claiming human accuracy."""
    summary = run_quality(PROJECT_ROOT, dataset, output, device)
    console.print_json(data=summary)
    if summary["machine_contract_pass"] != summary["cases"]:
        raise typer.Exit(2)


@app.command("ask")
def ask_command(
    question: Annotated[str | None, typer.Argument()] = None,
    file: Annotated[Path | None, typer.Option(help="UTF-8 question file; omit question")] = None,
    product: Annotated[str | None, typer.Option()] = None,
    version: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "cuda:1",
    model: Annotated[str, typer.Option()] = "baseline",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Answer from local manual evidence; persist answer and retrieval trace."""
    if question is not None and file is not None:
        raise typer.BadParameter("use either question or --file")
    if file is not None:
        question = file.read_text(encoding="utf-8")
    elif question is None and not sys.stdin.isatty():
        question = sys.stdin.read(8001)
    if not question or not question.strip() or len(question) > 2000:
        raise typer.BadParameter("provide 1–2000 characters via argument, --file, or stdin")
    if model not in {"baseline", "target"}:
        raise typer.BadParameter("model must be baseline or target")
    try:
        report = ask(PROJECT_ROOT, question.strip(), product, version, device, model)
    except (ValueError, OSError, RetrievalWorkerError, LocalModelError) as error:
        console.print(str(error), markup=False)
        raise typer.Exit(1) from error
    if json_output:
        console.print_json(data=report)
    else:
        console.print(render_answer(report), markup=False)
        console.print(f"记录：{PROJECT_ROOT / 'runs/p3' / report['run_id']}", markup=False)
    if report["status"] == "failed":
        raise typer.Exit(1)
    if report["status"] in {"partial", "missing_source"}:
        raise typer.Exit(2)


@app.command("manual-chat")
def manual_chat_command(
    product: Annotated[str | None, typer.Option()] = None,
    version: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "cuda:1",
    model: Annotated[str, typer.Option()] = "baseline",
    jsonl: Annotated[
        bool, typer.Option(help="Read one question per line; emit one JSON result per line")
    ] = False,
) -> None:
    """Ask multiple manual questions while keeping local retrieval models loaded."""
    if model not in {"baseline", "target"}:
        raise typer.BadParameter("model must be baseline or target")
    models = load_model_config(PROJECT_ROOT / "config/models.yaml")
    with RetrievalWorker(PROJECT_ROOT, "embed", models.embedding.path, device) as embedding:
        with RetrievalWorker(PROJECT_ROOT, "rerank", models.reranker.path, device) as reranking:
            if jsonl:
                print(json.dumps({"event": "ready"}), flush=True)
            else:
                console.print("手册问答已就绪。输入问题，/exit 退出。", markup=False)
            while True:
                try:
                    question = input() if jsonl else input("UPF> ")
                except (EOFError, KeyboardInterrupt):
                    break
                question = question.strip()
                if question == "/exit":
                    break
                if not question or len(question) > 2000:
                    if jsonl:
                        print(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "error": "question must contain 1–2000 characters",
                                }
                            ),
                            flush=True,
                        )
                    else:
                        console.print("问题应为 1–2000 字符。", markup=False)
                    continue
                try:
                    report = ask(
                        PROJECT_ROOT,
                        question,
                        product,
                        version,
                        device,
                        model,
                        (embedding, reranking),
                    )
                except (ValueError, OSError, RetrievalWorkerError, LocalModelError) as error:
                    if jsonl:
                        print(
                            json.dumps(
                                {"status": "failed", "error": str(error)}, ensure_ascii=False
                            ),
                            flush=True,
                        )
                    else:
                        console.print(f"问答失败：{error}", markup=False)
                    continue
                if jsonl:
                    print(json.dumps(report, ensure_ascii=False), flush=True)
                else:
                    console.print(render_answer(report), markup=False)
                    console.print(
                        f"记录：{PROJECT_ROOT / 'runs/p3' / report['run_id']}", markup=False
                    )


@app.command("source")
def source_command(
    chunk_id: Annotated[str, typer.Argument()],
    generation: Annotated[str, typer.Option(help="Index generation recorded in answer")],
) -> None:
    """Verify the source hash and inspect the archived cited chunk."""
    console.print_json(data=source_lookup(PROJECT_ROOT, generation, chunk_id))


@app.command("ablate-retrieval")
def ablate_retrieval_command(
    dataset: Annotated[Path, typer.Option()] = PROJECT_ROOT / "eval/p3_dev_probes.jsonl",
    output: Annotated[Path, typer.Option()] = PROJECT_ROOT / "evidence/p3/ablation.json",
    device: Annotated[str, typer.Option()] = "cuda:1",
) -> None:
    """Compare BGE, Qwen, BM25, exact commands, and reranking on development cases."""
    console.print_json(data=run_ablation(PROJECT_ROOT, dataset, output, device))


@app.command("ingest-documents")
def ingest_documents() -> None:
    """Ingest the four SHA-verified DOCX manuals into an atomic local generation."""
    console.print_json(data=ingest(PROJECT_ROOT))


@app.command("manual-blocks")
def manual_blocks(
    product: Annotated[str | None, typer.Option()] = None,
    version: Annotated[str | None, typer.Option()] = None,
    kind: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=500)] = 20,
) -> None:
    """Read active corpus blocks with product and version filtering."""
    blocks = query_blocks(
        PROJECT_ROOT / "data/p2/corpus.sqlite3",
        product=product,
        version=version,
        kind=kind,
        limit=limit,
    )
    console.print_json(data=[b.model_dump() for b in blocks])


@app.command("p2-review-packet")
def p2_review_packet() -> None:
    """Write 60 source-located command/table/step cases for explicit review."""
    root = PROJECT_ROOT
    queue = build_queue(
        root / "data/p2/corpus.sqlite3", root / "evidence/p2/image_review_queue.json"
    )
    source_root = Path(
        yaml.safe_load((root / "config/corpus.yaml").read_text(encoding="utf-8"))["source_root"]
    )
    result = make_review_packet(
        root / "data/p2/corpus.sqlite3", root / "evidence/p2/structure_review_60.json", source_root
    )
    sheet = write_review_sheet(
        root / "evidence/p2/structure_review_60.json",
        root / "evidence/p2/image_review_queue.json",
        root / "evidence/p0/sources.json",
        root / "evidence/p2" / f"review_sheet_{result['generation']}_v3.md",
    )
    console.print_json(
        data={**result, "screenshot_candidates": len(queue), "review_sheet": str(sheet)}
    )


@app.command("ocr-search")
def ocr_search(
    query: Annotated[str, typer.Argument(help="Literal text in screenshot OCR or its heading")],
    limit: Annotated[int, typer.Option(min=1, max=30)] = 10,
) -> None:
    """Find screenshot OCR with its source locator and unverified status."""
    path = PROJECT_ROOT / "evidence/p2/ocr_corrected.json"
    if not path.exists():
        path = active_ocr_path(PROJECT_ROOT)
    if not path.is_file():
        console.print("[yellow]OCR artifact is not available yet.[/yellow]")
        raise typer.Exit(2)
    data = json.loads(path.read_text(encoding="utf-8"))
    active = json.loads(active_ocr_path(PROJECT_ROOT).read_text())
    if data["generation"] != active["generation"]:
        raise typer.BadParameter("OCR generation is stale; run repair-ocr after ingest")
    needle = query.casefold()
    hits = []
    for item in data["images"]:
        lines = [
            line
            for line in item["lines"]
            if line.get("usable", True) and needle in line["text"].casefold()
        ]
        if lines or needle in " ".join(item["section_path"]).casefold():
            hits.append(
                {
                    "source_id": item["source_id"],
                    "block_id": item["block_id"],
                    "locator": item["locator"],
                    "section_path": item["section_path"],
                    "lines": lines[:20],
                    "status": item["status"],
                    "human_review_status": item.get("human_review_status", "pending"),
                    "coverage": item.get("coverage", "visible_screenshot_only"),
                    "excluded_lines": sum(not line.get("usable", True) for line in item["lines"]),
                }
            )
    console.print_json(data=hits[:limit])


@app.command("attach-p2-assets")
def attach_p2_assets() -> None:
    """Idempotently attach unverified OCR and isolated paper items to the active generation."""
    root = PROJECT_ROOT
    rebase_ocr(root)
    sources = json.loads((root / "evidence/p0/sources.json").read_text(encoding="utf-8"))
    paper_sha = next(item["sha256"] for item in sources if item["role"] == "paper")
    result = attach_assets(
        root / "data/p2/corpus.sqlite3",
        active_ocr_path(root),
        root / "evidence/p2/paper_docling.json",
        paper_sha,
    )
    console.print_json(data=result)


@app.command("repair-ocr")
def repair_ocr_command() -> None:
    """Apply source-checked corrections while preserving originals and unverified status."""
    console.print_json(data=repair_ocr(PROJECT_ROOT))


@app.command("paper-search")
def paper_search(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
) -> None:
    """Literal search of paper material, isolated from product manual search."""
    result = search_auxiliary(PROJECT_ROOT / "data/p2/corpus.sqlite3", query, "pdf", limit)
    console.print_json(data=result)


@app.command("verify-p2")
def verify_p2() -> None:
    """Recheck source SHA, locators, command text, OCR/PDF roles, and counts."""
    result = verify(PROJECT_ROOT, PROJECT_ROOT / "evidence/p2/validation.json")
    console.print_json(data=result)


@app.command("build-index")
def build_index_command(
    device: Annotated[str, typer.Option(help="Embedding device: cuda:1 or cpu")] = "cuda:1",
) -> None:
    """Build and atomically publish P3 Qdrant/BM25/exact indexes."""
    result = build_index(PROJECT_ROOT, device=device)
    console.print_json(data=result)


@app.command("search-index")
def search_index_command(
    query: Annotated[str, typer.Argument(help="Manual retrieval query")],
    product: Annotated[str | None, typer.Option(help="UPF or 5GC网管")] = None,
    version: Annotated[
        str | None, typer.Option(help="5.0 or V1.3; unknown versions remain eligible")
    ] = None,
    device: Annotated[str, typer.Option(help="Retrieval model device")] = "cuda:1",
    output: Annotated[Path | None, typer.Option(help="Optional inspectable JSON path")] = None,
) -> None:
    """Inspect three retrieval routes, fusion, reranking, and bounded evidence context."""
    result = search(PROJECT_ROOT, query, product=product, version=version, device=device)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=result)


@app.command("verify-index")
def verify_index_command() -> None:
    """Recheck P3 source/model fingerprints, Qdrant dimension, and scoped routes."""
    result = verify_index(PROJECT_ROOT, PROJECT_ROOT / "evidence/p3/validation.json")
    console.print_json(data=result)


@app.command("p0-audit")
def p0_audit() -> None:
    """Write source, workflow, model, and legacy baseline artifacts."""
    result = run_p0(PROJECT_ROOT)
    console.print_json(data=result)


@app.command("doctor")
def doctor(
    output: Annotated[Path | None, typer.Option(help="Optional JSON report path")] = None,
    probe_bf16: Annotated[bool, typer.Option(help="Run a small BF16 CUDA computation")] = False,
) -> None:
    """Check local assets and current hardware without changing services."""
    result = run_doctor(PROJECT_ROOT, probe_bf16=probe_bf16)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=result)


@app.command("smoke")
def smoke(
    model: Annotated[str, typer.Option(help="baseline or target")] = "baseline",
    output: Annotated[Path | None, typer.Option(help="Optional JSON report path")] = None,
) -> None:
    """Test text, JSON, and full tool roundtrip on a dedicated local server."""
    try:
        result = run_smoke(PROJECT_ROOT, profile=model)
    except (LocalModelError, KeyError, ValueError) as error:
        console.print(f"[red]Smoke test failed:[/red] {error}")
        raise typer.Exit(1) from error
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=result)
    checks = [result["text"], result["json"], result["tool_roundtrip"]]
    if "reasoning" in result:
        checks.append(result["reasoning"])
    if not all(item["pass"] for item in checks):
        raise typer.Exit(1)


@app.command("serve")
def serve(
    model: Annotated[str, typer.Option(help="baseline or target")] = "baseline",
    gpu: Annotated[int, typer.Option(min=0, help="Physical GPU index")] = 0,
    max_model_len: Annotated[
        int | None, typer.Option(help="16384 or 32768 for the staged smoke")
    ] = None,
    max_num_seqs: Annotated[int | None, typer.Option(min=1, max=2)] = None,
) -> None:
    """Start an independent loopback model process when GPU headroom permits."""
    try:
        launch_local_model(PROJECT_ROOT, model, gpu, max_model_len, max_num_seqs)
    except ServiceLaunchBlocked as error:
        console.print(f"[yellow]Launch blocked:[/yellow] {error}")
        raise typer.Exit(2) from error


@app.command("retrieval-smoke")
def retrieval_smoke(
    device: Annotated[str, typer.Option(help="cpu or cuda:0 in the retrieval worker")] = "cpu",
    output: Annotated[Path | None, typer.Option(help="Optional JSON report path")] = None,
) -> None:
    """Load both local retrieval models and check shape, norm, ranking, and stability."""
    from math import isfinite, sqrt

    config = load_model_config(PROJECT_ROOT / "config" / "models.yaml")
    query = "PFCP节点不可达告警如何定位？"
    positive = "PFCP节点不可达告警：检查N4链路和心跳超时。"
    negative = "苹果蛋糕制作需要面粉、牛奶和烤箱。"
    try:
        with RetrievalWorker(PROJECT_ROOT, "embed", config.embedding.path, device=device) as worker:
            first = worker.request({"role": "query", "texts": [query]})
            repeated = worker.request({"role": "query", "texts": [query]})
            docs = worker.request({"role": "document", "texts": [positive, negative]})
        vector = first["vectors"][0]
        assert isinstance(vector, list)
        norm = sqrt(sum(float(value) ** 2 for value in vector))
        embedding_ok = (
            first["dimension"] == config.embedding.dimension
            and docs["dimension"] == config.embedding.dimension
            and all(isfinite(float(value)) for value in vector)
            and abs(norm - 1.0) < 1e-3
            and first["vectors"] == repeated["vectors"]
        )
        with RetrievalWorker(PROJECT_ROOT, "rerank", config.reranker.path, device=device) as worker:
            scores = worker.request({"pairs": [[query, positive], [query, negative]]})
        ranking_ok = (
            len(scores["raw_logit_difference"]) == 2
            and scores["raw_logit_difference"][0] > scores["raw_logit_difference"][1]
            and all(0.0 < value < 1.0 for value in scores["p_yes"])
        )
    except (RetrievalWorkerError, OSError, ValueError, AssertionError) as error:
        console.print(f"[red]Retrieval smoke failed:[/red] {error}")
        raise typer.Exit(1) from error
    result = {
        "device": device,
        "embedding": {"pass": embedding_ok, "dimension": first["dimension"], "l2_norm": norm},
        "reranker": {
            "pass": ranking_ok,
            "raw_logit_difference": scores["raw_logit_difference"],
            "p_yes": scores["p_yes"],
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print_json(data=result)
    if not (embedding_ok and ranking_ok):
        raise typer.Exit(1)
