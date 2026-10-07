"""Reviewed development and one-use heldout execution; no automatic semantic signoff."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from upf_agent.agent_contracts import AgentData, AgentTask
from upf_agent.agent_engine import run_agent
from upf_agent.agent_worker import CountedClient
from upf_agent.config import load_model_config
from upf_agent.p3_answer import answer_from_retrieval
from upf_agent.p3_search import search
from upf_agent.p6_contracts import ranking_metrics, read_suite
from upf_agent.p6_evaluation import pipeline_identity, reserve_heldout
from upf_agent.quality_repair import sha, write_json
from upf_agent.retrieval import RetrievalWorker
from upf_agent.topology import import_topology, lookup_topology


def freeze_configuration(
    root: Path, suite: Path, development: Path, decision: Path, output: Path
) -> dict:
    dev = read_suite(suite)
    heldout = read_suite(suite, "heldout")
    report, human = json.loads(development.read_text()), json.loads(decision.read_text())
    if len(dev) != 60 or len(heldout) != 60:
        raise ValueError("Final release requires the reviewed 60/60 protocol")
    if (
        report.get("suite_sha256") != sha(suite)
        or report.get("pipeline_identity") != pipeline_identity(root)
        or report.get("label_mode") != "reviewed"
        or report.get("split") != "dev"
        or report.get("cases") != 60
    ):
        raise ValueError("Need a reviewed development run from the current runtime")
    if (
        human.get("configuration_accepted") is not True
        or human.get("development_sha256") != sha(development)
        or not human.get("reviewer")
        or not human.get("date")
        or human.get("model_profile") != report["model_profile"]
    ):
        raise ValueError("Human development decision is missing or not bound to this run")
    adjudication = (root / str(human.get("adjudication_path", ""))).resolve()
    if not adjudication.is_relative_to(root.resolve()) or not adjudication.is_file():
        raise ValueError("Need a project-local human answer adjudication")
    judged = json.loads(adjudication.read_text())
    if (
        judged.get("summary_sha256") != sha(development)
        or judged.get("human_review_complete") is not True
        or judged.get("semantic_gate_pass") is not True
    ):
        raise ValueError("Development semantic gates have not passed")
    result = {
        "status": "frozen_after_reviewed_development",
        "suite_sha256": sha(suite),
        "pipeline_identity": pipeline_identity(root),
        "model_profile": report["model_profile"],
        "development_sha256": sha(development),
        "decision_sha256": sha(decision),
        "adjudication_sha256": sha(adjudication),
        "frozen_at": datetime.now(UTC).isoformat(),
        "final_semantic_support_must_be_reviewed": True,
    }
    if output.exists():
        raise ValueError("Frozen configuration cannot be overwritten")
    write_json(output, result)
    return result


def evaluate_reviewed(
    root: Path, suite: Path, profile: str, split: str, output: Path, frozen: Path | None = None
) -> dict:
    if output.exists():
        raise ValueError("Use a new evaluation output directory")
    if split == "heldout":
        if frozen is None:
            raise ValueError("Heldout requires a frozen configuration")
        if json.loads(frozen.read_text())["model_profile"] != profile:
            raise ValueError("Model differs from frozen configuration")
        cases = reserve_heldout(root, suite, frozen, output)
    elif split == "dev":
        cases = read_suite(suite)
    else:
        raise ValueError("Unknown evaluation partition")
    output.mkdir(parents=True)
    model = load_model_config(root / "config/models.yaml")
    rows = []
    with RetrievalWorker(root, "embed", model.embedding.path, "cuda:1") as embedding:
        with RetrievalWorker(root, "rerank", model.reranker.path, "cuda:1") as reranking:
            for case in cases:
                try:
                    if case.category in {"command", "knowledge", "troubleshooting"}:
                        result = search(
                            root,
                            case.question,
                            case.product,
                            case.version,
                            "cuda:1",
                            (embedding, reranking),
                        )
                        task = AgentTask(
                            run_id=uuid4().hex,
                            question=case.question,
                            task_type="manual",
                            model_profile=profile,
                        )
                        with CountedClient(root, AgentData(task=task), 180, 4) as client:
                            answer = answer_from_retrieval(root, case.question, result, client)
                        status = answer["status"]
                        payload = {
                            "retrieval": result,
                            "answer": answer,
                            "retrieval_metrics": {
                                name: ranking_metrics(ids, {eid: 1 for eid in case.evidence_ids})
                                for name, ids in result["rankings"].items()
                            },
                        }
                    elif case.category == "topology":
                        path = (root / case.evidence_ids[0].split(":row")[0]).resolve()
                        if not path.is_relative_to((root / "eval/p6").resolve()):
                            raise ValueError("Topology truth source outside P6 fixture directory")
                        source = import_topology(path, root / "data/p6/topology", synthetic=True)
                        number = case.source_groups[0].split(":")[-1]
                        at = datetime.fromisoformat(
                            "2025-06-01T00:00:00+00:00"
                            if "邻接" in case.question
                            else "2026-06-01T00:00:00+00:00"
                        )
                        result = lookup_topology(
                            root / "data/p6/topology" / source["generation"],
                            f"SYN-UPF-{number}",
                            "N3",
                            at,
                        )
                        status, payload = result.status.value, result.model_dump(mode="json")
                        payload["n30_negative_control"] = lookup_topology(
                            root / "data/p6/topology" / source["generation"],
                            f"SYN-UPF-{number}",
                            "N30",
                            at,
                        ).model_dump(mode="json")
                    else:
                        task = AgentTask(
                            run_id=uuid4().hex,
                            question=case.question,
                            task_type="diagnose" if case.category == "log" else "other",
                            log_generation=case.log_generation,
                            product=case.product,
                            version=case.version,
                            model_profile=profile,
                        )
                        state = run_agent(root, task)
                        status, payload = state.report.status.value, state.model_dump(mode="json")
                    failure = None
                except (RuntimeError, ValueError, OSError) as error:
                    status, payload, failure = "failed", {}, str(error)
                row = {
                    "case": case.id,
                    "category": case.category,
                    "synthetic": case.synthetic,
                    "status": status,
                    "expected_status": status in case.expected_statuses,
                    "failure": failure,
                    "human_answer_semantics": "pending",
                }
                rows.append(row)
                write_json(output / f"{case.id}.json", {"row": row, "result": payload})
                write_json(output / "progress.json", {"completed": len(rows), "rows": rows})
    summary = {
        "model_profile": profile,
        "split": split,
        "cases": len(rows),
        "rows": rows,
        "suite_sha256": sha(suite),
        "pipeline_identity": pipeline_identity(root),
        "label_mode": "reviewed",
        "human_answer_semantic_review_complete": False,
        "M6_pass": False,
        "formal_business_accuracy_claim": False,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "answer_review.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "rows": [
                {
                    "case": row["case"],
                    "supported": None,
                    "complete": None,
                    "command_exact": None,
                    "unsupported_critical_parameter": None,
                    "missing_source_correct": None,
                    "reviewer": None,
                    "date": None,
                    "notes": "待人工核对答案",
                }
                for row in rows
            ],
        },
    )
    write_json(output / "summary.json", summary)
    return summary
