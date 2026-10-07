"""Archive actual development evidence and report scopes, denominators and failures."""

import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from upf_agent.p6_evaluation import pipeline_identity
from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/p6"
DEST = ROOT / "evidence/p6"


def main():
    if (DEST / "summary.json").exists():
        raise ValueError("Archive an existing P6 release before creating another")
    required = [RUN / "retrieval-dev/manifest.json"]
    for profile in ("baseline", "target"):
        required += [
            RUN / f"model-{profile}/summary.json",
            RUN / f"parameters-{profile}-v2.json",
            RUN / f"agents-{profile}/summary.json",
            RUN / f"rag-{profile}.json",
        ]
        generation_profile = "baseline" if profile == "baseline" else "target-off"
        required += [
            RUN / f"perf-{generation_profile}-{context}-c{concurrency}.json"
            for context in (16384, 32768)
            for concurrency in (1, 2)
        ]
    required += [RUN / f"perf-target-16384-c{n}.json" for n in (1, 2)]
    required.append(RUN / "runner-performance.json")
    if missing := [str(p) for p in required if not p.is_file()]:
        raise ValueError(f"Development probes not complete: {missing}")
    suites = {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in (ROOT / "eval/p6/candidates.jsonl").read_text().splitlines()
        )
    }
    models, parameters, performance, agents, rag = {}, {}, {}, {}, {}
    failures, pair_review = (
        [],
        [
            "# P6 开发草案模型答案对照",
            "",
            "同一检索文件；机器检查与相关章节命中均不是人工业务正确率。",
            "此处不包含heldout。旧草案兼容/预算实测，不是新版完整业务集成绩。",
            "正式业务真值复核入口是eval/p6/business_v3/truth_review.md。",
            "",
        ],
    )
    for profile in ("baseline", "target"):
        source = RUN / f"model-{profile}"
        report = json.loads((source / "summary.json").read_text())
        if report["cases"] != 42 or len(report["rows"]) != 42:
            raise ValueError("Need the complete 42-case development comparison")
        shutil.copytree(source, DEST / f"model-{profile}", dirs_exist_ok=True)
        models[profile] = {
            key: report[key]
            for key in (
                "cases",
                "machine_pass",
                "latency_ms",
                "generation_requests",
                "pipeline_identity",
            )
        }
        models[profile]["draft_source_group_hits"] = sum(
            row["draft_source_group_hit"] is True for row in report["rows"]
        )
        models[profile]["draft_source_group_denominator"] = sum(
            row["draft_source_group_hit"] is not None for row in report["rows"]
        )
        models[profile]["categories"] = dict(
            Counter(suites[row["id"]]["category"] for row in report["rows"])
        )
        for row in report["rows"]:
            case = suites[row["id"]]
            cached = json.loads((RUN / "retrieval-dev" / f"{row['id']}.json").read_text())
            literal = (
                case["expected_extracts"][0].splitlines()[1]
                if case["category"] == "command"
                else None
            )
            literal_present = (
                any(
                    literal in item["text"]
                    or any(literal in neighbor["text"] for neighbor in item.get("neighbors", []))
                    for item in cached["evidence"]
                )
                if literal
                else None
            )
            if (
                not row["machine_checks"]["machine_contract_pass"]
                or row["draft_source_group_hit"] is False
            ):
                answer = json.loads((source / f"{row['id']}.json").read_text())["answer"]
                failure = {
                    "profile": profile,
                    "case": row["id"],
                    "machine_checks": row["machine_checks"],
                    "source_group_hit": row["draft_source_group_hit"],
                    "failure": row["failure"],
                    "answer_error": answer.get("error"),
                    "validation_errors": answer.get("validation_errors"),
                    "relevance_gate": answer.get("relevance_gate"),
                    "candidate_command_literal_in_context": literal_present,
                    "stage_proxy": "model_or_publication"
                    if literal_present
                    else "requires_source_review",
                }
                failures.append(failure)
        param = RUN / f"parameters-{profile}-v2.json"
        shutil.copy2(param, DEST / param.name)
        d = json.loads(param.read_text())
        if d["cases"] != 100 or d["families"] != 10:
            raise ValueError("Need all 100 parameter requests across 10 families")
        parameters[profile] = {
            key: d[key]
            for key in (
                "cases",
                "families",
                "first_pass",
                "after_one_repair",
                "by_family",
                "tools_executed",
                "prompt_version",
                "sampling_note",
            )
        }
        a = RUN / f"agents-{profile}"
        if (a / "summary.json").exists():
            shutil.copytree(a, DEST / a.name, dirs_exist_ok=True)
            agents[profile] = json.loads((a / "summary.json").read_text())
            if agents[profile]["cases"] != 5 or len(agents[profile]["rows"]) != 15:
                raise ValueError("Need all five log cases across three routes")
        r = RUN / f"rag-{profile}.json"
        if r.exists():
            shutil.copy2(r, DEST / r.name)
            data = json.loads(r.read_text())
            if data["complete_answer_ms"]["n"] != 20:
                raise ValueError("Need the 20 warm full-RAG measurements")
            rag[profile] = {
                key: data[key]
                for key in ("case", "complete_answer_ms", "first_visible_ms", "resources", "scope")
            }
    for case in (item for item in suites.values() if item["split"] == "dev"):
        if not (RUN / "model-baseline" / f"{case['id']}.json").exists():
            continue
        pair_review += [
            f"## {case['id']} · {case['category']}",
            "",
            case["question"],
            "",
            "- 人工相关性/完整性/版本/关键参数判断：待复核",
            "",
        ]
        for profile in ("baseline", "target"):
            record = json.loads((RUN / f"model-{profile}" / f"{case['id']}.json").read_text())
            pair_review += [
                f"### {profile}",
                "",
                f"状态：{record['answer']['status']}；机器契约：{record['row']['machine_checks']['machine_contract_pass']}",
                "",
            ]
            for claim in record["answer"]["claims"]:
                pair_review += [
                    "```text",
                    claim["text"],
                    "```",
                    "引用：" + ", ".join(cite["chunk_id"] for cite in claim["citations"]),
                    "",
                ]
    (DEST / "model_answer_review.md").write_text("\n".join(pair_review) + "\n")
    for p in RUN.glob("perf-*.json"):
        shutil.copy2(p, DEST / p.name)
        d = json.loads(p.read_text())
        if len(d["rows"]) != 2 or any(row["total_ms"]["n"] != 20 for row in d["rows"]):
            raise ValueError("Need 20 warm measurements for both context budgets")
        performance[p.stem] = {
            key: d[key]
            for key in (
                "profile",
                "server_context_capacity",
                "client_concurrency",
                "server_max_num_seqs",
                "resources",
                "pipeline_identity",
            )
        }
        performance[p.stem]["thinking_policy"] = d.get(
            "thinking_policy", "auto (as-run source, not an original metadata field)"
        )
        performance[p.stem]["chat_template_kwargs"] = d.get(
            "chat_template_kwargs",
            {"reasoning_effort": "low"} if d["profile"] == "target" else None,
        )
        performance[p.stem]["budgets"] = [
            {
                key: row[key]
                for key in (
                    "requested_context_budget_characters",
                    "total_ms",
                    "first_visible_ms",
                    "nonempty_completed",
                )
            }
            for row in d["rows"]
        ]
        for budget, raw in zip(performance[p.stem]["budgets"], d["rows"], strict=True):
            budget["http_or_parse_failures"] = sum(
                row["failure"] is not None for row in raw["runs"]
            )
            budget["finish_reasons"] = dict(Counter(row["finish_reason"] for row in raw["runs"]))
    shutil.copytree(RUN / "retrieval-dev", DEST / "retrieval-dev", dirs_exist_ok=True)
    for name in ("parameters-baseline.json", "runner.json", "runner-performance.json"):
        shutil.copy2(RUN / name, DEST / name)
    (DEST / "execution-logs").mkdir(exist_ok=True)
    for path in RUN.glob("*.log"):
        shutil.copy2(path, DEST / "execution-logs" / path.name)
    incomplete = RUN / "model-target-low-thinking-incomplete"
    if incomplete.exists():
        shutil.copytree(incomplete, DEST / incomplete.name, dirs_exist_ok=True)
    retrieval = json.loads((RUN / "retrieval-dev/manifest.json").read_text())
    gates = {
        "business_labels_reviewed": 0,
        "human_answers_reviewed": 0,
        "heldout_executed": False,
        "heldout_marker_absent": not (ROOT / "eval/p6/heldout_consumed.json").exists(),
        "model_switch_approved": False,
        "M6_pass": False,
        "full_agent_business_gain_validated": False,
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "development_probes_complete_formal_acceptance_pending",
        "pipeline_identity": pipeline_identity(ROOT),
        "suite_sha256": sha(ROOT / "eval/p6/candidates.jsonl"),
        "formal_business_suite_sha256": sha(ROOT / "eval/p6/business_v3/candidates.jsonl"),
        "formal_business_suite_path": "eval/p6/business_v3/candidates.jsonl",
        "formal_business_suite_evaluated": False,
        "execution_policy": {
            "baseline": "original request serialization unchanged",
            "target": "structured JSON disables thinking; plain replies retain low thinking",
            "same_retrieval_files_and_budget": True,
            "entire_code_fingerprint_identical_between_profiles": False,
            "as_run_sources": [
                "archive/low-thinking",
                "archive/structured-v2",
                "archive/formal-preflight",
                "archive/optimized-performance-before-row-fix",
                "archive/optimized-performance",
            ],
            "generation_optimization": "target-off benchmark only; Agent plain policy remains low",
            "partial_low_thinking_run_preserved": True,
        },
        "models": models,
        "parameters": parameters,
        "agents": agents,
        "rag": rag,
        "performance": performance,
        "performance_profiles": len(performance),
        "performance_main_profiles": 8,
        "performance_low_thinking_failure_profiles": 2,
        "performance_measured_requests": sum(
            budget["total_ms"]["n"] for p in performance.values() for budget in p["budgets"]
        ),
        "retrieval": {
            key: retrieval[key] for key in ("model_loading_ms", "warm_retrieval_ms", "resources")
        },
        "gates": gates,
        "default_model": "baseline",
        "formal_business_accuracy_claim": False,
    }
    write_json(
        DEST / "failures.json",
        {
            "scope": "draft machine/selection failure classification; human truth pending",
            "rows": failures,
        },
    )
    write_json(DEST / "summary.json", summary)
    write_results(summary)
    print(
        json.dumps(
            {
                "models": models,
                "parameters": {
                    key: {k: v for k, v in d.items() if k in ("first_pass", "after_one_repair")}
                    for key, d in parameters.items()
                },
                "performance_profiles": len(performance),
            },
            ensure_ascii=False,
        )
    )


def write_results(summary):
    lines = [
        "# P6 本机开发实测结果",
        "",
        "自动实测已完成；M6正式验收未完成。保留2507默认模型，3.8继续候选。",
        "42题及以下指标属于旧助手草案；不是新版完整120题成绩，也不是人工业务正确率。",
        "",
        "| 模型 | 机器契约 | 草案相关章节命中 | 问答p50/p95（秒） | 模型请求数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in summary["models"].items():
        t = row["latency_ms"]
        lines.append(
            f"| {name} | {row['machine_pass']}/{row['cases']} | "
            f"{row['draft_source_group_hits']}/{row['draft_source_group_denominator']} | "
            f"{t['p50'] / 1000:.2f}/{t['p95'] / 1000:.2f} | {row['generation_requests']} |"
        )
    lines += [
        "",
        "热检索42题的p50/p95为2.09/2.65秒，模型加载约13.49秒；GPU1采样峰值26030 MiB。"
        "此缓存用于两模型同证据开发对照，未获得人工相关性，不能当作正式Recall/nDCG。",
        "",
        "两模型逐文件复用相同证据、4次请求预算与发布检查。3.8首版low-thinking仅执行5题后因短JSON节点无可见内容终止；部分失败完整保留。",
        "3.8的结构化节点随后关闭思考，普通输出仍low；基线请求序列化不变。两阶段代码指纹不同，实际源代码分别归档，不能宣称完全相同模型适配参数。",
        "命令漏答、错章节、误拒仍存在，见failures.json和model_answer_review.md；机器契约通过不能证明回答完整。",
        "",
        "参数测试：",
        "",
    ]
    for name, row in summary["parameters"].items():
        lines.append(
            f"- {name}：首次 {row['first_pass']}/{row['cases']}，"
            f"一次修复后 {row['after_one_repair']}/{row['cases']}；"
            f"实际工具执行{row['tools_executed']}。"
        )
    lines += [
        "",
        "100次是10类各10次确定性重复。schema-only旧提示90/100首次、100/100修复已保留；新版明确用户确认范围后改善，不把提示修订成绩合并成一次独立盲测。",
        "",
        "| 模型 | 日志路线 | 原生目标证据命中 | 总工具/模型次数 | 每题耗时范围（秒） |",
        "|---|---|---:|---:|---:|",
    ]
    for name, report in summary["agents"].items():
        for mode in ("single_evidence_pass", "fixed_tools", "finite_agent"):
            rows = [r for r in report["rows"] if r["mode"] == mode]
            times = [r["elapsed_ms"] / 1000 for r in rows]
            lines.append(
                f"| {name} | {mode} | "
                f"{sum(r['requested_event_evidence_hit'] for r in rows)}/{len(rows)} | "
                f"{sum(r['tool_attempts'] for r in rows)}/{sum(r['model_calls'] for r in rows)} | "
                f"{min(times):.2f}–{max(times):.2f} |"
            )
    lines += [
        "",
        "固定工具预先获得明确事件类型，单次路线只有统计摘要。此实验不是完整单次手册RAG或复杂跨资料Agent业务增益；不报告根因准确率。",
        "",
        "完整RAG（单题20次、热检索、非流式经校验发布）：",
        "",
    ]
    for name, row in summary["rag"].items():
        t = row["complete_answer_ms"]
        lines.append(
            f"- {name}：n={t['n']}，p50/p95={t['p50'] / 1000:.2f}/{t['p95'] / 1000:.2f}秒。"
        )
    lines += [
        "",
        "重复题仅用于计时，已有语义疑点，不证明可交付质量。CLI无答案流式发布，不能用生成组件TTFT证明普通问答首输出p95≤10秒。",
        "",
        "| 生成组件 | 提示预算（字符） | n | 总时延p50/p95（秒） | 有可见内容n | 非空且正常结束 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, profile in sorted(summary["performance"].items()):
        for row in profile["budgets"]:
            t = row["total_ms"]
            lines.append(
                f"| {name} | {row['requested_context_budget_characters']} | {t['n']} | "
                f"{t['p50'] / 1000:.2f}/{t['p95'] / 1000:.2f} | "
                f"{row['first_visible_ms']['n']} | {row['nonempty_completed']}/{t['n']} |"
            )
    lines += [
        "",
        "主矩阵为基线4档与target-off优化4档；另保留target low的16K并发1/2失败对照，"
        "共400次计时请求，另有20次不计入统计的预热。",
        "目标low在并发1的40次全部截断，并发2的40次中34次截断、6次非空正常结束，"
        "故停止32K low扩测，转测关闭思考的完整矩阵。target-off四档160次均非空正常结束；"
        "这里只验证生成组件，不改变普通Agent的low策略或证明业务质量改善。",
        "16K/32K是服务容量，输入是同样短提示；不能据此声称32K长输入更优。"
        "输出上限256 tokens，low档的思考也消耗该预算；重复前缀缓存可使计时偏乐观。",
        "旧low探针只监听reasoning_content而未监听reasoning别名，其first_model_event_ms可能漏计思考事件；可见content与总时延指标不受此问题影响，未据此判首输出SLA。后续探针已兼容两种字段。",
        "GPU/RSS各文件独立保存，0.5秒采样含共享任务占用，RSS为进程树求和、不是去重物理内存；完整RAG采样不包含生成服务RSS。",
        "生成矩阵GPU0采样峰值69353 MiB（约67.73 GiB），生成服务进程树RSS峰值约4.06 GiB；"
        "各档资源采样无错误，计时请求无HTTP/解析错误，但low档截断仍计失败。"
        "矩阵服务max_num_seqs=2，分别施加客户端并发1/2；默认配置保留max_num_seqs=1。"
        "本轮自启服务已停止，两卡显存归零，用户原8000端口进程未停止。",
        "",
        "下一步先完成eval/p6/business_v3/truth_review.md；执行并复核新版开发集，修复失败，满足质量和资源门槛后再冻结、一次运行保留集。",
        "检索Recall/nDCG、人工支持率/命令/缺资料评分、复杂Agent增益及正式冻结验收仍待完成；现场时区、设备身份、真实拓扑和P7离线隔离也未验收。",
    ]
    (DEST / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
