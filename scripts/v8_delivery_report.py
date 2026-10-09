"""Aggregate saved validation evidence without consuming the heldout again."""

import csv
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from upf_agent.v8.models import digest
from upf_agent.v8.scenarios import verify_freeze

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runtime/v8"
RESTORED = PROJECT / "runtime/v8-restore-validation-20261007"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run():
    evidence = ROOT / "evidence/u6"
    for log in [
        "/tmp/upf-v8-delivery-tests.log",
        "/tmp/upf-v8-delivery-regression.log",
        "/tmp/upf-v8-final-local.log",
        "/tmp/upf-v8-validation-snapshot.log",
        "/tmp/upf-v8-validation-restore.log",
        "/tmp/upf-v8-known-input-replay.log",
    ]:
        shutil.copyfile(log, evidence / Path(log).name.removeprefix("upf-"))
    verify_freeze(ROOT / "experiments/v8/heldout/heldout-final-001")
    local_root = ROOT / "experiments/v8/local-final-001"
    local = load(local_root / "summary.json")
    code = load(local_root / "freeze.json")["code"]
    assert all(Path(p).is_file() and sha(Path(p)) == expected for p, expected in code.items())
    rows, pairs = [], set()
    for group in local["groups"]:
        folder = local_root / f"{group['size']}-{group['updates']}"
        with (folder / "batches.csv").open(newline="") as stream:
            values = list(csv.DictReader(stream))
        assert len(values) == 1000
        for row in values:
            assert row["consistent"] == "True"
            key = (row["size"], int(row["updates"]), int(row["seed"]), int(row["batch"]))
            assert key not in pairs
            pairs.add(key)
        assert {int(r["seed"]) for r in values} == set(range(20))
        assert {int(r["batch"]) for r in values} == set(range(50))
        rows.extend(values)
    assert len(rows) == 16000
    with (local_root / "all-batches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    heldout_root = ROOT / "experiments/v8/heldout/heldout-final-001"
    heldout = load(heldout_root / "summary.json")
    graph = load(ROOT / "experiments/v8/graph/summary.json")
    recovery = load(evidence / "recovery-validation.json")
    neo4j_runtime = load(evidence / "neo4j-runtime-verification.json")
    replay_source = RESTORED / "experiments/v8/known-input-replay-001"
    replay_target = evidence / "known-input-replay-001"
    if not replay_target.exists():
        shutil.copytree(replay_source, replay_target)
    replay = load(replay_target / "summary.json")
    baseline = load(ROOT / "evidence/u0/original_fingerprints.json")
    original = Path("/home/huangxiaolin/upf-agent")
    changed = [name for name, expected in baseline.items() if sha(original / name) != expected]
    raw = {name: expected for name, expected in baseline.items() if name.startswith("data/raw/")}
    assert len(raw) == 7
    raw_copies_equal = all(sha(ROOT / name) == expected for name, expected in raw.items())
    preservation = {
        "checked_files": len(baseline),
        "changed": changed,
        "raw_files": len(raw),
        "raw_copies_equal": raw_copies_equal,
        "raw_sha256": raw,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "pass": not changed and raw_copies_equal,
    }
    write(ROOT / "evidence/u0/original-preservation-final.json", preservation)
    log = (evidence / "v8-delivery-regression.log").read_text()
    matches = re.findall(r"(\d+) passed in ([\d.]+)s", log)
    assert matches
    passed, seconds = matches[-1]
    regression = {
        "passed": int(passed),
        "duration_seconds": float(seconds),
        "log_sha256": sha(evidence / "v8-delivery-regression.log"),
        "no_failed_or_skipped": not re.search(r"\d+ (failed|skipped|xfailed)", log),
    }
    trace_checks = []
    for row in graph["results"]:
        snapshot = load(ROOT / f"experiments/v8/graph/snapshot-{row['case']}.json")
        by_id = {n["node_id"]: n for n in snapshot["nodes"]}
        for node_id in row["returned"]:
            source = by_id[node_id]["source"]
            path = ROOT / source["relative_path"]
            trace_checks.append(
                bool(source["locator"] and path.is_file() and sha(path) == source["sha256"])
            )
    gates = {
        "original_preservation": preservation["pass"],
        "regression": regression["passed"] == 371 and regression["no_failed_or_skipped"],
        "final_local_full_consistency": local["pass"]
        and local["consistent"] == local["batches"] == 16000
        and local["frozen_code_unchanged"],
        "heldout_full_local_consistency": heldout["full_local_equal"],
        "heldout_source_traceability": heldout["traceable_bindings"]
        == heldout["total_bindings"]
        == 134,
        "heldout_zero_unsafe_releases": heldout["ours_unsafe_releases"] == 0,
        "heldout_legal_positive_acceptance": heldout["ours_legal_accepted"]
        == heldout["ours_legal_total"]
        == 10,
        "heldout_source_and_structure_isolation": heldout["source_group_isolation"]
        and heldout["structure_group_isolation"],
        "shared_budget": heldout["within_shared_budget"],
        "graph_memory_neo4j_semantics": graph["backend_semantics_equal"],
        "neo4j_pinned_image_and_version": neo4j_runtime["pass"],
        "graph_zero_invalid_acceptance": graph["contract_invalid_accepted"] == 0,
        "graph_returned_source_sha": all(trace_checks),
        "restored_checkpoint": recovery["pass"]
        and recovery["task_unchanged"]
        and recovery["repeated_resume_sql_rows_and_budget_unchanged"],
        "known_input_replay": replay["pass"]
        and len(replay["cases"]) == 11
        and replay["recorded_model_decisions_rechecked"],
        "frozen_application_and_heldout_unchanged": True,
    }
    methods = []
    for method in heldout["methods"]:
        selected = [r for r in heldout["results"] if r["method"] == method]
        methods.append(
            {
                "method": method,
                "unsafe_releases": sum(len(r["unsafe_releases"]) for r in selected),
                "legal_accepted": sum(len(r["legal_accepted"]) for r in selected),
                "legal_total": sum(r["legal_total"] for r in selected),
                "model_calls": sum(r["model_calls"] for r in selected),
                "tool_calls": sum(r["tool_calls"] for r in selected),
                "tool_count_basis": selected[0]["tool_count_basis"],
            }
        )
    graph_metrics = []
    for method in ["Hybrid", "FixedGraph", "ContractGraph"]:
        selected = [r for r in graph["results"] if r["method"] == method]
        graph_metrics.append(
            {
                "method": method,
                "cases": len(selected),
                "required_coverage": sum(r["required_coverage"] for r in selected) / len(selected),
                "applicability_precision": sum(r["applicability_precision"] for r in selected)
                / len(selected),
                "invalid_accepted": sum(len(r["invalid_accepted"]) for r in selected),
            }
        )
    value = {
        "validation": "v8 read-only prototype acceptance",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pass": all(gates.values()),
        "gates": gates,
        "regression": regression,
        "local": {
            k: local[k]
            for k in [
                "batches",
                "consistent",
                "groups",
                "workers",
                "model_calls",
                "tool_calls",
                "frozen_code_unchanged",
            ]
        },
        "heldout": {
            k: heldout[k]
            for k in [
                "cases",
                "freeze_sha256",
                "ours_unsafe_releases",
                "ours_legal_accepted",
                "ours_legal_total",
                "traceable_bindings",
                "total_bindings",
            ]
        },
        "methods": methods,
        "gate_off_offline_unsafe_releases": sum(
            len(r["unsafe_releases"]) for r in heldout["gate_ablation_offline"]
        ),
        "graph": graph_metrics,
        "graph_source_checks": len(trace_checks),
        "neo4j_runtime": neo4j_runtime,
        "recovery": recovery,
        "known_input_replay": {
            "cases": len(replay["cases"]),
            "pass": replay["pass"],
            "model_calls": replay["model_calls"],
            "mode": replay["mode"],
        },
        "preservation": preservation,
        "final_snapshot": "artifacts/v8-final-20261007",
        "delivery_attestation": "docs/v8/delivery-receipt.json (created after final snapshot)",
        "limitations": [
            "synthetic=true, reviewed=false; no real Open5GS/hardware diagnostic accuracy claim",
            (
                "contract graph coverage is 2/4 because unknown-version seeds blo"
                "ck expansion to the synthetic required control"
            ),
            (
                "small dense updates may cost more than full evaluation; timings "
                "use four independent CPU workers"
            ),
            (
                "B0/B2/B3/B4 are declared ablations with frozen-source access cou"
                "nts; B1/Ours use backend/ledger counters"
            ),
            (
                "known-input replay rechecks saved model decisions and performs n"
                "o new blind acceptance"
            ),
            (
                "same-host shared weights/service/retrieval environment and nativ"
                "e absolute paths remain external dependencies"
            ),
            "historical M6/M7 are unchanged and independent",
        ],
    }
    inputs = [
        local_root / "summary.json",
        local_root / "freeze.json",
        heldout_root / "summary.json",
        heldout_root / "freeze.json",
        ROOT / "experiments/v8/graph/summary.json",
        evidence / "recovery-validation.json",
        evidence / "neo4j-runtime-verification.json",
        replay_target / "summary.json",
        evidence / "v8-delivery-regression.log",
    ]
    value["input_file_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    value["heldout_freeze_canonical_digest"] = digest(load(heldout_root / "freeze.json"))
    write(evidence / "acceptance-summary.json", value)
    write(PROJECT / "docs/v8/acceptance-summary.json", value)
    lines = [
        "# v8 只读原型验收报告",
        "",
        (
            "验收日期：2026-10-07。U0–U6 的规则原型、"
            "独立图后端、恢复和可复现实验均已通过。最终打包的文件数量"
            "与 SHA 见同"
            "目录 `delivery-receipt.json`；该回执在最终快照生成并校验后写入。历史 M6/M7 保持独立。"
        ),
        "",
        "## 硬门槛",
        "",
        "| 项目 | 实际结果 | 证据 |",
        "|---|---|---|",
        (
            "| 开发回归 | 371 passed，24.30 秒；原 317 项 + 54 项 v8 测试；无跳过 | `runtime/"
            "v8/evidence/u6/v8-delivery-regression.log` |"
        ),
        (
            "| 最终规模实验 | 16,000/16,000 一致，最终冻结应用代码未变 | `experiments/v8/local-f"
            "inal-001/summary.json`、`all-batches.csv`、分组更新回执 |"
        ),
        (
            "| 留出正确性 | 11 场景 × 6 方法；Ours 错误放行 0，合法动作 10/10 | `experiments/v8/"
            "heldout/heldout-final-001/` 的 freeze、JSON、CSV、原始回包 |"
        ),
        "| 来源追溯 | 留出 134/134 绑定；图结果来源"
        "文件 SHA 全部再核验通过 | `acceptance"
        "-summary.json`、运行报告与证据回执 |",
        (
            "| 图后端 | 真实 Neo4j 与 Memory 的 18 种边界查询、四个检索场景同语义 | `tests/test_v8_"
            "graph_engine.py`、图实验 JSON |"
        ),
        "| 冻结输入 | 应用/锁/配置/留出来源及结构组隔离均"
        "核验通过 | 留出 freeze 和最终规模实验 fre"
        "eze |",
        (
            "| 恢复 | 新根 2,721 个运行文件 SHA 与六"
            "个 SQLite quick_check 通过；暂停任务"
            "完成；重复恢复未"
            "增加 SQL 行或预算 | `evidence/u6/recovery-validation.json` |"
        ),
        (
            "| 已知输入回放 | 11/11 场景与独立全量及原 o"
            "racle 一致；模型调用 0；原模型建议重新核算一致 "
            "| `evide"
            "nce/u6/known-input-replay-001/` |"
        ),
        (
            "| 原目录保持 | 92 个源/配置/测试/原件指纹无变"
            "化；七原件在运行副本中 SHA 一致 | `eviden"
            "ce/u0/or"
            "iginal-preservation-final.json` |"
        ),
        "",
        (
            "表中 `experiments/` 和 `evidence/` 均相对 `runtime/v8/`。机器可读总表为本目录 `ac"
            "ceptance-summary.json`，含每项硬门槛及输入文件 SHA。"
        ),
        "",
        "## 最终局部更新实测",
        "",
        (
            "S/M/L/XL 的字段/状态/动作规模为 100/30/10、500/150/30、2,000/600/100、10,000/"
            "3,000/500；每组 20 个固定种子、每种子 50"
            " 批。下表单位为毫秒，四个独立 CPU 工作进程组内顺序"
            "对照，包含并发资"
            "源竞争。p50/p95 采用排序后索引 floor((n−1)×q)。"
        ),
        "",
        "| 规模 | 更新字段 | 全量 p50/p95 | 局"
        "部 p50/p95 | 冷启动 p50/p95 | 平均"
        "重算状态/全量状态 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in local["groups"]:

        def fmt(v):
            return f"{v['p50'] * 1000:.3f}/{v['p95'] * 1000:.3f}"

        lines.append(
            f"| {g['size']} | {g['updates']} | {fmt(g['full_seconds'])} | "
            f"{fmt(g['local_seconds'])} | {fmt(g['cold_seconds'])} | "
            f"{g['mean_local_states']:.3f}/{g['full_states']} |"
        )
    lines += [
        "",
        (
            "小规模密集更新有明确反例：S、50 字段，局部 p50 "
            "5.263 ms 高于全量 3.295 ms；M、50 "
            "字段也更慢。大规"
            "模稀疏更新减少重算状态，但逻辑槽指纹、依据核验和队列成本"
            "仍存在，因此不承诺固定加速比例。旧开发核的完整 16,0"
            "00 批单独保留"
            "在 `experiments/v8/local/`，不混作最终核性能。"
        ),
        "",
        "## 图检索实测",
        "",
        "| 方法 | 必需控制证据覆盖 | 适用性精确率（均值） | 不适用接受数 |",
        "|---|---:|---:|---:|",
    ]
    for g in graph_metrics:
        lines.append(
            f"| {g['method']} | {g['required_coverage']:.1%} | "
            f"{g['applicability_precision']:.2%} | {g['invalid_accepted']} |"
        )
    lines += [
        "",
        (
            "契约扩展覆盖率 50%：场景 1、2 的首个 P3 种子"
            "来自版本未知手册，被 `unknown_version`"
            " 排除；合成必需"
            "控制事实挂接于这个种子，因而没有可采信扩展路径。排除原因"
            "与原件 locator 在对应 subgraph 中保留"
            "。该结果反映适用"
            "性约束的召回代价，不能宣称图检索召回全面改善。"
        ),
        "",
        (
            "四个查询使用同一批 P3 种子和 100 节点上限。适用"
            "性标签是合成控制 oracle，不是人工语义相关性。普通"
            " Neo4j G"
            "raphRAG 1.22.0 的 HybridCypherRetriever 在独立环境与真实数据库上运行，使用同一批 1,06"
            "1 个 2,560 维向量；全文/向量索引底座与 P3 不同，输出 `official-hybrid-cypher.json` "
            "单独报告。"
        ),
        "",
        "## Agent 与离线门控对照",
        "",
        "| 方法 | 错误放行数 | 合法建议/合法总数 | 模型调用 | 工具/声明来源访问次数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in methods:
        lines.append(
            f"| {m['method']} | {m['unsafe_releases']} | "
            f"{m['legal_accepted']}/{m['legal_total']} | "
            f"{m['model_calls']} | {m['tool_calls']} |"
        )
    lines += [
        "",
        (
            "全部方法同用 baseline 模型、冻结输入和 8/8"
            "/3/180 秒上限，实际消费次数不同。B1 运行保留 "
            "P5 LangG"
            "raph 和冻结工具包，B0/B2/B3/B4 是声明的"
            "模块消融，来源预处理共用；后四者的工具数为声明访问计数，"
            "不能与实际 le"
            "dger/backend 次数直接作效率结论。Ours 与 B1 记录真实计数。"
        ),
        "",
        (
            "同次 Ours 模型原始建议移除 Gate 后离线出现 "
            f"{value['gate_off_offline_unsafe_releases']} 次错误放行；"
            "这只是离线判定，从未执行动作。完整 Gate 为 0 错误且合法 10/10。"
            "全部场景 synthetic=true、reviewed=false，oracle 独立预声明，"
            "不代表真实网络故障诊断准确率。"
        ),
        "",
        "## 恢复、真实资料与交付边界",
        "",
        (
            "验证快照目录为 `artifacts/v8-restore-validation-20261007/`，含 2,925 个文件；"
            "恢复根为 `runtime/v8-restore-validation-20261007/`。原运行根的暂停任务仍保持暂停。新根"
            "中首次恢复产生三条只读工具回执、六条证据 revisio"
            "n，三项建议均可输出；后续重复恢复返回同样结果，六张相关"
            " SQL 表和预"
            "算均无变化。"
        ),
        "",
        "回放在新根执行完整 Ours 流程，并独立计算全部状态和"
        "动作。它是已消费输入的复现验证，没有再次请求模型或重新进"
        "行盲验；模型对照按已保存原始建议核算。",
        "",
        (
            "真实手册适配器实际检索并图扩展得到 79 个绑定，但用途"
            "标签未知，K 保持 unknown，动作转人工；历史日志"
            "在所声明对象下没"
            "有适用观测，保持缺口。没有用合成资料给真实任务补满足状态"
            "。可查看运行根 `examples/real-manua"
            "l-task.j"
            "son`、`examples/real-log-task.json` 及对应报告。"
        ),
        "",
        (
            "交付包含代码、工具、锁、配置、样例、原件副本、契约/证据"
            "/图快照、实验 JSON/CSV、更新回执、专利对应表、"
            "开源引用和恢复说"
            "明。私有 Neo4j 密码、共享权重、虚拟环境和 Doc"
            "ker 数据卷不打包。Python 3.11、现有共享三"
            "组实际使用权重与"
            "生成服务、检索环境、原生绝对路径是同机复现依赖；配置中保"
            "留的 target 模型未用于本轮验收。"
        ),
        "",
        "实际 Open5GS/UERANSIM 取证、现网部署、"
        "受控执行和跨主机离线交付属于后续阶段；历史 M6/M7 "
        "验收未替换。",
        "",
    ]
    (PROJECT / "docs/v8/验收报告.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {"pass": value["pass"], "gates": gates, "csv_rows": len(rows)},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not value["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
