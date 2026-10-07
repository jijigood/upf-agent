"""Build source-backed candidate labels; never assert that assistant labels are human gold."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from upf_agent.log_tools import LogStore
from upf_agent.p3_answer import source_lookup
from upf_agent.p3_index import load_active_index, load_chunks
from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    destination = ROOT / "eval/p6"
    destination.mkdir(parents=True, exist_ok=True)
    suite = destination / "candidates.jsonl"
    if suite.exists():
        raise ValueError(
            "Candidate suite already exists; preserve labels and generate a new version"
        )
    directory, manifest = load_active_index(ROOT)
    chunks = load_chunks(directory)
    probes = [
        json.loads(line) for line in (ROOT / "eval/p3_dev_probes.jsonl").read_text().splitlines()
    ]
    exposed = {chunks[key].parent_id for probe in probes for key in probe["relevance"]}
    cases = []
    used = set()

    def manual(category, chunk, questions, split, group):
        siblings = [c for c in chunks.values() if c.parent_id == chunk.parent_id]
        for question in questions:
            cases.append(
                EvalCase(
                    id=f"{category}-{sum(c.category == category for c in cases) + 1:03d}",
                    category=category,
                    split=split,
                    question=question,
                    source_groups=[chunk.parent_id, group],
                    product=chunk.product,
                    version=chunk.version if chunk.version != "unknown" else None,
                    index_generation=manifest["index_generation"],
                    evidence_ids=[c.chunk_id for c in siblings],
                    expected_extracts=[chunk.text],
                    forbidden_claims=["执行了设备命令", "示例地址就是当前现场地址"],
                )
            )
        used.add(chunk.parent_id)

    families = [
        "show",
        "upfmgr",
        "set-upf-prt-sw",
        "echo",
        "tcpdump",
        "systemctl",
        "helm",
        "ls",
        "chmod",
        "journalctl",
    ]
    for index, family in enumerate(families):
        split = "dev" if index < 5 else "heldout"
        matches = []
        for chunk in chunks.values():
            syntax = chunk.text.splitlines()[1] if len(chunk.text.splitlines()) > 1 else ""
            syntax = re.sub(r"^(?:UPFD?>|UPF#)\s*", "", syntax)
            if chunk.kind == "command" and syntax.split() and syntax.split()[0] == family:
                if split == "heldout" and chunk.parent_id in exposed:
                    continue
                matches.append((chunk, syntax))
        if not matches:
            raise ValueError(f"No unexposed command source for {family}")
        chunk, syntax = matches[0]
        context = " / ".join(chunk.section_path[-2:])
        manual(
            "command",
            chunk,
            [
                f"在‘{context}’中，{family} 命令的原文写法是什么？",
                f"请引用‘{context}’中的命令 {syntax}，保留占位符和参数。",
                f"手册‘{context}’如何使用 {family}？只摘录原文并说明示例适用范围。",
                f"核对 {syntax} 在‘{context}’中的原始语法，不把示例当现场参数。",
            ],
            split,
            f"command-family:{family}",
        )

    for category in ("knowledge", "troubleshooting"):
        selected = []
        for chunk in chunks.values():
            if chunk.kind != "prose" or chunk.parent_id in used or len(chunk.text) < 160:
                continue
            topic = " / ".join(chunk.section_path)
            suitable = (
                ("说明：" in chunk.text or "功能描述" in chunk.text)
                if category == "knowledge"
                else bool(re.search(r"解决方法|排查|故障", topic))
            )
            if suitable:
                selected.append(chunk)
                used.add(chunk.parent_id)
        # Prior development target groups remain development; heldout uses unexposed sections.
        dev = selected[:5]
        heldout = [c for c in selected[5:] if c.parent_id not in exposed][:5]
        if len(dev) != 5 or len(heldout) != 5:
            raise ValueError(f"Not enough disjoint source sections for {category}")
        for index, chunk in enumerate(dev + heldout):
            context = " / ".join(chunk.section_path[-3:])
            wording = "说明和关键字段" if category == "knowledge" else "排查步骤和操作前提"
            manual(
                category,
                chunk,
                [
                    f"手册中‘{context}’的{wording}是什么？请给出原文依据。",
                    f"请依据‘{context}’解释{wording}，不要推断真实设备当前状态。",
                ],
                "dev" if index < 5 else "heldout",
                f"manual-topic:{chunk.parent_id}",
            )

    log_references = {}
    event_types = [
        "pfcp_heartbeat_request",
        "pfcp_node_unreachable",
        "far_handover",
        "gtpu_echo_retry",
        "license_not_installed",
        "pfcp_heartbeat_response",
        "pfcp_session_deleted",
        "interface_index_failure",
        "n4_name_failure",
        "gtpu_unreachable_detected",
    ]
    for index, event in enumerate(event_types):
        reference = LogStore(ROOT / "data/p4/logs", "a6f18ac81a18aa2125fa").find(
            event_type=event, limit=3
        )
        for wording in [
            f"查询 {event} 的代表日志，保留发生/解除/正常状态。",
            f"从日志原文核对 {event}，说明能否据此确定独立故障和根因。",
        ]:
            cases.append(
                EvalCase(
                    id=f"log-{sum(c.category == 'log' for c in cases) + 1:03d}",
                    category="log",
                    split="dev" if index < 5 else "heldout",
                    question=wording,
                    source_groups=[f"log-template:{event}"],
                    index_generation=manifest["index_generation"],
                    log_generation="a6f18ac81a18aa2125fa",
                    required_event_types=[event],
                    evidence_ids=[item["event_id"] for item in reference.data["events"]],
                    expected_extracts=[
                        "消息数不等于独立故障数；源时区和真实身份未知，不能推断根因。"
                    ],
                    forbidden_claims=["已确定根因", "根据脱敏地址确定真实设备"],
                    expected_statuses=["partial", "need_input"],
                )
            )

            log_references[cases[-1].id] = reference.model_dump(mode="json")
    write_json(destination / "source_references.json", log_references)

    for scenario in range(5):
        fixture = destination / f"topology_synthetic_{scenario}.csv"
        with fixture.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "device_id",
                    "device_port",
                    "peer_device_id",
                    "peer_port",
                    "interface_type",
                    "valid_from",
                    "valid_to",
                ]
            )
            writer.writerow(
                [
                    f"SYN-UPF-{scenario}",
                    "eth0",
                    f"SYN-SW-{scenario}",
                    "port1",
                    "N3",
                    "2025-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ]
            )
        for operation in ["查询 N3 邻接设备和端口", "检查过期关系与 N30 是否能混作 N3"]:
            cases.append(
                EvalCase(
                    id=f"topology-{sum(c.category == 'topology' for c in cases) + 1:03d}",
                    category="topology",
                    split="dev" if scenario < 3 else "heldout",
                    synthetic=True,
                    question=f"合成拓扑样例 {scenario}：{operation}；不能当作真实现场。",
                    source_groups=[f"synthetic-topology:{scenario}"],
                    index_generation=manifest["index_generation"],
                    evidence_ids=[f"{fixture.relative_to(ROOT)}:row2"],
                    expected_extracts=[
                        f"SYN-UPF-{scenario}:eth0 -> SYN-SW-{scenario}:port1; N3; synthetic"
                    ],
                    forbidden_claims=["已确认现场拓扑"],
                    expected_statuses=["completed", "partial", "missing_source"],
                )
            )

    missing = [
        ("live-cpu", ["当前 UPF 的 CPU 温度是多少？", "现在 UPF 的 CPU 负载是多少？"], "dev"),
        ("version-9", ["UPF 9.9 版本如何配置 N3 接口？", "UPF 9.9 的 CLI 配置指令是什么？"], "dev"),
        (
            "no-log",
            ["没有日志文件，请诊断昨天的 UPF 中断。", "未提供日志，能确认 UPF 昨日故障根因吗？"],
            "heldout",
        ),
        (
            "no-topology",
            [
                "没有实际拓扑，请确定现场 N3 交换机端口。",
                "未提供现场连接表，UPF 连到哪台实际交换机？",
            ],
            "heldout",
        ),
        (
            "no-identity",
            [
                "仅有脱敏日志，实际发生告警的是哪台真实设备？",
                "请把掩码地址还原成实际 UPF 的设备身份。",
            ],
            "heldout",
        ),
    ]
    for group, questions, split in missing:
        for question in questions:
            cases.append(
                EvalCase(
                    id=f"missing-{sum(c.category == 'missing' for c in cases) + 1:03d}",
                    category="missing",
                    split=split,
                    question=question,
                    source_groups=[f"missing:{group}"],
                    index_generation=manifest["index_generation"],
                    product="UPF",
                    version="9.9" if group == "version-9" else None,
                    expected_statuses=["missing_source", "need_input", "partial"],
                    expected_extracts=["应说明缺资料，不能模拟实时网管或猜测版本命令。"],
                    forbidden_claims=["已接入实时网管", "已还原真实设备"],
                )
            )
    validate_partition(cases)
    assert Counter(c.split for c in cases) == {"dev": 60, "heldout": 60}
    suite.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    review = [
        "# P6 业务真值复核单",
        "",
        "120 题候选，开发/保留各 60；全部为助手草案，不是已审核业务真值。",
        "原文抽取通过不能代替题目的相关性、完整性和现场适用性。",
        "逐题勾选并写复核人/日期；必要时修改问题、证据集合与预期事实。近义问法归同一来源组，禁止跨组泄漏。",
        "保留集在配置冻结前不执行推理；下面仅显示资料与拟定真值，便于人工复核。",
        "",
    ]
    for case in cases:
        review += [
            f"## {case.id} · {case.split} · {case.category}",
            "",
            "- [ ] 问题及预期事实/相关证据已核对",
            f"- 问题：{case.question}",
            "- 结论：待复核",
            "- 复核人：",
            "- 日期：",
            "- 修正意见：",
            f"- 分组：{', '.join(case.source_groups)}",
            f"- synthetic：{case.synthetic}",
            "- 禁止结论：" + "；".join(case.forbidden_claims),
        ]
        for eid in case.evidence_ids:
            if eid.startswith("chunk-"):
                source = source_lookup(ROOT, case.index_generation, eid)
                review += [
                    f"- 来源：{source['relative_path']} · {', '.join(source['locators'])}"
                    f" · `{eid}`",
                    "```text",
                    source["text"],
                    "```",
                ]
            else:
                review += [f"- 来源：{eid}"]
        review += ["- 预期原文/缺口：", "```text", *case.expected_extracts, "```", ""]
    (destination / "truth_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    write_json(
        destination / "manifest.json",
        {
            "suite_sha256": sha(suite),
            "counts": dict(Counter(c.category for c in cases)),
            "splits": dict(Counter(c.split for c in cases)),
            "human_reviewed": 0,
            "topology_source_sha256": {
                p.name: sha(p) for p in destination.glob("topology_synthetic_*.csv")
            },
            "index_generation": manifest["index_generation"],
            "heldout_executed": False,
            "prior_p3_target_groups_excluded_from_heldout": True,
            "formal_business_accuracy_claim": False,
        },
    )
    print(
        json.dumps(
            {
                "suite": str(suite),
                "review": str(destination / "truth_review.md"),
                "cases": len(cases),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
