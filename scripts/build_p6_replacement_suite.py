"""Draft a source-group-disjoint replacement heldout; never certify its truth."""

import csv
from collections import Counter
from pathlib import Path

from upf_agent.log_tools import LogStore
from upf_agent.p3_index import load_active_index, load_chunks
from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "eval/p6/business_v4/reviewed_suite.jsonl"
DEST = ROOT / "eval/p6/business_v5_replacement"


def main() -> None:
    if DEST.exists():
        raise ValueError("Replacement draft already exists; preserve it and create a new version")
    old = [EvalCase.model_validate_json(line) for line in OLD.read_text().splitlines()]
    old_groups = {group for case in old for group in case.source_groups}
    directory, manifest = load_active_index(ROOT)
    chunks = load_chunks(directory)
    cases = []

    def add(category: str, question: str, groups: list[str], **kwargs: object) -> None:
        overlap = set(groups) & old_groups
        if overlap:
            raise ValueError(f"Replacement heldout reuses old source groups: {overlap}")
        cases.append(
            EvalCase(
                id=f"v5-{category}-{sum(c.category == category for c in cases) + 1:03d}",
                category=category,
                split="heldout",
                question=question,
                source_groups=groups,
                index_generation=manifest["index_generation"],
                **kwargs,
            )
        )

    command_families = ["vim", "curl", "kubectl", "set-redis-start", "ping"]
    for family in command_families:
        matches = [
            chunk
            for chunk in chunks.values()
            if chunk.kind == "command"
            and chunk.parent_id not in old_groups
            and len(chunk.text.splitlines()) > 1
            and chunk.text.splitlines()[1].lstrip().split()[0] == family
        ]
        if not matches:
            raise ValueError(f"No new command source for {family}")
        chunk = matches[0]
        context = " / ".join(chunk.section_path[-2:])
        syntax = chunk.text.splitlines()[1]
        questions = [
            f"手册‘{context}’中 {family} 的命令原文是什么？",
            f"请逐字引用‘{context}’里的 {syntax}，保留参数和占位符。",
            f"‘{context}’的 {family} 示例应如何理解适用范围？给出原文。",
            f"核对‘{context}’的 {family} 语法；不要把示例值写成现场值。",
        ]
        for question in questions:
            add(
                "command",
                question,
                [chunk.parent_id, f"command-family:{family}"],
                product=chunk.product,
                version=chunk.version if chunk.version != "unknown" else None,
                evidence_ids=[chunk.chunk_id],
                expected_extracts=[chunk.text],
                forbidden_claims=["已在现场执行命令", "手册示例值就是现场参数"],
            )

    knowledge_parents = [
        "section-4e604a1760d0fdd7acce",
        "section-c6eceb0a5b5d9976db2f",
        "section-43f41ac3d75696c571ca",
        "section-2039c44d4c3fdff32e68",
        "section-ae50873a1889a599e9cb",
    ]
    for parent in knowledge_parents:
        source = next(c for c in chunks.values() if c.parent_id == parent and c.kind == "prose")
        context = " / ".join(source.section_path[-2:])
        for question in (
            f"手册‘{context}’的功能和关键字段是什么？请给原文依据。",
            f"依据‘{context}’说明操作范围及已有字段；空白信息要说明缺口。",
        ):
            add(
                "knowledge",
                question,
                [parent, f"v5-knowledge-topic:{parent}"],
                product=source.product,
                version=source.version if source.version != "unknown" else None,
                evidence_ids=[source.chunk_id],
                expected_extracts=[source.text],
                forbidden_claims=["空白字段已有手册定义", "已核实现场配置"],
            )

    fault_titles = [
        "k8s 异常：大部分pod处于Evicted状态",
        "N3N6接口地址未配置进data plane",
        "刚部署完后curl配置失败: Unexecuted MML commands",
        "UPF N4 IPv6 接口地址到网关链路异常",
        "UPF 到基站链路异常",
    ]
    for title in fault_titles:
        related = [c for c in chunks.values() if title in c.section_path]
        if not related:
            raise ValueError(f"Missing historical case: {title}")
        groups = sorted({c.parent_id for c in related}) + [f"v5-fault-topic:{title}"]
        evidence = [
            next((c.chunk_id for c in related if label in c.section_path), related[0].chunk_id)
            for label in ("症状现象", "信息收集", "解决方法")
        ]
        evidence = list(dict.fromkeys(evidence))
        for question in (
            f"手册历史案例‘{title}’记录了什么症状、信息收集及处理方法？",
            f"引用‘{title}’的排查依据与处理前提，指出它不能直接证明当前故障。",
        ):
            add(
                "troubleshooting",
                question,
                groups,
                product="UPF",
                evidence_ids=evidence,
                expected_extracts=[
                    "逐项核对原件中的症状、信息收集、解决方法；历史案例不是当前故障。"
                ],
                forbidden_claims=["历史案例已证明当前故障根因"],
                expected_statuses=["partial"],
            )

    log_store = LogStore(ROOT / "data/p4/logs", "a6f18ac81a18aa2125fa")
    for event, count in (
        ("upfd_start", 4),
        ("timer_init_failure", 3),
        ("gtpu_unreachable_report", 3),
    ):
        sample = log_store.find(event_type=event, limit=3).data["events"]
        if not sample:
            raise ValueError(f"No real log examples for {event}")
        questions = [
            f"查询 {event} 的代表日志原文、状态和行号。",
            f"从 {event} 的真实日志核对时间和设备身份有哪些缺口。",
            f"{event} 的消息数能否直接算作独立故障数和根因？给出代表日志。",
            f"仅凭 {event} 的脱敏记录，能否确定现场设备和故障结论？",
        ]
        for question in questions[:count]:
            add(
                "log",
                question,
                [f"log-template:{event}"],
                log_generation="a6f18ac81a18aa2125fa",
                required_event_types=[event],
                evidence_ids=[row["event_id"] for row in sample],
                expected_extracts=[
                    "原始日志行可核对；源时区和真实身份未确认，不能仅凭计数推断根因。"
                ],
                forbidden_claims=["已确定真实设备", "已确定独立故障根因"],
                expected_statuses=["partial", "need_input"],
            )

    DEST.mkdir(parents=True)
    for number in range(10, 15):
        fixture = DEST / f"topology_synthetic_{number}.csv"
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
                    f"SYN-UPF-{number}",
                    f"eth{number}",
                    f"SYN-SW-{number}",
                    f"port{number}",
                    "N3",
                    "2025-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ]
            )
        add(
            "topology",
            f"合成拓扑样例 {number}：查询 N3 邻接设备和端口，说明这不代表真实现场。",
            [f"synthetic-topology:{number}"],
            synthetic=True,
            evidence_ids=[f"{fixture.relative_to(ROOT)}:row2"],
            expected_extracts=[
                f"SYN-UPF-{number}:eth{number} -> SYN-SW-{number}:port{number}; N3; synthetic"
            ],
            forbidden_claims=["已确认真实现场拓扑"],
        )

    gaps = [
        (
            "live-n4-peer",
            "没有现场链路数据，请确定现在 N4 对端 SMF 的真实 IP。",
            "缺现场N4链路与身份资料",
        ),
        (
            "live-n6-counters",
            "没有采集当前网口计数器，请给出此刻 N6 的实际丢包数。",
            "缺现场N6计数器",
        ),
        ("log-timezone", "仅有未标时区的日志时间，请给出事件发生的准确 UTC 时刻。", "缺原日志时区"),
        (
            "config-identity",
            "只有手册中的服务器名示例，请确认实际运行的 UPF 主机名。",
            "缺目标主机身份",
        ),
        ("alarm-live", "未接入实时网管，请确认当前活动告警的具体条目。", "缺实时告警来源"),
    ]
    for key, question, gap in gaps:
        add(
            "missing",
            question,
            [f"missing:v5-{key}"],
            product="UPF",
            expected_extracts=[gap],
            expected_statuses=["missing_source", "need_input", "partial"],
            forbidden_claims=["已查询实时现场", "已确定真实设备或UTC时间"],
        )

    previous_heldout = [c.model_copy(update={"split": "dev"}) for c in old if c.split == "heldout"]
    combined = previous_heldout + cases
    validate_partition(combined)
    assert Counter(c.category for c in cases) == {
        "command": 20,
        "knowledge": 10,
        "troubleshooting": 10,
        "log": 10,
        "topology": 5,
        "missing": 5,
    }
    suite = DEST / "candidate_suite.jsonl"
    suite.write_text("\n".join(c.model_dump_json() for c in combined) + "\n", encoding="utf-8")
    review = [
        "# 新独立留出集真值复核草案",
        "",
        "仅需复核下面新出的60题；旧保留60题已转为开发资料。新题尚未由人工核对，不能运行正式留出评测。",
        "命令和参数需逐字核对原手册；日志请核对原日志行；拓扑都是合成数据。",
        "本版真实日志仅有3种未使用模板，外推范围有限。",
        "",
    ]
    raw_log = ROOT / "data/raw/upfd_masked.log"
    for case in cases:
        review += [
            f"## {case.id} · {case.category}",
            "",
            "- [ ] 问题及预期事实/相关证据已核对",
            f"- 问题：{case.question}",
            "- 结论：待复核",
            "- 复核人：",
            "- 日期：",
            "- 修正意见：",
            f"- 来源组：{', '.join(case.source_groups)}",
            f"- 证据ID：{', '.join(case.evidence_ids) or '无'}",
            f"- 预期状态：{', '.join(case.expected_statuses)}",
            f"- 预期原文/缺口：{'；'.join(case.expected_extracts)}",
            "",
        ]
        for eid in case.evidence_ids:
            if eid.startswith("chunk-"):
                source = chunks[eid]
                review += [
                    f"- 原件：`{ROOT / 'data/raw' / source.relative_path}`",
                    f"- 定位：`{', '.join(source.locators)}`；块：`{eid}`",
                    "```text",
                    source.text[:1200],
                    "```",
                    "",
                ]
            elif eid.startswith("log-"):
                sample = log_store.context(eid).data
                location = f"{sample['line_start']}–{sample['line_end']}"
                review += [
                    f"- 原件：`{raw_log}`；行：{location}；事件：`{eid}`",
                    "```text",
                    sample["event"]["raw_excerpt"].rstrip(),
                    "```",
                    "",
                ]
            else:
                review += [
                    f"- 合成拓扑文件：`{ROOT / eid.split(':row')[0]}`；记录：`{eid}`",
                    "",
                ]
    (DEST / "truth_review.md").write_text("\n".join(review), encoding="utf-8")
    write_json(
        DEST / "manifest.json",
        {
            "suite_sha256": sha(suite),
            "old_reviewed_suite_sha256": sha(OLD),
            "old_heldout_reclassified_as_development": 60,
            "new_heldout_draft": 60,
            "new_heldout_human_reviewed": 0,
            "new_heldout_executed": False,
            "old_new_group_overlap": 0,
            "category_counts": dict(Counter(c.category for c in cases)),
            "topology_fixture_sha256": {
                p.name: sha(p) for p in DEST.glob("topology_synthetic_*.csv")
            },
            "formal_M6_pass": False,
        },
    )
    print(f"Created 60 unreviewed, source-group-disjoint candidate cases: {suite}")


if __name__ == "__main__":
    main()
