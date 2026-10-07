"""Prepare broader business candidates without changing the running development controls."""

import json
from collections import Counter
from pathlib import Path

from upf_agent.p3_answer import source_lookup
from upf_agent.p3_index import load_active_index, load_chunks
from upf_agent.p6_contracts import EvalCase, validate_partition
from upf_agent.quality_repair import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "eval/p6/business_v3"


def main():
    if DEST.exists():
        raise ValueError("Preserve the existing business suite; create a new revision")
    directory, manifest = load_active_index(ROOT)
    chunks = load_chunks(directory)
    original = ROOT / "eval/p6/candidates.jsonl"
    cases = [EvalCase.model_validate_json(line) for line in original.read_text().splitlines()]
    # The earlier development answers incidentally published the initial Helm source.
    helm = next(
        c
        for c in chunks.values()
        if c.kind == "command"
        and c.text.splitlines()[1] == "helm -n kube-flannel uninstall flannel"
    )
    siblings = [c for c in chunks.values() if c.parent_id == helm.parent_id]
    context = " / ".join(helm.section_path[-2:])
    questions = [
        f"在‘{context}’中，helm 命令的原文写法是什么？",
        f"请引用‘{context}’中的命令 helm -n kube-flannel uninstall flannel，保留参数。",
        f"手册‘{context}’如何使用 helm？摘录原文并说明命名空间的示例适用范围。",
        f"核对 helm -n kube-flannel uninstall flannel 在‘{context}’中的原始语法。",
    ]
    for number, case in enumerate(c for c in cases if "command-family:helm" in c.source_groups):
        case.question = questions[number]
        case.source_groups = [helm.parent_id, "command-family:helm"]
        case.evidence_ids = [c.chunk_id for c in siblings]
        case.expected_extracts = [helm.text]
    used = {
        group
        for case in cases
        if case.category not in {"knowledge", "troubleshooting"}
        for group in case.source_groups
    }
    exposed = {
        chunks[key].parent_id
        for line in (ROOT / "eval/p3_dev_probes.jsonl").read_text().splitlines()
        for key in json.loads(line)["relevance"]
    }
    prior = json.loads((ROOT / "evidence/p3/quality_regression.json").read_text())
    for row in prior["rows"]:
        exposed.update(chunks[c["chunk_id"]].parent_id for c in row["answer"]["citations"])
    for profile in ("baseline", "target"):
        for path in (ROOT / f"runs/p6/model-{profile}").glob("*.json"):
            data = json.loads(path.read_text())
            if "answer" in data:
                exposed.update(chunks[c["chunk_id"]].parent_id for c in data["answer"]["citations"])
    for case in cases:
        if case.split == "dev":
            exposed.update(group for group in case.source_groups if group.startswith("section-"))

    def choose(prefix, file_hint):
        found = [
            c
            for c in chunks.values()
            if file_hint in c.relative_path and " / ".join(c.section_path).startswith(prefix)
        ]
        if not found:
            raise ValueError(f"No source for {prefix}")
        return sorted(found, key=lambda c: c.ordinal_start)

    selected = []

    def add(category, prefix, file_hint, split, topic):
        source = choose(prefix, file_hint)
        parents = sorted({c.parent_id for c in source})
        if set(parents) & used:
            raise ValueError(f"Source shared with retained command groups: {prefix}")
        if split == "heldout" and set(parents) & exposed:
            raise ValueError(f"Prior development source cannot become heldout: {prefix}")
        group = f"business-topic:{file_hint}:{prefix}"
        for variant in range(2):
            number = sum(c.category == category for c in selected) + 1
            if category == "knowledge":
                question = (
                    f"依据手册原生文字，{topic}有哪些说明与适用范围？"
                    if variant == 0
                    else f"请核对手册对{topic}的描述，哪些内容有文字依据？"
                )
                question += "仅见截图或缺少字段时说明缺口；不要报告当前现场数值。"
            else:
                question = (
                    f"在手册的‘{topic}’案例中，症状、信息收集和处理建议是什么？"
                    if variant == 0
                    else f"请引用手册‘{topic}’的排查依据与处理前提。"
                )
                question += "区分历史案例与当前故障；原件有接口、版本或步骤矛盾时指出。"
            selected.append(
                EvalCase(
                    id=f"{category}-{number:03d}",
                    category=category,
                    split=split,
                    question=question,
                    source_groups=parents + [group],
                    product=source[0].product,
                    version=source[0].version if source[0].version != "unknown" else None,
                    index_generation=manifest["index_generation"],
                    evidence_ids=[c.chunk_id for c in source],
                    expected_extracts=[c.text for c in source],
                    forbidden_claims=[
                        "历史案例原因就是当前根因",
                        "实际执行了手册操作",
                        "截图缺失字段可以自行补造",
                        "已确认现场数值或真实拓扑",
                    ],
                    expected_statuses=["completed", "partial"],
                )
            )
        used.update(parents)

    stats_root = "8 性能管理（Performance Management） / 8.2 Manage / 8.2.3 Manage-UPF"
    knowledge = [
        ("产品配置 / 网管配置管理 / 接口地址配置", "产品使用", "dev", "UPF接口地址配置功能"),
        ("产品配置 / 网管配置管理 / CP IP地址配置", "产品使用", "dev", "CP IP地址配置功能"),
        ("产品监控 / 网管告警监控 / 清除告警", "产品使用", "dev", "UPF网管清除告警功能"),
        (
            "7 故障管理（Fault Management） / 7.1 实时告警 / 7.1.1 告警查询",
            "网管使用",
            "dev",
            "5GC网管实时告警查询",
        ),
        ("产品配置 / 网管配置管理 / UP功能特性配置", "产品使用", "dev", "UP功能特性配置"),
        ("产品配置 / 网管配置管理 / PFCP定时器配置", "产品使用", "heldout", "PFCP定时器配置"),
        ("产品配置 / 网管配置管理 / DNN带宽配置", "产品使用", "heldout", "DNN带宽配置"),
        (
            "产品监控 / 网管性能监控 / 非实时测量任务创建",
            "产品使用",
            "heldout",
            "非实时测量任务创建",
        ),
        (
            " / ".join(
                (
                    stats_root,
                    "8.2.3.5 N4 Node Level Procedures",
                    "8.2.3.5.1 PFCP Assosication Related Statistics",
                )
            ),
            "网管使用",
            "heldout",
            "PFCP偶联相关统计",
        ),
        (
            " / ".join((stats_root, "8.2.3.10 N4 PFCP Heartbeat Statistics")),
            "网管使用",
            "heldout",
            "N4 PFCP心跳统计",
        ),
    ]
    for spec in knowledge:
        add("knowledge", *spec)
    faults = [
        ("UPF N4 IPv6 接口地址到 SMF 链路异常", "dev"),
        ("UPF N6 IPv4 接口地址到网关链路异常", "dev"),
        ("UPF N3 IPv4 接口地址到网关链路异常", "dev"),
        ("UPF（4.0版本）upfd pod一直crash：未安装license", "dev"),
        ("upfmgr n4switch后重启upfd pod一直crash", "dev"),
        ("upf pod都是pending状态", "heldout"),
        ("upfmgr hc执行报错/UPF_OM_Inspect.sh: line 29", "heldout"),
        ("upfmgr hc执行显示N3和N6的v4/v6地址都为空", "heldout"),
        ("UPF（4.0版本）upfd pod一直crash：upfd创建不了ipv6 socket", "heldout"),
        ("UPFD 的N4 接口8805端口未侦听", "heldout"),
    ]
    for topic, split in faults:
        root = (
            "系统平台故障案例"
            if topic.startswith("upf pod") or topic.startswith("upfmgr hc")
            else "业务故障案例"
        )
        add("troubleshooting", f"{root} / {topic}", "紧急故障", split, topic)
    replacements = {c.id: c for c in selected}
    result = [replacements.get(c.id, c) for c in cases]
    validate_partition(result)
    assert Counter(c.split for c in result) == {"dev": 60, "heldout": 60}
    DEST.mkdir(parents=True)
    suite = DEST / "candidates.jsonl"
    suite.write_text("\n".join(c.model_dump_json() for c in result) + "\n")
    review = [
        "# P6 业务真值复核单 v3",
        "",
        "120题助手候选、全部待人工确认。此版本补入UPF真实手册故障案例与网管/协议功能，替换初轮偏安装维护的40题，并替换4道已曝光来源的保留命令题；共44题替换、76题保留。",
        "初轮模型对照属于旧草案的兼容、选择与预算实测；v3完整业务集尚未评测，保留题的历史输出不能当作新版整集成绩。",
        "命令40题仍为10个命令族各4种问法，不是40种独立命令。两次核对关键语法与版本。",
        "开发/保留各60，分组不得跨集；历史已曝光源不能移入新保留集。",
        "日志仅有未知时区与脱敏身份；拓扑仅是合成CSV。原件矛盾、截图缺字段必须标注，不能补猜。",
        "问题、相关块与事实有错误时写修正意见；结论、人名和日期必须真实填写。",
        "",
    ]
    references = json.loads((ROOT / "eval/p6/source_references.json").read_text())
    for case in result:
        review += [
            f"## {case.id} · {case.split} · {case.category}",
            "",
            "- [ ] 问题及预期事实/相关证据已核对",
            f"- 问题：{case.question}",
            "- 结论：待复核",
            "- 复核人：",
            "- 日期：",
            "- 修正意见：",
            "- 分组：" + ", ".join(case.source_groups),
            f"- synthetic：{case.synthetic}",
            "- 禁止结论：" + "；".join(case.forbidden_claims),
        ]
        for eid in case.evidence_ids:
            if eid.startswith("chunk-"):
                original_source = source_lookup(ROOT, case.index_generation, eid)
                review += [
                    f"- 来源：{original_source['relative_path']} · `{eid}`",
                    "- 定位：" + ", ".join(original_source["locators"]),
                    "```text",
                    original_source["text"],
                    "```",
                ]
            else:
                review += [f"- 来源：{eid}"]
        if case.category == "log":
            for event in references[case.id]["data"]["events"]:
                review += [
                    f"- 原始日志：{event['event_id']}",
                    f"- 定位：行{event['line_start']} · 候选状态{event['event_state']}",
                    "```text",
                    event["raw_excerpt"].rstrip(),
                    "```",
                ]
        review += ["- 预期原文/缺口：", "```text", *case.expected_extracts, "```", ""]
    (DEST / "truth_review.md").write_text("\n".join(review) + "\n")
    coverage = {
        category: dict(
            Counter(
                chunks[c.evidence_ids[0]].relative_path for c in result if c.category == category
            )
        )
        for category in ("command", "knowledge", "troubleshooting")
    }
    write_json(
        DEST / "manifest.json",
        {
            "suite_sha256": sha(suite),
            "supersedes_draft_sha256": sha(original),
            "human_reviewed": 0,
            "full_version_business_evaluation_executed": False,
            "replaced_cases": 44,
            "retained_cases": 76,
            "retained_cases_may_have_historical_development_outputs": True,
            "heldout_executed": False,
            "counts": dict(Counter(c.category for c in result)),
            "splits": dict(Counter(c.split for c in result)),
            "primary_manual_coverage": coverage,
            "prior_exposed_source_groups": sorted(exposed),
            "reason": "business representativeness, prepared before any formal evaluation",
        },
    )
    print(json.dumps({"cases": len(result), "coverage": coverage}, ensure_ascii=False))


if __name__ == "__main__":
    main()
