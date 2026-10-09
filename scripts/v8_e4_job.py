"""Durable foreground worker for a detached E4 full matrix, with explicit completion status."""

import argparse
import hashlib
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from upf_agent.v8.e4 import run_matrix, write_json


def performance_report(output, summary):
    lines = [
        "# E4 全量四模式性能结果",
        "",
        f"方法批次：{summary['method_batches']}；一致性通过：{summary['pass']}；"
        f"worker：{summary['workers']}；计时用途：{summary['measurement']}。",
        "",
        "每批方法顺序随机，使用相同冻结输入；序列化、比较和持久化不计入更新延迟。"
        "初始化成本与已初始化更新成本分开；没有清除 OS 缓存，也没有独占共享机器。",
        "",
        "RSS 是进程累计峰值，不能归因到单一方法；fallback 耗时是重算耗时子集，不能相加。"
        "Full 和 Local 共用确定性规则，Full 不调用 DependencyIndex。合成结果不代表现网收益。",
        "",
        "| 规模/变更 | 方法 | 批次 | 更新 P50/ms | P95/ms | 初始化 P50/ms | "
        "Full/方法总耗时 | CPU/s | fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    negative = []
    for name, group in summary["groups"].items():
        for method, result in group["methods"].items():
            ratio = result["paired_total_full_over_method"]
            lines.append(
                f"| {name} | {method} | {result['batches']} | "
                f"{result['seconds']['p50'] * 1000:.2f} | {result['seconds']['p95'] * 1000:.2f} | "
                f"{result['cold_seconds']['p50'] * 1000:.2f} | {ratio:.3f} | "
                f"{result['cpu_seconds']:.2f} | {result['fallbacks']} |"
            )
            if method != "L0-Full" and ratio < 1:
                negative.append(f"{name}/{method}={ratio:.3f}")
    lines += [
        "",
        "比值 <1 的实际负收益组：" + ("；".join(negative) or "本批没有") + "。",
        "",
        "详细逐批记录、变更序列、构建分项及审计/重算成本见各组 CSV、changes.jsonl "
        "和 summary.json。不同规模不是独立故障；比值为配对总耗时比，不是统计显著性。",
    ]
    (output / "e4-performance.md").write_text("\n".join(lines) + "\n")


def run(args):
    if args.status_file.exists():
        raise ValueError("job status already exists; choose a new job ID")
    now = lambda: datetime.now(UTC).isoformat()  # noqa: E731
    status = {
        "schema": "v8-e4-job-1",
        "status": "running",
        "started_at_utc": now(),
        "output": str(args.output.resolve()),
        "code_root": str(args.project.resolve()),
        "expected_method_batches": 64000,
        "workers": args.workers,
    }
    write_json(args.status_file, status)
    try:
        result = run_matrix(args.project, args.output, "full", args.workers, args.correctness)
        performance_report(args.output, result)
        status.update(
            status="completed" if result["pass"] else "failed",
            ended_at_utc=now(),
            completed_method_batches=result["method_batches"],
            summary_sha256=hashlib.sha256((args.output / "summary.json").read_bytes()).hexdigest(),
            source_sha256=result["source"]["sha256"],
        )
        status["pass"] = result["pass"]
        # Only mark the concrete expansion step, after the same-source stress run also passed.
        # Never mark WP07/phase3 complete automatically or overwrite a changed checklist entry.
        if args.plan and args.stress and result["pass"]:
            stress = json.loads(args.stress.read_text())
            if stress.get("pass") and stress.get("source") == result["source"]:
                pending = (
                    "  - [ ] 正确性通过后扩至每方法 16,000 批，共 64,000 方法批次；"
                    "增加稠密及高扇出变更。"
                )
                body = args.plan.read_text()
                if body.count(pending) == 1:
                    completed = pending.replace("[ ]", "[x]", 1) + (
                        f"（自动完成记录：{now()}，`{args.output.name}`；"
                        "16,000/方法、64,000 总批次完整一致，另批稠密/高扇出通过。）"
                    )
                    # No user-facing state is touched before a successful summary exists.
                    args.plan.write_text(
                        body.replace(pending, completed, 1)
                        + (
                            f"\n### E4 全量任务自动完成：{now()}\n\n"
                            f"- `{args.output.name}`：64,000 方法批次完成并一致；结果、成本、"
                            "负收益保存在该运行目录的 `summary.json`、`e4-performance.md`。"
                            "WP07 总项仍等待性能结果复核；WP08/WP09 和人工 Gold 状态不变。\n"
                        )
                    )
                    status["plan_expansion_step_recorded"] = True
        write_json(args.status_file, status)
        return 0 if result["pass"] else 1
    except BaseException:
        status.update(status="failed", ended_at_utc=now(), traceback=traceback.format_exc())
        write_json(args.status_file, status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 4], default=1)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--stress", type=Path)
    raise SystemExit(run(parser.parse_args()))
