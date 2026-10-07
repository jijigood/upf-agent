"""Conservative command extraction; unverified OCR is never command authority."""

import hashlib
import re

from upf_agent.contracts import CommandRecord, DocumentBlock, SourceRef

COMMAND_PARSER_VERSION = "commands-v4"

PROMPT = re.compile(r"^(?:\[[^\n]{1,80}[@ ][^\n]{1,80}\][#$]|[\w-]{2,30}[>#])\s*")
COMMAND = re.compile(
    r"^(?:show|set|add|delete|del|display|upfmgr|vppctl|systemctl|journalctl|"
    r"kubectl|docker|helm|ip|ping|traceroute|tcpdump|ethtool|cat|grep|awk|sed|"
    r"tail|head|vim|vi|cd|ls|cp|mv|mkdir|chmod|chown|reboot|grub-mkconfig|"
    r"netplan|ifconfig|route|ss|ps|kill|curl|wget|tar|unzip|date|df|free|top|"
    r"lscpu|modprobe|sysctl|export|source|echo|find|crictl|ctr|nmcli|create|telnet|ssh)\b",
)


def _family(syntax: str) -> str:
    lower = syntax.lower()
    if lower.startswith(("upfmgr", "show", "set", "add", "delete", "display")):
        return "MML" if lower.startswith("upfmgr") else "CLI"
    if lower.startswith("vppctl"):
        return "VPP"
    return "other"


def extract_commands(source: SourceRef, blocks: list[DocumentBlock]) -> list[CommandRecord]:
    result: list[CommandRecord] = []
    candidate_blocks = [b for b in blocks if b.kind in {"paragraph", "code"}]
    by_section: dict[tuple[str, ...], list[DocumentBlock]] = {}
    for block in blocks:
        by_section.setdefault(tuple(block.section_path), []).append(block)
    for block in candidate_blocks:
        if "table_of_contents" in block.quality_flags or "ocr_unverified" in block.quality_flags:
            continue
        raw = block.raw_text.strip()
        if not raw or "\n" in raw or len(raw) > 400:
            continue
        syntax = PROMPT.sub("", raw)
        if re.match(r"^[a-zA-Z][\w-]*:\s", syntax):
            # Shell diagnostics such as cp: ... are output, not executable commands.
            continue
        if not COMMAND.match(syntax):
            continue
        if syntax.startswith("ip ") and not re.match(
            r"ip (?:route|addr|link|rule|netns)\b", syntax
        ):
            continue
        if syntax.startswith(("docker.io/", "docker.com/")):
            continue
        # A sentence that merely mentions a command is not executable syntax.
        syntax_without_parameters = re.sub(r"\{[^{}]+\}|<[^<>]+>|（[^（）]+）", "", syntax)
        if re.search(r"[\u3400-\u9fff]", syntax_without_parameters):
            continue
        near = by_section[tuple(block.section_path)]
        idx = next((i for i, item in enumerate(near) if item.block_id == block.block_id), -1)
        window = near[max(0, idx - 12) : idx + 20] if idx >= 0 else []
        parameters, examples, preconditions = [], [], []
        for item in window:
            if item.block_id == block.block_id:
                continue
            t = item.normalized_text
            if item.kind == "table" and re.search(r"参数|取值|说明|含义|parameter", t, re.I):
                parameters.append(item.block_id)
            elif item.kind == "paragraph" and re.search(
                r"^(?:前提|注意|执行前|条件|\d+[.、．\s]*登录|telnet\b|ssh\b)", t
            ):
                preconditions.append(item.block_id)
        if idx >= 0:
            for offset, item in enumerate(near[idx + 1 : idx + 12], idx + 1):
                if item.kind != "paragraph" or not re.match(
                    r"^(?:举例|示例|例如|例：|example)", item.normalized_text, re.I
                ):
                    continue
                for candidate in near[offset + 1 : offset + 5]:
                    if candidate.kind == "paragraph" and PROMPT.match(candidate.raw_text.strip()):
                        examples.append(candidate.block_id)
                        break
        key = hashlib.sha256(f"{source.source_id}\0{block.locator}\0{raw}".encode()).hexdigest()[
            :16
        ]
        result.append(
            CommandRecord(
                command_id=f"cmd-{key}",
                source_id=source.source_id,
                family=_family(syntax),
                syntax=raw,
                syntax_block_id=block.block_id,
                parameter_tokens=re.findall(r"\{[^{}]+\}|<[^<>]+>|--[\w-]+", syntax),
                parameter_block_ids=parameters,
                example_block_ids=examples,
                precondition_block_ids=preconditions,
                product=source.product,
                version=source.version,
                quality_flags=["candidate_requires_review"],
            )
        )
    return result
