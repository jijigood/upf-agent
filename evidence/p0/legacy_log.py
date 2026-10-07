import re
from datetime import datetime, time
from collections import Counter

# =========================
# 日志分类规则
# =========================
PATTERNS = [
    ("定时器初始化失败", r"upf return error: Invalid upfd public start timer"),
    ("N4链路名称识别失败", r"upf return error: get N4 comm name by cmd ifconfig \| grep -o 'n4-svc1@\*\.\*:' \| awk -F: '\{print \$1\}' failed"),
    ("GTP-U 链路探测请求", r"resend gtpu echo req, peer ip: (?P<peerip>[^,]+), seqnum: (?P<seqnum>[0-9]+), count:(?P<count>[0-9]+)"),
    ("接口索引获取失败（接口状态异常）", r"upf return error: iface index get fail!"),
    ("FAR更新触发 GTP-U 切换", r"#GtpuNode# handover old ip: .+ to new ip: .+\. CAUSE farId:(?P<farId>0x[0-9A-Fa-f]+) update, #session# \{session:(?P<session>0x[0-9A-Fa-f]+) dnn:(?P<dnn>[^,]+),imsi:(?P<imsi>[0-9]+),local seid:(?P<seid>0x[0-9A-Fa-f]+),local teid:(?P<teid>0x[0-9A-Fa-f]+)\}"),
    ("GTP-U 不可达上报", r"report GTPU node unreachable alarm"),
    ("GTP-U 节点不可达", r"alarm gtpu unreachable ipv(?P<ip>[46])!"),
    ("PFCP 会话释放", r"free reason: 'success pfcp delete req' stats: (?P<stats>[0-9]+)"),
    ("PFCP 节点不可达", r"Recv #MsgHeartbeat# Req from cp(.+)"),
    ("PFCP 节点不可达", r"#Alarm# report PFCP node unreachable!"),
    ("PFCP 节点恢复", r"#Alarm# clear PFCP node unreachable\."),
    ("UPF重启/服务启动", r"#init# upfd pod start up!"),
    ("许可证未安装告警", r"Clear license not installed alarm"),
]

DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
    r"(?:\s+(?P<rest>.*))?$"
)

# =========================
# 通用工具
# =========================
def normalize_value(value):
    if value is None:
        return ""

    s = str(value)
    s = s.replace("\\|", "|")
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # 把常见空值统一处理为空串
    if s.lower() in {"nan", "none", "null", ""}:
        return ""

    return s

# =========================
# 日志分类相关
# =========================
def classify_text(text: str) -> str:
    for category, pattern in PATTERNS:
        if re.search(pattern, text):
            return category
    return "Unclassified"

def parse_input_time(value, is_end=False):
    if value is None:
        return None

    value = str(value).strip()
    if value == "":
        return None

    if DATE_ONLY_RE.fullmatch(value):
        d = datetime.strptime(value, "%Y-%m-%d").date()
        if is_end:
            return datetime.combine(d, time(23, 59, 59))
        return datetime.combine(d, time(0, 0, 0))

    candidates = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ]

    normalized = value.replace(",", ".")
    for fmt in candidates:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    raise ValueError("时间格式错误，请使用 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'")

def parse_log_datetime(date_str: str, time_str: str):
    normalized = f"{date_str} {time_str}".replace(",", ".")
    for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None

def parse_logs(file_text: str):
    records = []
    for raw_line in str(file_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = LOG_LINE_RE.match(line)
        if not m:
            continue

        dt = parse_log_datetime(m.group("date"), m.group("time"))
        if dt is None:
            continue

        records.append({
            "datetime": dt,
            "log_content": line
        })
    return records

# =========================
# 拓扑表格解析
# 适配文档提取器把 Excel 转出来的 Markdown 表格
# =========================
def split_markdown_row(line):
    line = str(line).strip()
    if "|" not in line:
        return []

    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]

    return [normalize_value(cell) for cell in line.split("|")]

def is_separator_row(cells):
    if not cells:
        return False

    for cell in cells:
        token = cell.replace(":", "").replace("-", "").replace(" ", "")
        if token != "":
            return False
    return True

def parse_one_topology_text(text):
    if not text:
        return []

    lines = [line for line in str(text).splitlines() if "|" in line]
    if len(lines) < 3:
        return []

    header_idx = None
    header_cells = None

    for i in range(len(lines) - 1):
        cells = split_markdown_row(lines[i])
        next_cells = split_markdown_row(lines[i + 1])

        if len(cells) >= 7 and is_separator_row(next_cells):
            header_idx = i
            header_cells = cells
            break

    if header_idx is None:
        return []

    sw_name = normalize_value(header_cells[1]) if len(header_cells) > 1 else ""

    records = []
    last_upf = ""

    for line in lines[header_idx + 2:]:
        cells = split_markdown_row(line)
        if not cells or is_separator_row(cells):
            continue

        while len(cells) < 8:
            cells.append("")

        row = cells[:8]

        current_upf = normalize_value(row[0])
        if current_upf:
            last_upf = current_upf
        upf = current_upf or last_upf

        sw_port = normalize_value(row[1])
        mode = normalize_value(row[2])
        vlan = normalize_value(row[3])
        seq_no = normalize_value(row[4])
        upf_port = normalize_value(row[5])
        purpose = normalize_value(row[6])
        remark = normalize_value(row[7])

        if not any([upf, sw_port, mode, vlan, seq_no, upf_port, purpose, remark]):
            continue

        row_text = " | ".join([
            upf, sw_port, mode, vlan, seq_no, upf_port, purpose, remark, sw_name
        ]).upper()

        records.append({
            "upf": upf,
            "sw_port": sw_port,
            "mode": mode,
            "vlan": vlan,
            "seq_no": seq_no,
            "upf_port": upf_port,
            "purpose": purpose,
            "remark": remark,
            "sw": sw_name,
            "_row_text": row_text,
        })

    return records

def parse_all_topology_texts(topology_files_text):
    texts = topology_files_text if isinstance(topology_files_text, list) else [topology_files_text]
    all_records = []
    for text in texts:
        all_records.extend(parse_one_topology_text(text))
    return all_records

# =========================
# 拓扑查询
# upf 在 upf 列精确匹配
# N3/N4 在整行文本中包含匹配
# =========================
def search_topology_by_upf_and_keyword(all_rows, upf, keyword):
    upf = normalize_value(upf)
    keyword = normalize_value(keyword).upper()

    matched = []
    for row in all_rows:
        row_upf = normalize_value(row.get("upf", ""))
        row_text = normalize_value(row.get("_row_text", "")).upper()

        if row_upf != upf:
            continue
        if keyword not in row_text:
            continue

        matched.append({
            "upf": row.get("upf", ""),
            "upf_port": row.get("upf_port", ""),
            "sw": row.get("sw", ""),
            "sw_port": row.get("sw_port", ""),
            "用途": row.get("purpose", ""),
        })

    return matched

def dedupe_result(items):
    deduped = []
    seen = set()
    for item in items:
        key = (
            item.get("upf", ""),
            item.get("upf_port", ""),
            item.get("sw", ""),
            item.get("sw_port", ""),
            item.get("用途", ""),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped

# =========================
# 主函数
# =========================
def main(file: str, topology_files_text, upf: str = "", start_time: str = "", end_time: str = "") -> dict:
    # 1) 日志分类统计
    records = parse_logs(file)

    if not records:
        return {
            "category_counts": {},
            "result": []
        }

    records.sort(key=lambda x: x["datetime"])
    log_start = records[0]["datetime"]
    log_end = records[-1]["datetime"]

    start_dt = parse_input_time(start_time, is_end=False) if start_time else log_start
    end_dt = parse_input_time(end_time, is_end=True) if end_time else log_end

    if start_dt > end_dt:
        return {
            "category_counts": {},
            "result": []
        }

    filtered_logs = [
        r["log_content"]
        for r in records
        if start_dt <= r["datetime"] <= end_dt
    ]

    if not filtered_logs:
        return {
            "category_counts": {},
            "result": []
        }

    categories = [classify_text(text) for text in filtered_logs]
    counts = Counter(categories)
    category_counts = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

    # 2) 默认不查拓扑
    result = []
    upf = normalize_value(upf)

    # 3) 如果传了 upf，且分类里含 GTP-U / PFCP，则查拓扑
    if upf:
        category_names = list(category_counts.keys())
        has_gtpu = any("GTP-U" in name for name in category_names)
        has_pfcp = any("PFCP" in name for name in category_names)

        if has_gtpu or has_pfcp:
            all_topology_rows = parse_all_topology_texts(topology_files_text)

            matched = []
            if has_gtpu:
                matched.extend(search_topology_by_upf_and_keyword(all_topology_rows, upf, "N3"))
            if has_pfcp:
                matched.extend(search_topology_by_upf_and_keyword(all_topology_rows, upf, "N4"))

            result = dedupe_result(matched)

    return {
        "category_counts": category_counts,
        "result": result
    }