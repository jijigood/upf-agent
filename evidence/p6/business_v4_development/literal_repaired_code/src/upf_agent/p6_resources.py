"""Sample physical GPU usage and explicitly owned Linux process trees."""

import os
import threading
from pathlib import Path

from upf_agent.doctor import gpu_snapshot


def process_tree_rss(roots: list[int]) -> int:
    processes = {}
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = dict(
                line.split(":", 1) for line in path.read_text().splitlines() if ":" in line
            )
            processes[int(path.parent.name)] = (
                int(fields["PPid"]),
                int(fields.get("VmRSS", "0 kB").split()[0]) * 1024,
            )
        except (OSError, ValueError, KeyError):
            continue
    selected = set(roots)
    changed = True
    while changed:
        additions = {pid for pid, (parent, _) in processes.items() if parent in selected}
        changed = bool(additions - selected)
        selected |= additions
    return sum(processes.get(pid, (0, 0))[1] for pid in selected)


class ResourceSampler:
    def __init__(self, server_pid: int | None = None) -> None:
        self.server_pid = server_pid
        self.stop = threading.Event()
        self.peak_gpu_mib = {}
        self.app_peak_rss = 0
        self.server_peak_rss = 0
        self.samples = 0
        self.errors = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            cards, error = gpu_snapshot()
            if error:
                self.errors.append(error)
            for card in cards:
                key = str(card["index"])
                self.peak_gpu_mib[key] = max(self.peak_gpu_mib.get(key, 0), int(card["used_mib"]))
            self.app_peak_rss = max(self.app_peak_rss, process_tree_rss([os.getpid()]))
            if self.server_pid:
                self.server_peak_rss = max(
                    self.server_peak_rss, process_tree_rss([self.server_pid])
                )
            self.samples += 1
            self.stop.wait(0.5)

    def __enter__(self) -> "ResourceSampler":
        self.thread.start()
        return self

    def __exit__(self, *_errors: object) -> None:
        self.stop.set()
        self.thread.join(timeout=5)

    def report(self) -> dict:
        return {
            "sampling_seconds": 0.5,
            "samples": self.samples,
            "gpu_device_peak_used_mib": self.peak_gpu_mib,
            "app_tree_peak_rss_bytes": self.app_peak_rss,
            "server_tree_peak_rss_bytes": self.server_peak_rss if self.server_pid else None,
            "server_pid": self.server_pid,
            "errors": self.errors,
            "scope": "sampled RSS/GPU totals; shared jobs included; peaks may be missed",
        }
