import json
import os
import threading
import time
from typing import Any

from src.callbacks.callback import Callback, CallbackArguments


class ProcessMemoryMonitorCallback(Callback):
    def __init__(
        self,
        sample_interval_seconds: float = 0.5,
        include_child_processes: bool = False,
    ) -> None:
        super().__init__()
        assert sample_interval_seconds > 0
        self.sample_interval_seconds = sample_interval_seconds
        self.include_child_processes = include_child_processes
        self.pid = os.getpid()
        self.page_size = os.sysconf("SC_PAGE_SIZE")

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_sample_key: str | None = None

        self.sample_count = 0
        self.rss_bytes_sum = 0
        self.rss_bytes_peak = 0
        self.first_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self.sample_record_dict: dict[str, dict[str, Any]] = {}

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def _get_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "memory_usage.json")

    def restore_state(self) -> None:
        state_path = self._get_state_path()
        if not os.path.exists(state_path):
            return
        with open(state_path, "r", encoding="utf-8") as file:
            state = json.load(file)
        summary = state.get("summary", {})
        self.sample_count = int(summary.get("sample_count", 0))
        self.rss_bytes_sum = int(summary.get("rss_bytes_sum", 0))
        self.rss_bytes_peak = int(summary.get("peak_rss_bytes", 0))
        self.first_timestamp = summary.get("first_timestamp")
        self.last_timestamp = summary.get("last_timestamp")
        self.sample_record_dict = state.get("per_sample", {})

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        self._current_sample_key = str(callback_args.current_session.sample_index)
        self._start_thread_if_needed()
        self._record_sample()

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        self._record_sample()

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        self._record_sample()

    def on_task_interact(self, callback_args: CallbackArguments) -> None:
        self._record_sample()

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        self._record_sample()

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        self._record_sample()
        self._dump_state()
        self._current_sample_key = None

    def _start_thread_if_needed(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._record_sample()

    def _record_sample(self) -> None:
        timestamp = time.time()
        rss_bytes = self._get_total_rss_bytes()
        with self._lock:
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.last_timestamp = timestamp
            self.sample_count += 1
            self.rss_bytes_sum += rss_bytes
            self.rss_bytes_peak = max(self.rss_bytes_peak, rss_bytes)

            if self._current_sample_key is not None:
                record = self.sample_record_dict.setdefault(
                    self._current_sample_key,
                    {
                        "sample_count": 0,
                        "rss_bytes_sum": 0,
                        "peak_rss_bytes": 0,
                        "first_timestamp": timestamp,
                        "last_timestamp": timestamp,
                    },
                )
                record["sample_count"] += 1
                record["rss_bytes_sum"] += rss_bytes
                record["peak_rss_bytes"] = max(record["peak_rss_bytes"], rss_bytes)
                record["last_timestamp"] = timestamp

    def _get_total_rss_bytes(self) -> int:
        pid_list = [self.pid]
        if self.include_child_processes:
            pid_list.extend(self._get_descendant_pid_list(self.pid))
        return sum(self._get_rss_bytes(pid) for pid in pid_list)

    def _get_rss_bytes(self, pid: int) -> int:
        try:
            with open(f"/proc/{pid}/statm", "r", encoding="utf-8") as file:
                fields = file.read().split()
            return int(fields[1]) * self.page_size
        except Exception:
            return 0

    def _get_descendant_pid_list(self, root_pid: int) -> list[int]:
        parent_to_child_dict: dict[int, list[int]] = {}
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as file:
                    stat_text = file.read()
                right_paren_index = stat_text.rfind(")")
                fields = stat_text[right_paren_index + 2 :].split()
                parent_pid = int(fields[1])
            except Exception:
                continue
            parent_to_child_dict.setdefault(parent_pid, []).append(pid)

        descendant_pid_list: list[int] = []
        stack = list(parent_to_child_dict.get(root_pid, []))
        while stack:
            pid = stack.pop()
            descendant_pid_list.append(pid)
            stack.extend(parent_to_child_dict.get(pid, []))
        return descendant_pid_list

    def _dump_state(self) -> None:
        with self._lock:
            summary = {
                "pid": self.pid,
                "include_child_processes": self.include_child_processes,
                "sample_interval_seconds": self.sample_interval_seconds,
                "sample_count": self.sample_count,
                "rss_bytes_sum": self.rss_bytes_sum,
                "average_rss_bytes": (
                    self.rss_bytes_sum / self.sample_count
                    if self.sample_count > 0
                    else 0
                ),
                "peak_rss_bytes": self.rss_bytes_peak,
                "average_rss_mib": (
                    self.rss_bytes_sum / self.sample_count / 1024 / 1024
                    if self.sample_count > 0
                    else 0
                ),
                "peak_rss_mib": self.rss_bytes_peak / 1024 / 1024,
                "first_timestamp": self.first_timestamp,
                "last_timestamp": self.last_timestamp,
            }
            per_sample = {
                sample_key: {
                    **record,
                    "average_rss_bytes": (
                        record["rss_bytes_sum"] / record["sample_count"]
                        if record["sample_count"] > 0
                        else 0
                    ),
                    "average_rss_mib": (
                        record["rss_bytes_sum"]
                        / record["sample_count"]
                        / 1024
                        / 1024
                        if record["sample_count"] > 0
                        else 0
                    ),
                    "peak_rss_mib": record["peak_rss_bytes"] / 1024 / 1024,
                }
                for sample_key, record in self.sample_record_dict.items()
            }
        os.makedirs(self.get_state_dir(), exist_ok=True)
        with open(self._get_state_path(), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "summary": summary,
                    "per_sample": per_sample,
                },
                file,
                indent=2,
            )
