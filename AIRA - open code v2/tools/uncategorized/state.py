"""Shared state management for FreqTrade Manager."""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class ProcessType(str, Enum):
    TRADE = "trade"
    BACKTEST = "backtest"
    HYPEROPT = "hyperopt"
    DOWNLOAD = "download"
    HYPEROPT_SHOW = "hyperopt_show"
    RELOAD = "reload"


class ProcessStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessStats:
    """Real-time process resource stats."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    num_threads: int = 0
    gpu_util: float | None = None        # GPU utilization %
    gpu_mem_mb: float | None = None      # GPU memory used MB
    gpu_temp: int | None = None          # GPU temperature °C
    gpu_fan: int | None = None           # GPU fan speed %
    gpu_name: str = ""                   # GPU model name
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "num_threads": self.num_threads,
        }
        if self.gpu_util is not None:
            d["gpu_util"] = round(self.gpu_util, 1)
            d["gpu_mem_mb"] = round(self.gpu_mem_mb, 1) if self.gpu_mem_mb is not None else None
            d["gpu_temp"] = self.gpu_temp
            d["gpu_fan"] = self.gpu_fan
            d["gpu_name"] = self.gpu_name
        return d


@dataclass
class EpochResult:
    epoch: int
    trades: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_profit: float = 0.0
    profit_total_pct: float = 0.0
    profit_total_abs: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_abs: float = 0.0
    avg_duration: str = ""
    objective: float = 0.0
    is_best: bool = False
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "trades": self.trades,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "avg_profit": self.avg_profit,
            "profit_total_pct": self.profit_total_pct,
            "profit_total_abs": self.profit_total_abs,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_abs": self.max_drawdown_abs,
            "avg_duration": self.avg_duration,
            "objective": self.objective,
            "is_best": self.is_best,
        }


@dataclass
class ProcessInfo:
    process_type: ProcessType
    strategy: str
    status: ProcessStatus = ProcessStatus.IDLE
    pid: int | None = None
    started_at: float = 0.0
    stopped_at: float = 0.0
    timeout_at: float = 0.0  # Timestamp when timeout will trigger (0=no timeout)
    output_lines: list[str] = field(default_factory=list)
    return_code: int | None = None
    error: str = ""
    stats: ProcessStats = field(default_factory=ProcessStats)          # Live stats (0 when dead)
    peak_stats: ProcessStats = field(default_factory=ProcessStats)     # Last non-zero stats
    command: str = ""

    def to_dict(self) -> dict:
        now = time.time()
        if self.status == ProcessStatus.RUNNING and self.started_at > 0:
            runtime = now - self.started_at
        elif self.stopped_at > 0 and self.started_at > 0:
            runtime = self.stopped_at - self.started_at
        else:
            runtime = 0

        remaining = 0
        if self.timeout_at > 0 and self.status == ProcessStatus.RUNNING:
            remaining = max(0, self.timeout_at - now)

        return {
            "process_type": self.process_type.value,
            "strategy": self.strategy,
            "status": self.status.value,
            "pid": self.pid,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "timeout_at": self.timeout_at,
            "remaining": remaining,
            "runtime": runtime,
            "return_code": self.return_code,
            "error": self.error,
            "stats": self.stats.to_dict(),
            "peak_stats": self.peak_stats.to_dict(),
            "command": self.command,
            "output_tail": self.output_lines[-50:] if self.output_lines else [],
        }


class AppState:
    """Thread-safe application state."""

    def __init__(self):
        self._lock = threading.RLock()
        self.processes: dict[str, ProcessInfo] = {}
        self.workflow_status: dict[str, WorkflowStatus] = {}
        self.workflow_step: dict[str, str] = {}
        self.hyperopt_epochs: dict[str, list[EpochResult]] = {}
        self.hyperopt_best: dict[str, EpochResult | None] = {}
        self.monitor_state: dict[str, dict] = {}  # Per-strategy monitor state for frontend restore
        self.log_buffer: list[str] = []
        self._ws_clients: list[asyncio.Queue] = []
        self._max_log_lines = 1000
        self._max_output_lines = 5000

    def _proc_key(self, ptype: ProcessType, strategy: str) -> str:
        return f"{ptype.value}:{strategy}"

    def set_process(self, ptype: ProcessType, strategy: str, info: ProcessInfo):
        with self._lock:
            self.processes[self._proc_key(ptype, strategy)] = info

    def get_process(self, ptype: ProcessType, strategy: str) -> ProcessInfo | None:
        with self._lock:
            return self.processes.get(self._proc_key(ptype, strategy))

    def append_output(self, ptype: ProcessType, strategy: str, line: str):
        with self._lock:
            key = self._proc_key(ptype, strategy)
            if key in self.processes:
                self.processes[key].output_lines.append(line)
                if len(self.processes[key].output_lines) > self._max_output_lines:
                    self.processes[key].output_lines = self.processes[key].output_lines[-self._max_output_lines:]

    def update_stats(self, ptype: ProcessType, strategy: str, stats: ProcessStats):
        with self._lock:
            key = self._proc_key(ptype, strategy)
            if key in self.processes:
                proc = self.processes[key]
                # Only accept stats for active processes (avoid ghost updates after stop)
                if proc.status not in (ProcessStatus.RUNNING, ProcessStatus.STARTING):
                    return
                proc.stats = stats
                # Preserve last non-zero stats for post-completion display
                if stats.cpu_percent > 0 or stats.memory_mb > 0:
                    proc.peak_stats = ProcessStats(
                        cpu_percent=stats.cpu_percent,
                        memory_mb=stats.memory_mb,
                        num_threads=stats.num_threads,
                        gpu_util=stats.gpu_util,
                        gpu_mem_mb=stats.gpu_mem_mb,
                        gpu_temp=stats.gpu_temp,
                        gpu_fan=stats.gpu_fan,
                        gpu_name=stats.gpu_name,
                        updated_at=stats.updated_at,
                    )

    def set_workflow(self, strategy: str, status: WorkflowStatus, step: str = ""):
        with self._lock:
            self.workflow_status[strategy] = status
            self.workflow_step[strategy] = step

    def add_epoch(self, strategy: str, epoch: EpochResult):
        with self._lock:
            if strategy not in self.hyperopt_epochs:
                self.hyperopt_epochs[strategy] = []
            self.hyperopt_epochs[strategy].append(epoch)

    def set_best_epoch(self, strategy: str, epoch: EpochResult | None):
        with self._lock:
            self.hyperopt_best[strategy] = epoch

    def clear_epochs(self, strategy: str):
        with self._lock:
            self.hyperopt_epochs[strategy] = []
            self.hyperopt_best[strategy] = None

    def clear_monitor_state(self, strategy: str):
        with self._lock:
            self.monitor_state.pop(strategy, None)

    def add_log(self, msg: str):
        with self._lock:
            self.log_buffer.append(msg)
            if len(self.log_buffer) > self._max_log_lines:
                self.log_buffer = self.log_buffer[-self._max_log_lines:]
        self.broadcast("system_log", {"message": msg})

    def get_full_state(self) -> dict:
        with self._lock:
            return {
                "processes": {k: v.to_dict() for k, v in self.processes.items()},
                "workflows": {
                    k: {"status": v.value, "step": self.workflow_step.get(k, "")}
                    for k, v in self.workflow_status.items()
                },
                "hyperopt": {
                    k: {
                        "total_epochs": len(v),
                        "best": self.hyperopt_best.get(k, None) and self.hyperopt_best[k].to_dict(),
                        "recent": [e.to_dict() for e in v[-20:]],
                    }
                    for k, v in self.hyperopt_epochs.items()
                },
                "monitor": dict(self.monitor_state),
                "logs": self.log_buffer[-500:],
            }

    def register_ws(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._ws_clients.append(q)
        return q

    def unregister_ws(self, q: asyncio.Queue):
        if q in self._ws_clients:
            self._ws_clients.remove(q)

    def broadcast(self, event_type: str, data: dict):
        # Auto-capture monitor state from hyperopt_monitor_status events
        if event_type == "hyperopt_monitor_status":
            self._update_monitor_state(data)

        msg = json.dumps({"type": event_type, "data": data, "ts": time.time()})
        dead = []
        for q in self._ws_clients:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._ws_clients.remove(q)

    def _update_monitor_state(self, data: dict):
        """Keep server-side mirror of frontend monitorState for restore on page load."""
        strat = data.get("strategy")
        if not strat:
            return
        with self._lock:
            ms = self.monitor_state.get(strat, {
                "file": None, "waiting": False, "old_file": None,
                "last_best": None, "last_extracted": None, "last_action": None,
            })
            action = data.get("action")
            if action == "waiting":
                ms["waiting"] = True
                ms["old_file"] = data.get("old_file")
                ms["file"] = None
            elif action == "watching":
                ms["waiting"] = False
                ms["file"] = data.get("file")
            elif action == "new_best" and data.get("epoch"):
                ms["last_best"] = data["epoch"]
                ms["last_action"] = "new_best"
            elif action == "extracted" and data.get("epoch"):
                ms["last_extracted"] = data["epoch"]
                ms["last_action"] = "extracted"
            elif action == "reloaded":
                ms["last_action"] = "reloaded"
            elif action == "extract_failed":
                ms["last_action"] = "extract_failed"
            self.monitor_state[strat] = ms
