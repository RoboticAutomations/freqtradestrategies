"""Hyperopt monitor - reads .fthypt results file and evaluates epochs against criteria.

FreqTrade stores hyperopt results in:
  user_data/hyperopt_results/strategy_<StrategyName>.fthypt

Each line is a JSON object with the epoch's metrics. This module monitors
that file for new lines and evaluates each epoch against the user's criteria.

Fallback: Also parses console output lines for epoch results.
"""

import os
import json
import re
import time
import threading
import logging
from typing import Callable

from .config import AppConfig, StrategyConfig, EpochCriterion
from .state import AppState, EpochResult

logger = logging.getLogger(__name__)

# Regex for parsing console output lines (fallback)
# Example: "   2/2000:     45 trades. 10/15/20 Wins/Draws/Losses. Avg profit   0.52%..."
EPOCH_LINE_RE = re.compile(
    r"^\s*\*?\s*(\d+)/(\d+):\s+"
    r"(\d+)\s+trades\.\s+"
    r"(?:(\d+)/(\d+)/(\d+)\s+Wins/Draws/Losses\.\s+)?"
    r"Avg profit\s+([\-\d.]+)%\."
    r".*?Total profit\s+([\-\d.]+)\s+\S+\s+\(\s*([\-\d.]+)"
)

# Pattern for drawdown in summary
DRAWDOWN_RE = re.compile(r"Max [Dd]rawdown\s*:?\s*([\-\d.]+)%")
OBJECTIVE_RE = re.compile(r"Objective:\s*([\-\d.]+)")


def parse_fthypt_file(file_path: str) -> list[EpochResult]:
    """Parse all epochs from a .fthypt file."""
    epochs = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                epoch = parse_fthypt_line(line, i)
                if epoch:
                    epochs.append(epoch)
    except Exception:
        pass
    return epochs


def parse_fthypt_line(line: str, epoch_num: int) -> EpochResult | None:
    """Parse a single line from .fthypt file into EpochResult.

    Field names from freqtrade source (hyperopt_tools.py):
      results_metrics.trade_count, .trade_count_long, .trade_count_short
      results_metrics.wins, .draws, .losses
      results_metrics.profit_mean (ratio), .profit_median (ratio)
      results_metrics.profit_total (ratio), .profit_total_abs
      results_metrics.max_drawdown_account (ratio), .max_drawdown_abs
      results_metrics.holding_avg
    """
    try:
        data = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    metrics = data.get("results_metrics", {})
    if not metrics:
        return None

    # trade_count: try direct field, then sum of long+short
    trade_count = metrics.get("trade_count", 0)
    if not trade_count:
        trade_count = metrics.get("trade_count_long", 0) + metrics.get("trade_count_short", 0)
    if not trade_count:
        trade_count = metrics.get("wins", 0) + metrics.get("draws", 0) + metrics.get("losses", 0)

    # profit_mean is a ratio (e.g. 0.01 = 1%)
    profit_mean = metrics.get("profit_mean", 0.0) or 0.0
    avg_profit = profit_mean * 100

    # profit_total is a ratio (e.g. 0.05 = 5%)
    profit_total = metrics.get("profit_total", 0.0) or 0.0
    profit_total_pct = profit_total * 100

    # max_drawdown_account is a ratio (e.g. 0.02 = 2%)
    max_dd_account = metrics.get("max_drawdown_account", 0.0) or 0.0
    max_drawdown_pct = max_dd_account * 100

    return EpochResult(
        epoch=data.get("current_epoch", epoch_num),
        trades=trade_count,
        wins=metrics.get("wins", 0),
        draws=metrics.get("draws", 0),
        losses=metrics.get("losses", 0),
        avg_profit=avg_profit,
        profit_total_pct=profit_total_pct,
        profit_total_abs=metrics.get("profit_total_abs", 0.0) or 0.0,
        max_drawdown=max_drawdown_pct,
        max_drawdown_abs=metrics.get("max_drawdown_abs", 0.0) or 0.0,
        avg_duration=str(metrics.get("holding_avg", "")),
        objective=data.get("loss", 0.0),
        is_best=data.get("is_best", False),
        params=data.get("params_dict", {}),
    )


def parse_console_line(line: str) -> EpochResult | None:
    """Parse an epoch result from freqtrade console output (fallback)."""
    m = EPOCH_LINE_RE.search(line)
    if not m:
        return None

    epoch_num = int(m.group(1))
    trades = int(m.group(3))
    wins = int(m.group(4)) if m.group(4) else 0
    draws = int(m.group(5)) if m.group(5) else 0
    losses = int(m.group(6)) if m.group(6) else 0
    avg_profit = float(m.group(7))
    profit_total_abs = float(m.group(8))
    profit_total_pct = float(m.group(9))

    obj_m = OBJECTIVE_RE.search(line)
    objective = float(obj_m.group(1)) if obj_m else 0.0

    is_best = line.strip().startswith("*")

    return EpochResult(
        epoch=epoch_num,
        trades=trades,
        wins=wins,
        draws=draws,
        losses=losses,
        avg_profit=avg_profit,
        profit_total_pct=profit_total_pct,
        profit_total_abs=profit_total_abs,
        objective=objective,
        is_best=is_best,
    )


def evaluate_criteria(epochs: list[EpochResult], criteria: list[EpochCriterion]) -> EpochResult | None:
    """Evaluate epochs against criteria chain.

    1. Apply ALL filter conditions (operator + value) to narrow candidates
    2. Apply multi-key sort: first criterion's sort is PRIMARY, subsequent are tiebreakers
       (stable sort applied in reverse order of priority)

    Returns the best epoch or None if no epoch passes all filters.
    """
    if not epochs:
        return None

    # Pre-filter: remove degenerate epochs (0 trades = no meaningful metrics)
    candidates = [e for e in epochs if e.trades > 0]
    logger.debug(f"evaluate_criteria: {len(epochs)} total, {len(candidates)} with trades>0")

    # Step 1: Apply all filters
    for criterion in criteria:
        if criterion.operator and criterion.value is not None:
            before = len(candidates)
            candidates = [e for e in candidates if _check_condition(e, criterion)]
            logger.debug(
                f"  filter {criterion.field} {criterion.operator} {criterion.value}: "
                f"{before} → {len(candidates)} candidates"
            )
        if not candidates:
            logger.debug(f"  no candidates left after {criterion.field} filter")
            return None

    # Step 2: Multi-key stable sort
    # Criteria are listed in priority order: LAST sort = most important (primary).
    # Stable sort: apply tiebreakers first, then primary last.
    # Example: [max_drawdown asc, profit_total_pct desc]
    #   Pass 1: sort by max_drawdown asc (tiebreaker)
    #   Pass 2: sort by profit_total_pct desc (primary — preserves dd order within ties)
    sort_criteria = [(c.field, c.sort.lower() == "desc") for c in criteria if c.sort]
    for field_name, reverse in sort_criteria:
        candidates.sort(key=lambda e, fn=field_name: _get_field(e, fn), reverse=reverse)

    if candidates:
        best = candidates[0]
        logger.debug(
            f"  best: epoch #{best.epoch} "
            f"profit={best.profit_total_pct:.2f}% dd={best.max_drawdown:.2f}% "
            f"trades={best.trades} obj={best.objective:.5f}"
        )
        if len(candidates) > 1:
            runner_up = candidates[1]
            logger.debug(
                f"  runner-up: epoch #{runner_up.epoch} "
                f"profit={runner_up.profit_total_pct:.2f}% dd={runner_up.max_drawdown:.2f}%"
            )
    return candidates[0] if candidates else None


def _get_field(epoch: EpochResult, field_name: str) -> float:
    """Get a numeric field from an EpochResult."""
    return getattr(epoch, field_name, 0.0)


def _check_condition(epoch: EpochResult, criterion: EpochCriterion) -> bool:
    """Check if an epoch meets a single criterion condition."""
    val = _get_field(epoch, criterion.field)
    op = criterion.operator
    threshold = criterion.value

    if op == "<":
        return val < threshold
    elif op == "<=":
        return val <= threshold
    elif op == ">":
        return val > threshold
    elif op == ">=":
        return val >= threshold
    elif op == "==":
        return abs(val - threshold) < 1e-9
    else:
        logger.warning(f"Unknown operator: {op}")
        return False


class HyperoptMonitor:
    """Monitors hyperopt results in real-time.

    Primary: watches the .fthypt file for new epoch lines.
    Fallback: parses console output lines.
    """

    def __init__(self, config: AppConfig, state: AppState):
        self.config = config
        self.state = state
        self._monitors: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}

    def start_monitoring(
        self,
        strategy: StrategyConfig,
        on_best_found: Callable[[EpochResult], None] | None = None,
    ):
        """Start monitoring hyperopt results for a strategy."""
        if strategy.name in self._monitors:
            self.stop_monitoring(strategy.name)

        self.state.clear_epochs(strategy.name)
        stop_event = threading.Event()
        self._stop_events[strategy.name] = stop_event

        t = threading.Thread(
            target=self._monitor_loop,
            args=(strategy, stop_event, on_best_found),
            name=f"hyperopt-monitor-{strategy.name}",
            daemon=True,
        )
        self._monitors[strategy.name] = t
        t.start()
        logger.info(f"Started hyperopt monitor for {strategy.name}")

    def stop_monitoring(self, strategy_name: str):
        """Stop monitoring a strategy."""
        if strategy_name in self._stop_events:
            self._stop_events[strategy_name].set()
        if strategy_name in self._monitors:
            self._monitors[strategy_name].join(timeout=10)
            del self._monitors[strategy_name]
        if strategy_name in self._stop_events:
            del self._stop_events[strategy_name]
        logger.info(f"Stopped hyperopt monitor for {strategy_name}")

    def process_console_line(self, strategy: StrategyConfig, line: str):
        """Process a console output line (fallback parsing)."""
        epoch = parse_console_line(line)
        if epoch:
            self._process_epoch(strategy, epoch)

    def _monitor_loop(
        self,
        strategy: StrategyConfig,
        stop_event: threading.Event,
        on_best_found: Callable[[EpochResult], None] | None,
    ):
        """Main monitoring loop — watches .last_result.json for new fthypt files.

        CRITICAL: Only uses .last_result.json to detect files, never directory scan.
        This guarantees sync with `hyperopt-show -n X` which also reads .last_result.json.
        If the monitor reads epochs from file A, `hyperopt-show` must also read from file A.

        Uses file mtime (not filename comparison) to distinguish old vs new files,
        fixing the bug where .last_result.json still pointed to the previous run's file
        at monitor startup.
        """
        poll_interval = self.config.monitoring.poll_interval
        errors = 0
        max_errors = self.config.monitoring.max_errors
        epoch_counter = 0
        current_best: EpochResult | None = None

        base_dir = os.path.join(self.config.freqtrade_dir, "user_data", "hyperopt_results")
        last_result_path = os.path.join(base_dir, ".last_result.json")

        # Record start time — only accept files modified AFTER this
        start_time = time.time()
        file_path = None   # Current fthypt file we're reading
        file_pos = 0       # Read position in that file

        def _read_last_result() -> tuple[str | None, float]:
            """Read the fthypt filename from .last_result.json.
            Returns (full_path, mtime) or (None, 0)."""
            try:
                with open(last_result_path, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())
                fname = data.get("latest_hyperopt")
                if fname:
                    fpath = os.path.join(base_dir, fname)
                    if os.path.isfile(fpath):
                        return fpath, os.path.getmtime(fpath)
            except Exception:
                pass
            return None, 0

        initial_file, initial_mtime = _read_last_result()
        initial_name = os.path.basename(initial_file) if initial_file else None
        mtime_str = time.strftime('%H:%M:%S', time.localtime(initial_mtime)) if initial_mtime else ''
        logger.info(
            f"Monitor start for {strategy.name} at {time.strftime('%H:%M:%S', time.localtime(start_time))}"
            + (f" (existing: {initial_name}, mtime: {mtime_str})" if initial_name else "")
        )
        self.state.add_log(
            f"[hyperopt-monitor:{strategy.name}] Waiting for new fthypt file..."
            f"{f' (ignoring: {initial_name})' if initial_name else ''}"
        )

        # Broadcast waiting state to frontend
        self.state.broadcast("hyperopt_monitor_status", {
            "strategy": strategy.name,
            "action": "waiting",
            "old_file": initial_name,
        })

        while not stop_event.is_set():
            try:
                # Check .last_result.json — the ONLY source of truth
                current_fthypt, current_mtime = _read_last_result()

                if current_fthypt and current_fthypt != file_path:
                    # Decide if this is a new file or the old one from a previous run
                    is_new = False
                    if file_path is not None:
                        # We already have a file — any different file is new
                        is_new = True
                    elif initial_file and current_fthypt == initial_file:
                        # Same file as at startup — NEVER new, regardless of mtime
                        is_new = False
                    elif current_mtime >= start_time - 2:
                        # First detection, different file (or no initial), recently modified
                        is_new = True
                    elif initial_file and current_fthypt != initial_file:
                        # .last_result.json switched to a different file than at startup
                        is_new = True

                    if is_new:
                        fname = os.path.basename(current_fthypt)
                        if file_path:
                            logger.info(f"fthypt file changed → {fname}")
                            self.state.add_log(f"[hyperopt-monitor:{strategy.name}] File changed: {fname}")
                        else:
                            det_mtime = time.strftime('%H:%M:%S', time.localtime(current_mtime))
                            logger.info(f"New fthypt file detected: {fname} (mtime: {det_mtime})")
                            self.state.add_log(f"[hyperopt-monitor:{strategy.name}] Monitoring: {fname}")
                        file_path = current_fthypt
                        file_pos = 0
                        epoch_counter = 0
                        self.state.clear_epochs(strategy.name)
                        self.state.broadcast("hyperopt_monitor_status", {
                            "strategy": strategy.name,
                            "action": "watching",
                            "file": fname,
                        })

                # Read new content from current fthypt file
                if file_path and os.path.isfile(file_path):
                    current_size = os.path.getsize(file_path)
                    if current_size > file_pos:
                        with open(file_path, "r", encoding="utf-8") as f:
                            f.seek(file_pos)
                            new_data = f.read()
                            file_pos = f.tell()

                        for line in new_data.strip().split("\n"):
                            if not line.strip():
                                continue
                            epoch_counter += 1
                            epoch = parse_fthypt_line(line, epoch_counter)
                            if epoch:
                                self.state.add_epoch(strategy.name, epoch)
                                self.state.broadcast("hyperopt_epoch", {
                                    "strategy": strategy.name,
                                    "epoch": epoch.to_dict(),
                                })

                                # Evaluate criteria
                                all_epochs = self.state.hyperopt_epochs.get(strategy.name, [])
                                best = evaluate_criteria(all_epochs, strategy.epoch_criteria)
                                if best:
                                    prev_best = current_best
                                    current_best = best
                                    self.state.set_best_epoch(strategy.name, best)
                                    self.state.broadcast("hyperopt_best", {
                                        "strategy": strategy.name,
                                        "epoch": best.to_dict(),
                                    })
                                    if on_best_found and (prev_best is None or best.epoch != prev_best.epoch):
                                        try:
                                            on_best_found(best)
                                        except Exception as e:
                                            logger.error(f"on_best_found error: {e}")

                errors = 0

            except Exception as e:
                errors += 1
                logger.error(f"Monitor error for {strategy.name} ({errors}/{max_errors}): {e}")
                if errors >= max_errors:
                    logger.error(f"Max errors reached for {strategy.name}, stopping monitor")
                    self.state.add_log(f"[hyperopt-monitor:{strategy.name}] STOPPED: too many errors")
                    break

            stop_event.wait(timeout=poll_interval)

    def _process_epoch(self, strategy: StrategyConfig, epoch: EpochResult):
        """Process a single epoch result."""
        self.state.add_epoch(strategy.name, epoch)
        self.state.broadcast("hyperopt_epoch", {
            "strategy": strategy.name,
            "epoch": epoch.to_dict(),
        })
