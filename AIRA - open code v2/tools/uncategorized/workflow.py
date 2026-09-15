"""Workflow orchestrator for the automated trading cycle.

Flow: stop trade → download data → backtest → hyperopt → monitor → extract best → restart trade
Each step respects its own enabled flag in config.
"""

import os
import glob
import json
import threading
import time
import logging

from typing import Callable

from .config import AppConfig, StrategyConfig
from .state import AppState, ProcessType, WorkflowStatus, EpochResult
from .process_manager import ProcessManager
from .hyperopt_monitor import HyperoptMonitor

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    pass


class Workflow:
    """Orchestrates the complete optimization workflow for a strategy."""

    def __init__(
        self,
        config: AppConfig,
        state: AppState,
        proc_mgr: ProcessManager,
        hyperopt_mon: HyperoptMonitor,
        config_path: str = "",
    ):
        self.config = config
        self.state = state
        self.proc_mgr = proc_mgr
        self.hyperopt_mon = hyperopt_mon
        self._config_path = config_path
        self._running: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._on_complete_callbacks: list[Callable] = []  # Called with (strategy_name) on any finish

    def register_on_complete(self, callback: Callable):
        """Register a callback to be called when any workflow finishes (success or failure)."""
        self._on_complete_callbacks.append(callback)

    def is_running(self, strategy_name: str) -> bool:
        return strategy_name in self._running and self._running[strategy_name].is_alive()

    def wait(self, strategy_name: str, timeout: float | None = None):
        """Block until workflow thread for this strategy finishes."""
        t = self._running.get(strategy_name)
        if t and t.is_alive():
            t.join(timeout=timeout)

    def cancel(self, strategy_name: str):
        if strategy_name in self._cancel_events:
            logger.info(f"Cancelling workflow for {strategy_name}")
            self._cancel_events[strategy_name].set()
            self.state.set_workflow(strategy_name, WorkflowStatus.CANCELLED, "Cancelling...")

    def start(self, strategy: StrategyConfig):
        if self.is_running(strategy.name):
            logger.warning(f"Workflow already running for {strategy.name}")
            return False

        cancel_event = threading.Event()
        self._cancel_events[strategy.name] = cancel_event

        t = threading.Thread(
            target=self._run_workflow,
            args=(strategy, cancel_event),
            name=f"workflow-{strategy.name}",
            daemon=True,
        )
        self._running[strategy.name] = t
        t.start()
        logger.info(f"Started workflow for {strategy.name}")
        return True

    def _run_workflow(self, strategy: StrategyConfig, cancel_event: threading.Event):
        name = strategy.name
        trade_was_running = self.proc_mgr.is_running(ProcessType.TRADE, name)
        restart_mode = strategy.restart_mode

        # Build summary of enabled steps
        steps_summary = []
        if strategy.download_data.enabled: steps_summary.append("download")
        if strategy.backtest.enabled: steps_summary.append("backtest")
        if strategy.hyperopt.enabled:
            h_parts = ["hyperopt"]
            if strategy.extract.enabled: h_parts.append("+extract")
            if strategy.restart.enabled: h_parts.append(f"+reload({restart_mode})")
            steps_summary.append("".join(h_parts))
        else:
            steps_summary.append("hyperopt:off")

        try:
            self.state.set_workflow(name, WorkflowStatus.RUNNING, "Starting workflow")
            # Clear previous monitor state so frontend doesn't show stale info
            self.state.clear_monitor_state(name)
            self.state.add_log(
                f"[workflow:{name}] === WORKFLOW STARTED === "
                f"steps=[{', '.join(steps_summary)}]"
            )
            self.state.broadcast("workflow_started", {"strategy": name})

            # Cleanup old fthypt files if configured
            if strategy.schedule.cleanup_days > 0:
                self._cleanup_old_fthypt(strategy)

            # Step 1: Stop trade if running (only for hard restart mode)
            if restart_mode == "hard" and strategy.restart.enabled:
                if self.proc_mgr.is_running(ProcessType.TRADE, name):
                    self._step(name, "Stopping trade (hard)...")
                    self.state.add_log(f"[workflow:{name}] Stopping active trade (hard restart mode)")
                    self.proc_mgr.stop_process(ProcessType.TRADE, name, timeout=60)
                    time.sleep(3)
                    if cancel_event.is_set():
                        raise WorkflowError("Cancelled")

            # Step 2: Download data
            if strategy.download_data.enabled:
                self._step(name, "Downloading data...")
                self.state.add_log(f"[workflow:{name}] Downloading pair data")
                rc = self.proc_mgr.run_and_wait(
                    ProcessType.DOWNLOAD, strategy,
                    cancel_event=cancel_event, timeout=3600,
                )
                if cancel_event.is_set():
                    raise WorkflowError("Cancelled")
                if rc != 0:
                    raise WorkflowError(f"Download failed with code {rc}")
            else:
                self.state.add_log(f"[workflow:{name}] Download: SKIPPED (disabled)")

            # Step 3: Backtest (updates LSTM models)
            if strategy.backtest.enabled:
                self._step(name, "Running backtest...")
                self.state.add_log(f"[workflow:{name}] Running backtest (model update)")
                rc = self.proc_mgr.run_and_wait(
                    ProcessType.BACKTEST, strategy,
                    cancel_event=cancel_event, timeout=7200,
                )
                if cancel_event.is_set():
                    raise WorkflowError("Cancelled")
                if rc != 0:
                    raise WorkflowError(f"Backtest failed with code {rc}")
            else:
                self.state.add_log(f"[workflow:{name}] Backtest: SKIPPED (disabled)")

            # Step 4: Run hyperopt with live extract+reload on new best
            best_epoch = None
            if strategy.hyperopt.enabled:
                self._step(name, "Running hyperopt...")
                best_epoch = self._run_hyperopt_with_monitoring(
                    strategy, cancel_event, restart_mode=restart_mode,
                )
                if cancel_event.is_set():
                    raise WorkflowError("Cancelled")
                if best_epoch:
                    self.state.add_log(
                        f"[workflow:{name}] Hyperopt complete. Final best: epoch {best_epoch.epoch} "
                        f"(profit={best_epoch.profit_total_pct:.2f}%, dd={best_epoch.max_drawdown:.2f}%)"
                    )
                else:
                    self.state.add_log(f"[workflow:{name}] Hyperopt complete. No epoch matched criteria.")
            else:
                self.state.add_log(f"[workflow:{name}] Hyperopt: SKIPPED (disabled)")

            # Ensure trade is running at the end (in case no best was found to trigger reload,
            # or hyperopt was disabled but trade was stopped for download/backtest)
            if trade_was_running and not self.proc_mgr.is_running(ProcessType.TRADE, name):
                self.state.add_log(f"[workflow:{name}] Restoring trade (was running before workflow)")
                self.proc_mgr.start_process(ProcessType.TRADE, strategy)

            self.state.set_workflow(name, WorkflowStatus.COMPLETED, "Workflow completed")
            self.state.add_log(f"[workflow:{name}] === WORKFLOW COMPLETED ===")
            self.state.broadcast("workflow_completed", {"strategy": name})

        except WorkflowError as e:
            logger.error(f"Workflow error for {name}: {e}")
            self.state.set_workflow(name, WorkflowStatus.FAILED, str(e))
            self.state.add_log(f"[workflow:{name}] FAILED: {e}")
            self.state.broadcast("workflow_failed", {"strategy": name, "error": str(e)})

            # Try to restart trade on failure if it was running before
            if trade_was_running and not self.proc_mgr.is_running(ProcessType.TRADE, name):
                self.state.add_log(f"[workflow:{name}] Attempting to restart trade after failure")
                self.proc_mgr.start_process(ProcessType.TRADE, strategy)

        except Exception as e:
            logger.exception(f"Unexpected error in workflow for {name}")
            self.state.set_workflow(name, WorkflowStatus.FAILED, f"Unexpected: {e}")
            self.state.add_log(f"[workflow:{name}] UNEXPECTED ERROR: {e}")
            self.state.broadcast("workflow_failed", {"strategy": name, "error": str(e)})

            if trade_was_running and not self.proc_mgr.is_running(ProcessType.TRADE, name):
                self.state.add_log(f"[workflow:{name}] Attempting to restart trade after error")
                self.proc_mgr.start_process(ProcessType.TRADE, strategy)

        finally:
            self.hyperopt_mon.stop_monitoring(name)
            if name in self._running:
                del self._running[name]
            if name in self._cancel_events:
                del self._cancel_events[name]
            # Notify listeners (scheduler) that workflow finished
            for cb in self._on_complete_callbacks:
                try:
                    cb(name)
                except Exception as e:
                    logger.error(f"on_complete callback error: {e}")

    def _step(self, strategy_name: str, step: str):
        logger.info(f"[workflow:{strategy_name}] {step}")
        self.state.set_workflow(strategy_name, WorkflowStatus.RUNNING, step)
        self.state.broadcast("workflow_step", {"strategy": strategy_name, "step": step})

    def _cleanup_old_fthypt(self, strategy: StrategyConfig):
        """Delete fthypt files older than cleanup_days. Broadcasts refresh to frontend."""
        days = strategy.schedule.cleanup_days
        base_dir = os.path.join(self.config.freqtrade_dir, "user_data", "hyperopt_results")
        if not os.path.isdir(base_dir):
            return

        cutoff = time.time() - (days * 86400)
        pattern = os.path.join(base_dir, f"*{strategy.strategy_name}*.fthypt")
        deleted = 0

        for fpath in glob.glob(pattern):
            try:
                if os.path.getmtime(fpath) < cutoff:
                    fname = os.path.basename(fpath)
                    os.remove(fpath)
                    deleted += 1
                    logger.info(f"Cleaned up old fthypt: {fname}")
            except Exception as e:
                logger.warning(f"Failed to delete {fpath}: {e}")

        if deleted > 0:
            self.state.add_log(
                f"[workflow:{strategy.name}] Cleaned up {deleted} fthypt file(s) older than {days} days"
            )
            # Tell frontend to refresh the file list
            self.state.broadcast("fthypt_files_changed", {"strategy": strategy.name})

    def _run_hyperopt_with_monitoring(
        self, strategy: StrategyConfig, cancel_event: threading.Event,
        restart_mode: str = "hard",
    ) -> EpochResult | None:
        last_extracted_epoch = [None]  # Track which epoch was last extracted
        best_found = [None]
        live_mode = strategy.extract.on_new_best  # True=extract on each best, False=after hyperopt

        def on_best(epoch: EpochResult):
            """Called when a new best epoch matching criteria is found."""
            best_found[0] = epoch

            self.state.add_log(
                f"[workflow:{strategy.name}] ★ New best: epoch {epoch.epoch} "
                f"(profit={epoch.profit_total_pct:.2f}%, dd={epoch.max_drawdown:.2f}%)"
            )

            # Broadcast the new best candidate
            self.state.broadcast("hyperopt_monitor_status", {
                "strategy": strategy.name,
                "action": "new_best",
                "epoch": epoch.to_dict(),
            })

            if not live_mode:
                # Deferred mode: just track, extract after hyperopt finishes
                return

            # Live mode: extract+reload on each new best
            if last_extracted_epoch[0] == epoch.epoch:
                return

            # Skip extract/reload if profit is not positive
            if epoch.profit_total_pct <= 0:
                self.state.add_log(
                    f"[workflow:{strategy.name}] Skipping extract — profit {epoch.profit_total_pct:.2f}% <= 0"
                )
                return

            if strategy.extract.enabled:
                self.state.add_log(f"[workflow:{strategy.name}] Extracting epoch {epoch.epoch}...")
                success = self._extract_epoch(strategy, epoch.epoch)
                if not success:
                    self.state.add_log(f"[workflow:{strategy.name}] Extract failed for epoch {epoch.epoch}")
                    self.state.broadcast("hyperopt_monitor_status", {
                        "strategy": strategy.name,
                        "action": "extract_failed",
                        "epoch": epoch.to_dict(),
                    })
                    return
                last_extracted_epoch[0] = epoch.epoch

                self.state.broadcast("hyperopt_monitor_status", {
                    "strategy": strategy.name,
                    "action": "extracted",
                    "epoch": epoch.to_dict(),
                })

                # Reload trade if enabled
                if strategy.restart.enabled:
                    self.state.add_log(f"[workflow:{strategy.name}] Reloading trade after new best...")
                    self._reload_trade(strategy, restart_mode, cancel_event)
                    self.state.broadcast("hyperopt_monitor_status", {
                        "strategy": strategy.name,
                        "action": "reloaded",
                        "epoch": epoch.to_dict(),
                    })

        mode_desc = "live extract+reload on new best" if live_mode else "extract best after completion"
        self.state.add_log(f"[workflow:{strategy.name}] Starting hyperopt ({mode_desc})")
        self.hyperopt_mon.start_monitoring(strategy, on_best_found=on_best)

        def on_line(line: str):
            self.hyperopt_mon.process_console_line(strategy, line)

        timeout = strategy.hyperopt.timeout_minutes * 60 if strategy.hyperopt.timeout_minutes > 0 else 0

        rc = self.proc_mgr.run_and_wait(
            ProcessType.HYPEROPT, strategy,
            on_line=on_line, timeout=timeout, cancel_event=cancel_event,
        )

        self.hyperopt_mon.stop_monitoring(strategy.name)

        if cancel_event.is_set():
            return None

        if rc != 0 and rc != -1:
            logger.warning(f"Hyperopt ended with code {rc}")

        # ── DEFINITIVE EVALUATION ──────────────────────────────────────
        # Parse the FULL fthypt file and evaluate ALL epochs against criteria.
        # This is the SAME code path as the HYPEROPT RESULTS panel API endpoint.
        # During live monitoring, the incremental evaluator may have extracted a
        # suboptimal epoch (e.g. #7 was best among epochs 1-7, but #3 is the true
        # best across all 18 epochs). This final pass corrects that.
        from .hyperopt_monitor import parse_fthypt_file, evaluate_criteria

        # Reload criteria from config (same as HYPEROPT RESULTS API does)
        from .config import load_config
        try:
            fresh_cfg = load_config(self._config_path)
            fresh_strat = fresh_cfg.get_strategy(strategy.name)
            criteria = fresh_strat.epoch_criteria if fresh_strat else strategy.epoch_criteria
        except Exception:
            criteria = strategy.epoch_criteria

        base_dir = os.path.join(self.config.freqtrade_dir, "user_data", "hyperopt_results")
        lr_path = os.path.join(base_dir, ".last_result.json")
        final_best = None
        try:
            with open(lr_path, "r", encoding="utf-8") as f:
                lr_data = json.loads(f.read())
            fthypt_name = lr_data.get("latest_hyperopt", "")
            if fthypt_name:
                fthypt_path = os.path.join(base_dir, fthypt_name)
                all_epochs = parse_fthypt_file(fthypt_path)
                final_best = evaluate_criteria(all_epochs, criteria)
                if final_best:
                    self.state.add_log(
                        f"[workflow:{strategy.name}] Final evaluation ({len(all_epochs)} epochs): "
                        f"best = epoch {final_best.epoch} "
                        f"(profit={final_best.profit_total_pct:.2f}%, dd={final_best.max_drawdown:.2f}%)"
                    )
                    self.state.set_best_epoch(strategy.name, final_best)
                    self.state.broadcast("hyperopt_monitor_status", {
                        "strategy": strategy.name,
                        "action": "new_best",
                        "epoch": final_best.to_dict(),
                    })
                else:
                    self.state.add_log(
                        f"[workflow:{strategy.name}] Final evaluation ({len(all_epochs)} epochs): "
                        f"no epoch matched criteria"
                    )
        except Exception as e:
            self.state.add_log(f"[workflow:{strategy.name}] Final evaluation error: {e}")
            logger.error(f"Final fthypt evaluation error: {e}")

        # ── EXTRACT + RELOAD (if needed) ───────────────────────────────
        # Use final_best as the definitive result for extraction.
        # In live mode: re-extract only if final best differs from what was already extracted.
        # In deferred mode: extract the final best (nothing was extracted during hyperopt).
        if final_best and strategy.extract.enabled:
            need_extract = False

            if live_mode:
                if last_extracted_epoch[0] != final_best.epoch:
                    self.state.add_log(
                        f"[workflow:{strategy.name}] Final best (epoch {final_best.epoch}) differs from "
                        f"last extracted (epoch {last_extracted_epoch[0]}) — correcting"
                    )
                    need_extract = True
                # else: already extracted the correct epoch during live monitoring
            else:
                # Deferred mode: nothing extracted yet
                need_extract = True

            if need_extract and final_best.profit_total_pct > 0:
                self.state.add_log(
                    f"[workflow:{strategy.name}] Extracting best: epoch {final_best.epoch} "
                    f"(profit={final_best.profit_total_pct:.2f}%, dd={final_best.max_drawdown:.2f}%)"
                )
                success = self._extract_epoch(strategy, final_best.epoch)
                if success:
                    self.state.broadcast("hyperopt_monitor_status", {
                        "strategy": strategy.name,
                        "action": "extracted",
                        "epoch": final_best.to_dict(),
                    })
                    if strategy.restart.enabled:
                        self.state.add_log(f"[workflow:{strategy.name}] Reloading trade with best epoch...")
                        self._reload_trade(strategy, restart_mode, cancel_event)
                        self.state.broadcast("hyperopt_monitor_status", {
                            "strategy": strategy.name,
                            "action": "reloaded",
                            "epoch": final_best.to_dict(),
                        })
                else:
                    self.state.add_log(f"[workflow:{strategy.name}] Extract failed for epoch {final_best.epoch}")
                    self.state.broadcast("hyperopt_monitor_status", {
                        "strategy": strategy.name,
                        "action": "extract_failed",
                        "epoch": final_best.to_dict(),
                    })
            elif need_extract and final_best.profit_total_pct <= 0:
                self.state.add_log(
                    f"[workflow:{strategy.name}] Skipping extract — best epoch profit "
                    f"{final_best.profit_total_pct:.2f}% <= 0"
                )

        return final_best

    def _reload_trade(self, strategy: StrategyConfig, restart_mode: str,
                       cancel_event: threading.Event):
        """Reload or restart the trade process."""
        try:
            if restart_mode == "graceful":
                self._graceful_reload(strategy, cancel_event)
            else:
                # Hard restart: stop then start
                if self.proc_mgr.is_running(ProcessType.TRADE, strategy.name):
                    self.proc_mgr.stop_process(ProcessType.TRADE, strategy.name, timeout=60)
                    time.sleep(3)
                self.proc_mgr.start_process(ProcessType.TRADE, strategy)
            self.state.add_log(f"[workflow:{strategy.name}] Trade reloaded ({restart_mode})")
        except Exception as e:
            self.state.add_log(f"[workflow:{strategy.name}] Trade reload failed: {e}")
            logger.error(f"Trade reload failed for {strategy.name}: {e}")

    def _extract_epoch(self, strategy: StrategyConfig, epoch_num: int) -> bool:
        cmd = self.proc_mgr.build_hyperopt_show_cmd(strategy, epoch_num)
        self.state.add_log(f"[workflow:{strategy.name}] Extracting: {' '.join(cmd)}")

        rc = self.proc_mgr.run_and_wait(
            ProcessType.HYPEROPT_SHOW, strategy, cmd=cmd, timeout=120,
        )
        if rc == 0:
            self.state.add_log(f"[workflow:{strategy.name}] Successfully extracted epoch {epoch_num}")
            return True
        else:
            self.state.add_log(f"[workflow:{strategy.name}] Extract failed (code {rc})")
            return False

    def _graceful_reload(self, strategy: StrategyConfig, cancel_event: threading.Event):
        """Reload trade config via freqtrade-client (no trade restart needed).
        Retries every 60s if the API is unresponsive (e.g. during heavy load)."""
        if not strategy.reload_client_config:
            self.state.add_log(
                f"[workflow:{strategy.name}] No reload_client_config set, falling back to hard restart"
            )
            if self.proc_mgr.is_running(ProcessType.TRADE, strategy.name):
                self.proc_mgr.stop_process(ProcessType.TRADE, strategy.name, timeout=60)
                time.sleep(3)
            self.proc_mgr.start_process(ProcessType.TRADE, strategy)
            return

        max_retries = 5
        retry_interval = 60

        for attempt in range(1, max_retries + 1):
            if cancel_event.is_set():
                return

            if attempt > 1:
                self.state.add_log(
                    f"[workflow:{strategy.name}] Reload retry {attempt}/{max_retries} "
                    f"(waiting {retry_interval}s)..."
                )
                # Wait with cancel check
                for _ in range(retry_interval):
                    if cancel_event.is_set():
                        return
                    time.sleep(1)

            self.state.add_log(
                f"[workflow:{strategy.name}] Sending reload_config via freqtrade-client"
                + (f" (attempt {attempt}/{max_retries})" if attempt > 1 else "")
            )

            rc = self.proc_mgr.run_and_wait(
                ProcessType.RELOAD, strategy, timeout=30, cancel_event=cancel_event,
            )
            if rc == 0:
                self.state.add_log(f"[workflow:{strategy.name}] Trade config reloaded successfully")
                return

            self.state.add_log(
                f"[workflow:{strategy.name}] Reload attempt {attempt} failed (code {rc})"
            )

        # All retries exhausted — fall back to hard restart
        self.state.add_log(
            f"[workflow:{strategy.name}] All reload attempts failed, falling back to hard restart"
        )
        if self.proc_mgr.is_running(ProcessType.TRADE, strategy.name):
            self.proc_mgr.stop_process(ProcessType.TRADE, strategy.name, timeout=60)
            time.sleep(3)
        self.proc_mgr.start_process(ProcessType.TRADE, strategy)
