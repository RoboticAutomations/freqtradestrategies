"""Scheduler for automated workflow execution — per-strategy schedules.

Two modes:
- cron: fires at fixed times (e.g. "0 2 * * *" = every day at 2 AM)
- interval_hours: fires N hours after last workflow COMPLETES (or from startup if never ran)
"""

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from .config import AppConfig, load_config
from .state import AppState
from .process_manager import ProcessManager
from .workflow import Workflow

logger = logging.getLogger(__name__)


class WorkflowScheduler:
    """Schedules workflow execution per strategy."""

    def __init__(self, config: AppConfig, state: AppState, workflow: Workflow,
                 proc_mgr: ProcessManager, config_path: str):
        self.config = config
        self.state = state
        self.workflow = workflow
        self.proc_mgr = proc_mgr
        self.config_path = config_path
        self.scheduler = BackgroundScheduler(
            job_defaults={"coalesce": True, "max_instances": 1}
        )
        self._job_ids: list[str] = []

    def _reload_config(self) -> AppConfig:
        """Reload config from disk and update references."""
        try:
            cfg = load_config(self.config_path)
            self.config = cfg
            self.proc_mgr.config = cfg
            return cfg
        except Exception as e:
            logger.error(f"Failed to reload config: {e} — using cached version")
            return self.config

    def start(self):
        """Start the scheduler with per-strategy jobs."""
        self._schedule_all()
        self.scheduler.start()
        # Register callback so ANY workflow completion reschedules interval mode
        self.workflow.register_on_complete(self._on_workflow_completed)
        logger.info("Scheduler started")

    def _on_workflow_completed(self, strategy_name: str):
        """Called when any workflow finishes (success, failure, or cancel).
        Reschedules interval-mode strategies from NOW = completion time."""
        try:
            cfg = self._reload_config()
            strat = cfg.get_strategy(strategy_name)
            if not strat or not strat.schedule.enabled:
                return
            if strat.schedule.interval_hours > 0:
                self._schedule_interval(
                    strategy_name,
                    strat.schedule.interval_hours,
                    from_time=datetime.now(timezone.utc),
                )
                logger.info(
                    f"Rescheduled {strategy_name} interval ({strat.schedule.interval_hours}h from now)"
                )
        except Exception as e:
            logger.error(f"Failed to reschedule {strategy_name} after completion: {e}")

    def reschedule(self):
        """Reload config from disk and reschedule all jobs."""
        self._reload_config()
        self._schedule_all()
        self.state.add_log("[scheduler] Jobs rescheduled from updated config")

    def _schedule_all(self):
        """Remove all existing jobs and recreate from current config."""
        for job_id in self._job_ids:
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass
        self._job_ids.clear()

        for strategy in self.config.strategies:
            if not strategy.enabled:
                continue
            if not strategy.schedule.enabled:
                self.state.add_log(f"[scheduler] {strategy.name}: disabled")
                continue

            sched = strategy.schedule

            if sched.interval_hours > 0:
                # Interval mode: schedule first run N hours from now (startup)
                self._schedule_interval(strategy.name, sched.interval_hours)
            else:
                # Cron mode: fixed schedule
                self._schedule_cron(strategy.name, sched.cron)

    def _schedule_cron(self, strat_name: str, cron_expr: str):
        """Schedule a strategy with a cron trigger."""
        job_id = f"workflow-{strat_name}"
        parts = cron_expr.split()
        if len(parts) != 5:
            logger.error(f"Invalid cron for {strat_name}: {cron_expr}")
            return

        trigger = CronTrigger(
            minute=parts[0], hour=parts[1],
            day=parts[2], month=parts[3],
            day_of_week=parts[4],
        )
        self.scheduler.add_job(
            lambda name=strat_name: self._run_workflow_by_name(name),
            trigger=trigger,
            id=job_id,
            name=f"Workflow for {strat_name}",
            replace_existing=True,
        )
        self._job_ids.append(job_id)
        self.state.add_log(f"[scheduler] {strat_name}: cron {cron_expr}")

    def _schedule_interval(self, strat_name: str, hours: float, from_time: datetime | None = None):
        """Schedule a one-shot run at from_time + hours. After completion, reschedules itself."""
        job_id = f"workflow-{strat_name}"
        base = from_time or datetime.now(timezone.utc)
        run_at = base + timedelta(hours=hours)

        trigger = DateTrigger(run_date=run_at)
        self.scheduler.add_job(
            lambda name=strat_name, h=hours: self._run_interval_workflow(name, h),
            trigger=trigger,
            id=job_id,
            name=f"Workflow for {strat_name}",
            replace_existing=True,
        )
        if job_id not in self._job_ids:
            self._job_ids.append(job_id)

        local_str = run_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.state.add_log(f"[scheduler] {strat_name}: every {hours}h → next at {local_str}")

    def _run_interval_workflow(self, strategy_name: str, interval_hours: float):
        """Run workflow. Rescheduling is handled by _on_workflow_completed callback."""
        self._run_workflow_by_name(strategy_name)

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    def trigger_now(self, strategy_name: str) -> bool:
        """Manually trigger a workflow run immediately."""
        return self._run_workflow_by_name(strategy_name)

    def _run_workflow_by_name(self, strategy_name: str) -> bool:
        """Reload config, find strategy, run workflow."""
        cfg = self._reload_config()
        strategy = cfg.get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Strategy not found in config: {strategy_name}")
            self.state.add_log(f"[scheduler] ERROR: Strategy {strategy_name} not found in config")
            return False
        if self.workflow.is_running(strategy.name):
            logger.warning(f"Workflow already running for {strategy.name}, skipping")
            self.state.add_log(f"[scheduler] Skipped {strategy.name}: already running")
            return False
        self.state.add_log(f"[scheduler] Triggering workflow for {strategy.name}")
        return self.workflow.start(strategy)

    def get_jobs_info(self) -> list[dict]:
        """Get info about scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = None
            if job.next_run_time:
                # Convert to local time for display
                next_run = job.next_run_time.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run,
                "trigger": str(job.trigger),
            })
        return jobs
