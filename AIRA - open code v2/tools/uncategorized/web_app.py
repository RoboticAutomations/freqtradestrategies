"""FastAPI web application with WebSocket support."""

import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .config import AppConfig, StrategyConfig, load_config
from .state import AppState, ProcessType
from .process_manager import ProcessManager
from .hyperopt_monitor import HyperoptMonitor
from .workflow import Workflow
from .scheduler import WorkflowScheduler

logger = logging.getLogger(__name__)


class ConfigHolder:
    """Holds config and reloads from disk on every access."""

    def __init__(self, config_path: str, initial: AppConfig):
        self._path = config_path
        self._config = initial

    def get(self) -> AppConfig:
        try:
            self._config = load_config(self._path)
        except Exception as e:
            logger.error(f"Failed to reload config: {e} — using cached version")
        return self._config

    def get_strategy(self, name: str) -> StrategyConfig | None:
        return self.get().get_strategy(name)


def create_app(
    config: AppConfig,
    state: AppState,
    proc_mgr: ProcessManager,
    hyperopt_mon: HyperoptMonitor,
    workflow: Workflow,
    scheduler: WorkflowScheduler,
    config_path: str,
) -> FastAPI:
    app = FastAPI(title="FreqTrade Manager")
    cfg_holder = ConfigHolder(config_path, config)
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

    # Prevent browser caching of API responses
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class NoCacheMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/api/"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                response.headers["Pragma"] = "no-cache"
            return response

    app.add_middleware(NoCacheMiddleware)

    def _get_strat(strategy_name: str) -> StrategyConfig:
        cfg = cfg_holder.get()
        proc_mgr.config = cfg
        strat = cfg.get_strategy(strategy_name)
        if not strat:
            raise HTTPException(404, f"Strategy {strategy_name} not found")
        return strat

    # --- Pages ---
    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(os.path.join(templates_dir, "index.html"), "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    # --- State API ---
    @app.get("/api/state")
    async def get_state():
        return JSONResponse(state.get_full_state())

    @app.get("/api/strategies")
    async def get_strategies():
        cfg = cfg_holder.get()
        return JSONResponse([
            {"name": s.name, "strategy_name": s.strategy_name, "enabled": s.enabled,
             "restart_mode": s.restart_mode}
            for s in cfg.strategies
        ])

    @app.get("/api/schedule")
    async def get_schedule():
        cfg = cfg_holder.get()
        schedules = []
        for s in cfg.strategies:
            schedules.append({
                "strategy": s.name,
                "enabled": s.schedule.enabled,
                "cron": s.schedule.cron,
                "interval_hours": s.schedule.interval_hours,
            })
        return JSONResponse({
            "schedules": schedules,
            "jobs": scheduler.get_jobs_info(),
        })

    # --- Process Control ---
    @app.post("/api/trade/start/{strategy_name}")
    async def start_trade(strategy_name: str):
        strat = _get_strat(strategy_name)
        state.add_log(f"[action] Starting trade for {strategy_name}")
        ok = proc_mgr.start_process(ProcessType.TRADE, strat)
        return JSONResponse({"success": ok})

    @app.post("/api/trade/stop/{strategy_name}")
    async def stop_trade(strategy_name: str):
        state.add_log(f"[action] Stopping trade for {strategy_name}")
        ok = proc_mgr.stop_process(ProcessType.TRADE, strategy_name)
        return JSONResponse({"success": ok})

    @app.post("/api/trade/reload/{strategy_name}")
    async def reload_trade(strategy_name: str):
        """Graceful reload via freqtrade-client reload_config."""
        strat = _get_strat(strategy_name)
        if not strat.reload_client_config:
            raise HTTPException(400, "No reload_client_config configured for this strategy")
        state.add_log(f"[action] Graceful reload for {strategy_name}")
        ok = proc_mgr.start_process(ProcessType.RELOAD, strat)
        return JSONResponse({"success": ok})

    @app.post("/api/download/start/{strategy_name}")
    async def start_download(strategy_name: str):
        strat = _get_strat(strategy_name)
        state.add_log(f"[action] Starting download for {strategy_name}")
        ok = proc_mgr.start_process(ProcessType.DOWNLOAD, strat)
        return JSONResponse({"success": ok})

    @app.post("/api/backtest/start/{strategy_name}")
    async def start_backtest(strategy_name: str):
        strat = _get_strat(strategy_name)
        state.add_log(f"[action] Starting backtest for {strategy_name}")
        ok = proc_mgr.start_process(ProcessType.BACKTEST, strat)
        return JSONResponse({"success": ok})

    @app.post("/api/hyperopt/start/{strategy_name}")
    async def start_hyperopt(strategy_name: str):
        strat = _get_strat(strategy_name)
        state.add_log(f"[action] Starting hyperopt for {strategy_name}")
        ok = proc_mgr.start_process(ProcessType.HYPEROPT, strat)
        if ok:
            hyperopt_mon.start_monitoring(strat)
        return JSONResponse({"success": ok})

    @app.post("/api/hyperopt/stop/{strategy_name}")
    async def stop_hyperopt(strategy_name: str):
        state.add_log(f"[action] Stopping hyperopt for {strategy_name}")
        hyperopt_mon.stop_monitoring(strategy_name)
        ok = proc_mgr.stop_process(ProcessType.HYPEROPT, strategy_name)
        return JSONResponse({"success": ok})

    @app.post("/api/process/stop/{process_type}/{strategy_name}")
    async def stop_process(process_type: str, strategy_name: str):
        try:
            ptype = ProcessType(process_type)
        except ValueError:
            raise HTTPException(400, f"Invalid process type: {process_type}")
        state.add_log(f"[action] Stopping {process_type} for {strategy_name}")
        ok = proc_mgr.stop_process(ptype, strategy_name)
        return JSONResponse({"success": ok})

    # --- Workflow Control ---
    @app.post("/api/workflow/start/{strategy_name}")
    async def start_workflow(strategy_name: str):
        strat = _get_strat(strategy_name)
        state.add_log(f"[action] Starting workflow for {strategy_name}")
        ok = workflow.start(strat)
        return JSONResponse({"success": ok})

    @app.post("/api/workflow/cancel/{strategy_name}")
    async def cancel_workflow(strategy_name: str):
        state.add_log(f"[action] Cancelling workflow for {strategy_name}")
        workflow.cancel(strategy_name)
        return JSONResponse({"success": True})

    @app.post("/api/workflow/trigger/{strategy_name}")
    async def trigger_workflow(strategy_name: str):
        cfg = cfg_holder.get()
        proc_mgr.config = cfg
        state.add_log(f"[action] Triggering scheduled workflow for {strategy_name}")
        ok = scheduler.trigger_now(strategy_name)
        return JSONResponse({"success": ok})

    # --- Hyperopt Results ---
    @app.get("/api/hyperopt/files/{strategy_name}")
    async def list_fthypt_files(strategy_name: str):
        """List all .fthypt files for a strategy."""
        import glob as _glob
        cfg = cfg_holder.get()
        base_dir = os.path.join(cfg.freqtrade_dir, "user_data", "hyperopt_results")
        if not os.path.isdir(base_dir):
            return JSONResponse({"files": []})
        # Use the actual freqtrade strategy class name for the glob
        strat = cfg.get_strategy(strategy_name)
        glob_name = strat.strategy_name if strat else strategy_name
        pattern = os.path.join(base_dir, f"*{glob_name}*.fthypt")
        files = []
        for fpath in sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True):
            files.append({
                "name": os.path.basename(fpath),
                "size": os.path.getsize(fpath),
                "modified": os.path.getmtime(fpath),
            })
        return JSONResponse({"files": files})

    @app.get("/api/hyperopt/evaluate/{strategy_name}")
    async def evaluate_fthypt(strategy_name: str, file: str = ""):
        """Load a .fthypt file, parse all epochs, evaluate against current criteria."""
        from manager.hyperopt_monitor import parse_fthypt_file, evaluate_criteria
        cfg = cfg_holder.get()
        strat = cfg.get_strategy(strategy_name)
        if not strat:
            raise HTTPException(404, f"Strategy {strategy_name} not found")

        base_dir = os.path.join(cfg.freqtrade_dir, "user_data", "hyperopt_results")
        if file:
            fpath = os.path.join(base_dir, os.path.basename(file))  # prevent path traversal
        else:
            # Use .last_result.json
            lr = os.path.join(base_dir, ".last_result.json")
            fpath = None
            if os.path.isfile(lr):
                try:
                    import json as _json
                    with open(lr, "r") as f:
                        data = _json.loads(f.read())
                    fname = data.get("latest_hyperopt", "")
                    if fname:
                        fpath = os.path.join(base_dir, fname)
                except Exception:
                    pass
            if not fpath:
                return JSONResponse({"total": 0, "best": None, "file": ""})

        if not os.path.isfile(fpath):
            return JSONResponse({"total": 0, "best": None, "file": os.path.basename(fpath)})

        epochs = parse_fthypt_file(fpath)
        best = evaluate_criteria(epochs, strat.epoch_criteria)
        return JSONResponse({
            "total": len(epochs),
            "best": best.to_dict() if best else None,
            "file": os.path.basename(fpath),
            "epochs": [e.to_dict() for e in epochs],
            "criteria": [
                {"field": c.field, "operator": c.operator, "value": c.value, "sort": c.sort}
                for c in strat.epoch_criteria
            ],
        })

    @app.get("/api/hyperopt/epochs/{strategy_name}")
    async def get_epochs(strategy_name: str):
        epochs = state.hyperopt_epochs.get(strategy_name, [])
        best = state.hyperopt_best.get(strategy_name)
        return JSONResponse({
            "total": len(epochs),
            "best": best.to_dict() if best else None,
            "epochs": [e.to_dict() for e in epochs[-100:]],
        })

    @app.post("/api/hyperopt/extract/{strategy_name}/{epoch_num}")
    async def extract_epoch(strategy_name: str, epoch_num: int):
        strat = _get_strat(strategy_name)
        state.add_log(f"[action] Extracting epoch {epoch_num} for {strategy_name}")
        cmd = proc_mgr.build_hyperopt_show_cmd(strat, epoch_num)
        ok = proc_mgr.start_process(ProcessType.HYPEROPT_SHOW, strat, cmd=cmd)
        return JSONResponse({"success": ok})

    # --- Output/Logs ---
    @app.get("/api/output/{process_type}/{strategy_name}")
    async def get_output(process_type: str, strategy_name: str, tail: int = 100):
        try:
            ptype = ProcessType(process_type)
        except ValueError:
            raise HTTPException(400, f"Invalid process type: {process_type}")
        info = state.get_process(ptype, strategy_name)
        if not info:
            return JSONResponse({"lines": []})
        return JSONResponse({"lines": info.output_lines[-tail:]})

    @app.get("/api/logs")
    async def get_logs(tail: int = 200):
        return JSONResponse({"logs": state.log_buffer[-tail:]})

    # --- Config ---
    @app.get("/api/config")
    async def get_config():
        cfg = cfg_holder.get()
        return JSONResponse({
            "freqtrade_dir": cfg.freqtrade_dir,
            "venv_path": cfg.venv_path,
            "strategies": [
                {
                    "name": s.name,
                    "strategy_name": s.strategy_name,
                    "enabled": s.enabled,
                    "restart_mode": s.restart_mode,
                    "trade_config": s.trade_config,
                    "download_config": s.download_config,
                    "backtest_config": s.backtest_config,
                    "hyperopt_config": s.hyperopt_config,
                    "reload_client_config": s.reload_client_config,
                    "download_data": {"enabled": s.download_data.enabled},
                    "backtest": {"enabled": s.backtest.enabled},
                    "hyperopt": {"enabled": s.hyperopt.enabled},
                    "extract": {"enabled": s.extract.enabled, "on_new_best": s.extract.on_new_best},
                    "restart": {"enabled": s.restart.enabled},
                    "schedule": {
                        "enabled": s.schedule.enabled,
                        "cron": s.schedule.cron,
                        "interval_hours": s.schedule.interval_hours,
                        "cleanup_days": s.schedule.cleanup_days,
                    },
                    "epoch_criteria": [
                        {"field": c.field, "operator": c.operator, "value": c.value, "sort": c.sort}
                        for c in s.epoch_criteria
                    ],
                }
                for s in cfg.strategies
            ],
        })

    @app.post("/api/config/step/{strategy_name}/{step}/{enabled}")
    async def toggle_step(strategy_name: str, step: str, enabled: str):
        """Toggle a workflow step enabled/disabled and persist to config.yaml."""
        import yaml
        valid_steps = {"download_data", "backtest", "hyperopt", "extract", "restart"}
        if step not in valid_steps:
            raise HTTPException(400, f"Invalid step: {step}. Must be one of {valid_steps}")
        val = enabled.lower() in ("true", "1", "yes")

        # Read raw YAML, modify, write back
        try:
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f)

            for s_raw in raw.get("strategies", []):
                if s_raw.get("name") == strategy_name:
                    if step in s_raw:
                        if isinstance(s_raw[step], dict):
                            s_raw[step]["enabled"] = val
                        else:
                            s_raw[step] = {"enabled": val}
                    else:
                        s_raw[step] = {"enabled": val}
                    break
            else:
                raise HTTPException(404, f"Strategy {strategy_name} not found")

            with open(config_path, "w") as f:
                yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            state.add_log(f"[config] {strategy_name}.{step}.enabled = {val}")
            return JSONResponse({"success": True, "step": step, "enabled": val})
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return JSONResponse({"success": False, "error": str(e)})

    @app.post("/api/config/schedule/{strategy_name}/{enabled}")
    async def toggle_schedule(strategy_name: str, enabled: str):
        """Toggle schedule enabled/disabled for a strategy and persist to config.yaml."""
        import yaml
        val = enabled.lower() in ("true", "1", "yes")
        try:
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f)
            for s_raw in raw.get("strategies", []):
                if s_raw.get("name") == strategy_name:
                    if "schedule" not in s_raw or not isinstance(s_raw["schedule"], dict):
                        s_raw["schedule"] = {}
                    s_raw["schedule"]["enabled"] = val
                    break
            else:
                raise HTTPException(404, f"Strategy {strategy_name} not found")
            with open(config_path, "w") as f:
                yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            scheduler.reschedule()
            state.add_log(f"[config] {strategy_name}.schedule.enabled = {val}")
            return JSONResponse({"success": True, "enabled": val})
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update schedule config: {e}")
            return JSONResponse({"success": False, "error": str(e)})

    @app.post("/api/config/reload")
    async def reload_config():
        try:
            cfg = cfg_holder.get()
            proc_mgr.config = cfg
            scheduler.reschedule()
            state.add_log("[config] Configuration reloaded from disk (scheduler updated)")
            return JSONResponse({"success": True, "strategies": [s.name for s in cfg.strategies]})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)})

    # --- WebSocket ---
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        queue = state.register_ws()
        try:
            await ws.send_json({"type": "init", "data": state.get_full_state()})
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    await ws.send_text(msg)
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WebSocket error: {e}")
        finally:
            state.unregister_ws(queue)

    return app
