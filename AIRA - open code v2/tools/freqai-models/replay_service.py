"""Replay service for freqtrade-replay Docker integration.

Uses the SAME subprocess pattern as StrategyLab backtest:
  asyncio.create_subprocess_exec("docker", "run", "--rm", ...)

Spawns a Freqtrade container with freqtrade_replay mounted in user_data,
runs the replay CLI, parses stdout, stores results.
"""

import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.runtime_settings import get_runtime_paths

logger = logging.getLogger(__name__)


class ReplayService:
    """Service to manage freqtrade-replay Docker runs."""

    FREQTRADE_IMAGE = "freqtradeorg/freqtrade:stable_freqaitorch"

    def __init__(self, user_data_path: str = "/opt/freqtrade/user_data"):
        # Will be overridden at runtime from DB settings
        self.user_data_path = user_data_path
        # Logs/reports written to /app/logs (writable mount in backend container)
        self.logs_dir = "/app/logs/replay_logs"
        self.reports_dir = "/app/logs/replay_reports"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Docker command builder — inline bash (no file writes needed)
    # ------------------------------------------------------------------ #
    def build_docker_command(
        self,
        strategy_name: str,
        timerange: str,
        config_path: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        timeframe: Optional[str] = None,
    ) -> list[str]:
        """Build docker run command with entrypoint override for replay."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Config is already copied to user_data — use container path directly
        container_config = f"/freqtrade/user_data/{os.path.basename(config_path or 'config.json')}"
        
        # Build replay args — NO hardcoded datadir, data is mounted at /freqtrade/user_data/data
        replay_args = f"--strategy {strategy_name} --timerange {timerange} --config {container_config}"
        if extra_args:
            replay_args += " " + " ".join(extra_args)
        
        # Inline bash: install freezegun then run replay
        bash_cmd = f"pip install -q freezegun 2>/dev/null || true && python /freqtrade/user_data/freqtrade_replay/cli.py {replay_args}"
        
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name", f"replay_{strategy_name}_{timestamp}",
            "--entrypoint", "bash",
            "-v", f"{self.user_data_path}:/freqtrade/user_data",
        ]
        
        # Mount shared data folder as /freqtrade/user_data/data (transparent, no --datadir needed)
        if os.path.exists("/opt/Freqtrade_data/data"):
            cmd.extend(["-v", "/opt/Freqtrade_data/data:/freqtrade/user_data/data:rw"])
        
        cmd.extend([
            self.FREQTRADE_IMAGE,
            "-c", bash_cmd,
        ])
        return cmd
    
    def _get_datadir_from_config(self, config_path: str) -> str:
        """Read exchange name from config and build correct datadir path.
        
        Priority:
        1. /opt/Freqtrade_data/data/{exchange} (shared data folder for all bots)
        2. Fallback: /freqtrade/user_data/data/{exchange}/{mode}
        """
        # Try to read config file — it may be at container path or host path
        cfg = None
        for try_path in [config_path, config_path.replace("/freqtrade-host/", "/opt/AlexFreqAlphaDashboard/freqtrade-host/")]:
            try:
                with open(try_path, "r") as f:
                    cfg = json.load(f)
                break
            except Exception:
                continue
        
        if cfg is None:
            logger.warning("[replay] Could not read config at %s — falling back to binance/futures", config_path)
            return "/freqtrade/user_data/data/binance/futures"
        
        exchange_name = cfg.get("exchange", {}).get("name", "binance").lower()
        trading_mode = cfg.get("trading_mode", "futures")
        
        # Priority 1: Use shared data folder if it exists
        shared_data = f"/opt/Freqtrade_data/data/{exchange_name}"
        if os.path.exists(shared_data):
            logger.info("[replay] Using shared data: exchange=%s → datadir=%s", exchange_name, shared_data)
            return shared_data
        
        # Priority 2: Fallback to user_data/data
        datadir = f"/freqtrade/user_data/data/{exchange_name}/{trading_mode}"
        logger.info("[replay] Using user_data data: exchange=%s mode=%s → datadir=%s", exchange_name, trading_mode, datadir)
        return datadir

    def _find_config_path(self, timeframe: Optional[str] = None) -> str:
        """Find a suitable config.json in freqtrade-host/config/ or user_data.
        
        If timeframe='1h', prefer config-torch1h.json over config-torch.json.
        """
        import glob
        
        # Look for freqtrade-host in common locations (host paths)
        possible_host_dirs = [
            # Dashboard deployment with freqtrade-host (most common)
            "/opt/AlexFreqAlphaDashboard/freqtrade-host/config",
            os.path.join(os.path.dirname(os.path.dirname(self.user_data_path)), "freqtrade-host", "config"),
            os.path.join(self.user_data_path, "freqtrade-host", "config"),
            # Directly in user_data
            self.user_data_path,
        ]
        
        # Determine config preference based on timeframe
        if timeframe == "1h":
            preferred_configs = ["config-torch1h.json", "config-torch.json", "config.json"]
        else:
            preferred_configs = ["config-torch.json", "config-torch1h.json", "config.json"]
        
        for host_dir in possible_host_dirs:
            candidates = []
            for cfg_name in preferred_configs:
                candidates.append(os.path.join(host_dir, cfg_name))
            candidates.extend(sorted(glob.glob(os.path.join(host_dir, "config*.json"))))
            
            for host_path in candidates:
                if os.path.exists(host_path):
                    # Convert host path to container path
                    if "freqtrade-host" in host_path:
                        return "/freqtrade-host/config/" + os.path.basename(host_path)
                    else:
                        return host_path.replace(self.user_data_path, "/freqtrade/user_data")
        
        # Last resort — return default and let it fail clearly
        return "/freqtrade/user_data/config.json"

    # ------------------------------------------------------------------ #
    #  Async spawn + monitor  (SAME pattern as StrategyLab backtest)
    # ------------------------------------------------------------------ #
    async def run_replay(
        self,
        db: AsyncSession,
        replay_id: int,
        strategy_name: str,
        timerange: str,
        bot_id: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> dict:
        """Run a freqtrade-replay via docker subprocess and parse results."""

        logger.info(
            "[replay:%s] Starting — strategy=%s timerange=%s",
            replay_id, strategy_name, timerange,
        )

        # Resolve runtime paths from DB settings (same as StrategyLab)
        paths = await get_runtime_paths()
        
        # freqtrade_root_dir might be /opt (for scanner) or /opt/freqtrade (for user_data)
        # Always use /opt/freqtrade/user_data for replay file operations
        freqtrade_root = paths["freqtrade_root_dir"]
        if os.path.exists("/opt/freqtrade/user_data"):
            self.user_data_path = "/opt/freqtrade/user_data"
        elif os.path.exists(os.path.join(freqtrade_root, "freqtrade", "user_data")):
            self.user_data_path = os.path.join(freqtrade_root, "freqtrade", "user_data")
        elif os.path.exists(os.path.join(freqtrade_root, "user_data")):
            self.user_data_path = os.path.join(freqtrade_root, "user_data")
        else:
            self.user_data_path = "/opt/freqtrade/user_data"  # final fallback
        
        # Also fix strategies_dir if it's /opt/user_data/strategies → /opt/freqtrade/user_data/strategies
        freqtrade_strategies_dir = paths["freqtrade_strategies_dir"]
        if "freqtrade" not in freqtrade_strategies_dir:
            freqtrade_strategies_dir = self.user_data_path + "/strategies"
        
        strategies_dir = paths["strategies_dir"]
        
        logger.info("[replay:%s] Using user_data_path: %s", replay_id, self.user_data_path)
        logger.info("[replay:%s] Using strategies_dir: %s", replay_id, freqtrade_strategies_dir)

        # ── Copy strategy file to freqtrade strategies dir (same as StrategyLab) ──
        strategy_file_path = None
        for root, dirs, files in os.walk(strategies_dir):
            for file in files:
                if file == f"{strategy_name}.py":
                    strategy_file_path = os.path.join(root, file)
                    break
            if strategy_file_path:
                break

        if strategy_file_path:
            dest_path = os.path.join(freqtrade_strategies_dir, os.path.basename(strategy_file_path))
            try:
                os.makedirs(freqtrade_strategies_dir, exist_ok=True)
                shutil.copy(strategy_file_path, dest_path)
                logger.info("[replay:%s] Copied strategy %s → %s", replay_id, strategy_name, dest_path)
            except OSError as exc:
                logger.warning("[replay:%s] Could not copy strategy: %s", replay_id, exc)

        # ── Copy freqtrade_replay to user_data (required for replay to work) ──
        replay_src = None
        for possible_src in [
            os.path.join(os.path.dirname(self.user_data_path), "scripts", "freqtrade_replay"),
            os.path.join(os.path.dirname(os.path.dirname(self.user_data_path)), "scripts", "freqtrade_replay"),
            os.path.join(strategies_dir, "..", "..", "scripts", "freqtrade_replay"),
            "/opt/AlexFreqAlphaDashboard/scripts/freqtrade_replay",
            "/app/scripts/freqtrade_replay",
        ]:
            if os.path.exists(possible_src) and os.path.isdir(possible_src):
                replay_src = possible_src
                break
        
        if replay_src:
            replay_dest = os.path.join(self.user_data_path, "freqtrade_replay")
            try:
                if os.path.exists(replay_dest):
                    shutil.rmtree(replay_dest)
                shutil.copytree(replay_src, replay_dest)
                logger.info("[replay:%s] Copied freqtrade_replay %s → %s", replay_id, replay_src, replay_dest)
            except OSError as exc:
                logger.warning("[replay:%s] Could not copy freqtrade_replay: %s", replay_id, exc)
        else:
            logger.warning("[replay:%s] freqtrade_replay not found — replay will fail", replay_id)

        # ── Copy config to user_data so Docker can find it easily ──
        host_config_path = self._find_config_path(timeframe=timeframe)
        # host_config_path is a host path (e.g. /opt/AlexFreqAlphaDashboard/freqtrade-host/config/config-torch.json)
        # Copy to user_data so it's at /freqtrade/user_data/config-torch.json inside the container
        if host_config_path and os.path.exists(host_config_path):
            config_basename = os.path.basename(host_config_path)
            config_dest = os.path.join(self.user_data_path, config_basename)
            try:
                shutil.copy(host_config_path, config_dest)
                logger.info("[replay:%s] Copied config %s → %s", replay_id, host_config_path, config_dest)
            except OSError as exc:
                logger.warning("[replay:%s] Could not copy config: %s", replay_id, exc)
        else:
            logger.warning("[replay:%s] Config not found at %s", replay_id, host_config_path)

        # Update status
        await db.execute(
            text(
                """
                UPDATE replay_runs
                SET status = 'running', started_at = NOW()
                WHERE id = :id
                """
            ),
            {"id": replay_id},
        )
        await db.commit()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            # Find config on host and copy it to user_data first
            host_config_path = self._find_config_path(timeframe=timeframe)
            if host_config_path and os.path.exists(host_config_path):
                config_basename = os.path.basename(host_config_path)
                config_dest = os.path.join(self.user_data_path, config_basename)
                try:
                    shutil.copy(host_config_path, config_dest)
                    logger.info("[replay:%s] Copied config %s → %s", replay_id, host_config_path, config_dest)
                except OSError as exc:
                    logger.warning("[replay:%s] Could not copy config: %s", replay_id, exc)
            
            cmd = self.build_docker_command(
                strategy_name, timerange,
                config_path=config_dest if 'config_dest' in dir() else None,
                timeframe=timeframe,
            )
            logger.info("[replay:%s] Docker cmd: %s", replay_id, " ".join(cmd))

            # ── SAME pattern as StrategyLab ──────────────────────────────
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await process.communicate()

            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            stdout_lines = stdout.splitlines()
            stderr_lines = stderr.splitlines()

            if process.returncode != 0:
                error_msg = f"Exit code {process.returncode}\nSTDERR:\n{stderr[:2000]}"
                raise RuntimeError(error_msg)

            # Parse stdout for metrics
            full_stdout = "\n".join(stdout_lines)
            parsed = self.parse_replay_output(full_stdout)

            # Save raw log (writable path inside backend container)
            log_path = f"{self.logs_dir}/replay_{replay_id}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
                f.write(f"\n=== RETURN CODE ===\n{process.returncode}")

            # Generate HTML report if we have trades
            report_path = None
            total_trades = parsed.get("total_trades") or 0
            if total_trades > 0:
                report_path = await self._generate_report(
                    replay_id, strategy_name, timerange, parsed, full_stdout
                )

            # Update DB with results
            await db.execute(
                text(
                    """
                    UPDATE replay_runs SET
                        status = 'completed',
                        completed_at = NOW(),
                        profit_pct = :profit_pct,
                        profit_abs = :profit_abs,
                        win_rate = :win_rate,
                        total_trades = :total_trades,
                        max_drawdown_pct = :max_drawdown_pct,
                        sharpe = :sharpe,
                        profit_factor = :profit_factor,
                        docker_log_path = :log_path,
                        report_html_path = :report_path,
                        report_json = :report_json,
                        error_message = NULL
                    WHERE id = :id
                    """
                ),
                {
                    "profit_pct": parsed.get("profit_pct"),
                    "profit_abs": parsed.get("profit_abs"),
                    "win_rate": parsed.get("win_rate"),
                    "total_trades": parsed.get("total_trades"),
                    "max_drawdown_pct": parsed.get("max_drawdown_pct"),
                    "sharpe": parsed.get("sharpe"),
                    "profit_factor": parsed.get("profit_factor"),
                    "log_path": log_path,
                    "report_path": report_path,
                    "report_json": json.dumps(parsed) if parsed else None,
                    "id": replay_id,
                },
            )
            await db.commit()

            logger.info(
                "[replay:%s] Completed — profit=%s%% trades=%s",
                replay_id,
                parsed.get("profit_pct"),
                parsed.get("total_trades"),
            )
            return {"status": "completed", **parsed}

        except Exception as exc:
            error_str = str(exc)
            if stderr_lines:
                error_str += f"\nLast stderr lines:\n" + "\n".join(stderr_lines[-20:])

            logger.error("[replay:%s] Failed — %s", replay_id, error_str[:500])

            await db.execute(
                text(
                    """
                    UPDATE replay_runs
                    SET status = 'failed',
                        completed_at = NOW(),
                        error_message = :error
                    WHERE id = :id
                    """
                ),
                {"error": error_str[:4000], "id": replay_id},
            )
            await db.commit()
            return {"status": "failed", "error": error_str}

    # ------------------------------------------------------------------ #
    #  Output parser
    # ------------------------------------------------------------------ #
    def parse_replay_output(self, stdout: str) -> dict:
        """Parse freqtrade-replay stdout for key metrics."""
        result: dict[str, Optional[float]] = {
            "profit_pct": None,
            "profit_abs": None,
            "win_rate": None,
            "total_trades": None,
            "max_drawdown_pct": None,
            "sharpe": None,
            "profit_factor": None,
        }

        for line in stdout.splitlines():
            line = line.strip()

            # Total P&L: +183.24 USDT  →  profit_abs
            m = re.search(
                r"Total\s+P&L\s*[:=]?\s*([+-]?[\d,.]+)",
                line, re.IGNORECASE,
            )
            if m:
                val = m.group(1).replace(",", "")
                try:
                    result["profit_abs"] = float(val)
                except ValueError:
                    pass

            # Closed trades: 42  →  total_trades
            m = re.search(
                r"Closed\s+trades\s*[:=]?\s*(\d+)",
                line, re.IGNORECASE,
            )
            if m:
                result["total_trades"] = int(m.group(1))

            # Win rate : 28/42 (66.7%)  →  win_rate
            m = re.search(
                r"Win\s+rate\s*[:=]?\s*\d+/\d+\s*\((\d+\.?\d*)%\)",
                line, re.IGNORECASE,
            )
            if m:
                result["win_rate"] = float(m.group(1)) / 100.0

            # Max drawdown: 12.34%  →  max_drawdown_pct
            m = re.search(
                r"Max\s+drawdown\s*[:=]?\s*([\d.]+)%",
                line, re.IGNORECASE,
            )
            if m:
                result["max_drawdown_pct"] = float(m.group(1))

            # Sharpe: 1.23  →  sharpe
            m = re.search(
                r"Sharpe\s*[:=]?\s*([+-]?[\d.]+)",
                line, re.IGNORECASE,
            )
            if m:
                result["sharpe"] = float(m.group(1))

            # Profit factor: 1.45  →  profit_factor
            m = re.search(
                r"Profit\s+factor\s*[:=]?\s*([\d.]+)",
                line, re.IGNORECASE,
            )
            if m:
                result["profit_factor"] = float(m.group(1))

            # Profit %: +15.3%  →  profit_pct
            m = re.search(
                r"Profit\s*%\s*[:=]?\s*([+-]?[\d.]+)%",
                line, re.IGNORECASE,
            )
            if m:
                result["profit_pct"] = float(m.group(1))

        return result

    # ------------------------------------------------------------------ #
    #  HTML report generator
    # ------------------------------------------------------------------ #
    async def _generate_report(
        self,
        replay_id: int,
        strategy_name: str,
        timerange: str,
        metrics: dict,
        raw_stdout: str,
    ) -> str:
        """Generate a simple HTML report for the replay run."""
        reports_dir = self.reports_dir
        os.makedirs(reports_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{reports_dir}/replay_{replay_id}_{strategy_name}_{timestamp}.html"

        profit_pct = metrics.get("profit_pct")
        win_rate = metrics.get("win_rate")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Replay Report — {strategy_name}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
    h1 {{ color: #38bdf8; }}
    .metric {{ display: inline-block; padding: 1rem 1.5rem; margin: 0.5rem; border-radius: 0.5rem; background: #1e293b; }}
    .metric label {{ display: block; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }}
    .metric value {{ display: block; font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }}
    .positive {{ color: #22c55e; }} .negative {{ color: #ef4444; }}
    pre {{ background: #1e293b; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <h1>🔄 Replay Report</h1>
  <p>Strategy: <strong>{strategy_name}</strong> | Timerange: <strong>{timerange}</strong></p>
  <div>
    <div class="metric">
      <label>Trades</label>
      <value>{metrics.get('total_trades', 'N/A')}</value>
    </div>
    <div class="metric">
      <label>Win Rate</label>
      <value class="{'positive' if (win_rate or 0) > 0.5 else 'negative'}">
        {(win_rate or 0) * 100:.1f}%
      </value>
    </div>
    <div class="metric">
      <label>Profit</label>
      <value class="{'positive' if (profit_pct or 0) >= 0 else 'negative'}">
        {metrics.get('profit_pct', 'N/A'):.2f}%
      </value>
    </div>
    <div class="metric">
      <label>Drawdown</label>
      <value>{metrics.get('max_drawdown_pct', 'N/A'):.2f}%</value>
    </div>
    <div class="metric">
      <label>Sharpe</label>
      <value>{metrics.get('sharpe', 'N/A')}</value>
    </div>
  </div>
  <h2>Raw Output</h2>
  <pre>{raw_stdout[-3000:]}</pre>
</body>
</html>"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        return report_path


# ------------------------------------------------------------------ #
#  Singleton factory
# ------------------------------------------------------------------ #
_replay_service: Optional[ReplayService] = None


def get_replay_service() -> ReplayService:
    global _replay_service
    if _replay_service is None:
        _replay_service = ReplayService()
    return _replay_service
