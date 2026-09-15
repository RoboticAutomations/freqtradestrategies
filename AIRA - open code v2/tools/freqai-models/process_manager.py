"""
Process manager for FreqTrade - Docker/Linux adaptation for Multibotdashboard V6
Replaces Windows-specific code with Docker container management
"""

import os
import subprocess
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional
import docker

from .config import AppConfig, StrategyConfig
from .state import AppState, ProcessInfo, ProcessStats, ProcessType, ProcessStatus

logger = logging.getLogger(__name__)


def calc_timerange(start_days_ago: int, end_days_ago: int) -> str:
    """Calculate freqtrade timerange string from days-ago values."""
    start = (datetime.now() - timedelta(days=start_days_ago)).strftime("%Y%m%d")
    if end_days_ago == 0:
        return f"{start}-"
    end = (datetime.now() - timedelta(days=end_days_ago)).strftime("%Y%m%d")
    return f"{start}-{end}"


class DockerProcessManager:
    """Manages freqtrade processes using Docker containers."""
    
    def __init__(self, config: AppConfig, state: AppState):
        self.config = config
        self.state = state
        self.docker_client = docker.from_env()
        self._lock = threading.Lock()
        
    def _get_container_name(self, strategy_name: str, process_type: ProcessType) -> str:
        """Generate unique container name for a process."""
        return f"ft_{strategy_name.lower()}_{process_type.value}_{int(time.time())}"
    
    def _build_docker_cmd(
        self,
        process_type: ProcessType,
        strategy: StrategyConfig,
        extra_args: list = None
    ) -> list:
        """Build Docker command for a process type."""
        
        base_cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.config.freqtrade.directory}/user_data:/freqtrade/user_data",
            "-v", f"{self.config.freqtrade.directory}/config:/freqtrade/config",
            self.config.freqtrade.docker_image or "freqtradeorg/freqtrade:stable_freqaitorch"
        ]
        
        if process_type == ProcessType.TRADE:
            cmd = base_cmd + [
                "trade",
                "--config", strategy.config_path,
                "--strategy", strategy.name
            ]
        elif process_type == ProcessType.BACKTEST:
            timerange = calc_timerange(
                strategy.backtest_timerange_start_days_ago,
                strategy.backtest_timerange_end_days_ago
            )
            cmd = base_cmd + [
                "backtesting",
                "--config", strategy.config_path,
                "--strategy", strategy.name,
                "--timerange", timerange,
                "--timeframe", strategy.timeframe or "15m",
                "--cache", "none"
            ]
        elif process_type == ProcessType.HYPEROPT:
            timerange = calc_timerange(
                strategy.hyperopt_timerange_start_days_ago,
                strategy.hyperopt_timerange_end_days_ago
            )
            cmd = base_cmd + [
                "hyperopt",
                "--config", strategy.config_path,
                "--strategy", strategy.name,
                "--timerange", timerange,
                "--timeframe", strategy.timeframe or "15m",
                "--epochs", str(strategy.epochs or 30),
                "--spaces", strategy.hyperopt_spaces or "buy sell",
                "--hyperopt-loss", strategy.hyperopt_loss or "SharpeHyperOptLoss",
                "-j", "4"  # 4 parallel jobs
            ]
        elif process_type == ProcessType.DOWNLOAD_DATA:
            cmd = base_cmd + [
                "download-data",
                "--config", strategy.config_path,
                "--timeframe", strategy.timeframe or "15m",
                "--days", "30"
            ]
        else:
            raise ValueError(f"Unknown process type: {process_type}")
        
        if extra_args:
            cmd.extend(extra_args)
            
        return cmd
    
    def start_process(
        self,
        process_id: str,
        process_type: ProcessType,
        strategy: StrategyConfig,
        on_output: Callable[[str], None] = None,
        on_complete: Callable[[int], None] = None
    ) -> ProcessInfo:
        """Start a Docker container for a process."""
        
        with self._lock:
            # Check if already running
            existing = self.state.get_process(process_id)
            if existing and existing.status == ProcessStatus.RUNNING:
                logger.warning(f"Process {process_id} already running")
                return existing
            
            # Build command
            cmd = self._build_docker_cmd(process_type, strategy)
            container_name = self._get_container_name(strategy.name, process_type)
            
            logger.info(f"Starting {process_type.value} for {strategy.name}: {' '.join(cmd)}")
            
            try:
                # Start container
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Create process info
                info = ProcessInfo(
                    id=process_id,
                    type=process_type,
                    strategy_name=strategy.name,
                    pid=process.pid,
                    status=ProcessStatus.RUNNING,
                    started_at=datetime.now(),
                    container_name=container_name
                )
                self.state.register_process(info)
                
                # Start output reader thread
                if on_output:
                    threading.Thread(
                        target=self._read_output,
                        args=(process, process_id, on_output, on_complete),
                        daemon=True
                    ).start()
                else:
                    threading.Thread(
                        target=self._wait_process,
                        args=(process, process_id, on_complete),
                        daemon=True
                    ).start()
                
                return info
                
            except Exception as e:
                logger.error(f"Failed to start process {process_id}: {e}")
                raise
    
    def _read_output(
        self,
        process: subprocess.Popen,
        process_id: str,
        on_output: Callable[[str], None],
        on_complete: Callable[[int], None] = None
    ):
        """Read output from process and call callback."""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    on_output(line.rstrip())
                    
            process.wait()
            return_code = process.returncode
            
        except Exception as e:
            logger.error(f"Error reading output for {process_id}: {e}")
            return_code = -1
            
        finally:
            self.state.update_process_status(process_id, ProcessStatus.COMPLETED if return_code == 0 else ProcessStatus.ERROR)
            if on_complete:
                on_complete(return_code)
    
    def _wait_process(
        self,
        process: subprocess.Popen,
        process_id: str,
        on_complete: Callable[[int], None] = None
    ):
        """Wait for process completion without reading output."""
        try:
            return_code = process.wait()
            self.state.update_process_status(
                process_id,
                ProcessStatus.COMPLETED if return_code == 0 else ProcessStatus.ERROR
            )
            if on_complete:
                on_complete(return_code)
        except Exception as e:
            logger.error(f"Error waiting for process {process_id}: {e}")
            self.state.update_process_status(process_id, ProcessStatus.ERROR)
            if on_complete:
                on_complete(-1)
    
    def stop_process(self, process_id: str, timeout: int = 60) -> bool:
        """Stop a running process (Docker container)."""
        with self._lock:
            info = self.state.get_process(process_id)
            if not info or info.status != ProcessStatus.RUNNING:
                return False
            
            try:
                # Try graceful stop first
                if info.container_name:
                    subprocess.run(
                        ["docker", "stop", "-t", str(timeout), info.container_name],
                        capture_output=True,
                        timeout=timeout + 10
                    )
                
                # Update state
                self.state.update_process_status(process_id, ProcessStatus.STOPPED)
                logger.info(f"Stopped process {process_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop process {process_id}: {e}")
                # Force kill as fallback
                try:
                    if info.container_name:
                        subprocess.run(
                            ["docker", "kill", info.container_name],
                            capture_output=True,
                            timeout=10
                        )
                except:
                    pass
                return False
    
    def get_process_stats(self, process_id: str) -> Optional[ProcessStats]:
        """Get current stats for a running process."""
        info = self.state.get_process(process_id)
        if not info or info.status != ProcessStatus.RUNNING:
            return None
        
        try:
            # Get container stats via docker
            if info.container_name:
                result = subprocess.run(
                    ["docker", "stats", info.container_name, "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    cpu_str = parts[0].replace('%', '').strip()
                    mem_str = parts[1].split('/')[0].strip() if len(parts) > 1 else "0MiB"
                    
                    # Parse CPU
                    try:
                        cpu = float(cpu_str)
                    except:
                        cpu = 0.0
                    
                    # Parse memory (convert to MB)
                    mem_mb = 0
                    if 'GiB' in mem_str:
                        mem_mb = float(mem_str.replace('GiB', '')) * 1024
                    elif 'MiB' in mem_str:
                        mem_mb = float(mem_str.replace('MiB', ''))
                    elif 'GB' in mem_str:
                        mem_mb = float(mem_str.replace('GB', '')) * 1024
                    elif 'MB' in mem_str:
                        mem_mb = float(mem_str.replace('MB', ''))
                    
                    return ProcessStats(
                        cpu_percent=cpu,
                        memory_mb=mem_mb,
                        timestamp=datetime.now()
                    )
        except Exception as e:
            logger.debug(f"Could not get stats for {process_id}: {e}")
        
        return None
    
    def cleanup_old_containers(self, strategy_name: str = None):
        """Clean up old/stopped freqtrade containers."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "ancestor=freqtradeorg/freqtrade", "--format", "{{.ID}} {{.Status}} {{.Names}}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    container_id = parts[0]
                    status = ' '.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                    
                    # Remove exited containers
                    if 'Exited' in status:
                        subprocess.run(
                            ["docker", "rm", container_id],
                            capture_output=True
                        )
                        logger.info(f"Cleaned up old container {container_id}")
                        
        except Exception as e:
            logger.error(f"Container cleanup failed: {e}")


# Backward compatibility - alias for old code
ProcessManager = DockerProcessManager
