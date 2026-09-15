"""
Docker Controller for Agent - Multibotdashboard V8
Uses same pattern as StrategyLab (docker run, not compose)
"""

import subprocess
import os
import logging
import time
from typing import Optional, Tuple

from src.services.runtime_settings import DEFAULT_RUNTIME_SETTINGS

logger = logging.getLogger(__name__)

# Agent paths
AGENT_PATH = os.environ.get(
    "FREQTRADE_AGENT_ROOT_DIR", DEFAULT_RUNTIME_SETTINGS["freqtrade_agent_root_dir"]
)
FREQTRADE_DATA_DIR = os.environ.get(
    "FREQTRADE_DATA_DIR", DEFAULT_RUNTIME_SETTINGS["freqtrade_data_dir"]
)
AGENT_IMAGE = "freqtradeorg/freqtrade:stable"


class AgentDockerController:
    """Controls the Freqtrade Agent Docker container - same pattern as StrategyLab"""
    
    @staticmethod
    def is_container_running() -> bool:
        """Check if freqtrade-agent container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", "name=freqtrade-agent"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.error(f"Failed to check container status: {e}")
            return False
    
    @staticmethod
    def start_container() -> Tuple[bool, str]:
        """Start the Freqtrade Agent container using docker run (like StrategyLab)"""
        try:
            # First check if already running
            if AgentDockerController.is_container_running():
                return True, "Container already running"
            
            # Build docker run command (same pattern as StrategyLab)
            cmd = [
                "docker", "run", "-d",
                "--name", "freqtrade-agent",
                "--restart", "unless-stopped",
                "-p", "8060:8060",
                "-v", f"{AGENT_PATH}/user_data/config:/freqtrade/user_data/config",
                "-v", f"{AGENT_PATH}/user_data/strategies:/freqtrade/user_data/strategies",
                "-v", f"{FREQTRADE_DATA_DIR}:/freqtrade/user_data/data",
                "-v", f"{AGENT_PATH}/user_data/logs:/freqtrade/user_data/logs",
                "-v", f"{AGENT_PATH}/user_data/backtest_results:/freqtrade/user_data/backtest_results",
                "-v", f"{AGENT_PATH}/user_data/hyperopt_results:/freqtrade/user_data/hyperopt_results",
                "-e", "TZ=Europe/Vienna",
                AGENT_IMAGE,
                "trade",
                "--db-url", "sqlite:////freqtrade/user_data/tradesv3.sqlite",
                "--strategy", "Alex_AgentStrategy",
                "--config", "/freqtrade/user_data/config/config-torch.json"
            ]
            
            logger.info(f"Starting Agent container: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                logger.info(f"Agent container started: {container_id[:12]}")
                return True, f"Container started: {container_id[:12]}"
            else:
                logger.error(f"Failed to start container: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Timeout starting container"
        except Exception as e:
            logger.error(f"Exception starting container: {e}")
            return False, str(e)
    
    @staticmethod
    def stop_container() -> Tuple[bool, str]:
        """Stop and remove the Freqtrade Agent container"""
        try:
            # Stop container
            stop_result = subprocess.run(
                ["docker", "stop", "freqtrade-agent"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Remove container
            rm_result = subprocess.run(
                ["docker", "rm", "freqtrade-agent"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if stop_result.returncode == 0 or rm_result.returncode == 0:
                logger.info("Agent container stopped and removed")
                return True, "Container stopped"
            else:
                # Container might not exist
                if "No such container" in stop_result.stderr:
                    return True, "Container not running"
                return False, stop_result.stderr or rm_result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Timeout stopping container"
        except Exception as e:
            logger.error(f"Exception stopping container: {e}")
            return False, str(e)
    
    @staticmethod
    def get_container_logs(lines: int = 50) -> str:
        """Get recent container logs"""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), "freqtrade-agent"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error getting logs: {e}"
    
    @staticmethod
    def restart_container() -> Tuple[bool, str]:
        """Restart the Freqtrade Agent container"""
        stop_success, stop_msg = AgentDockerController.stop_container()
        if not stop_success:
            return False, f"Failed to stop: {stop_msg}"
        
        # Wait a moment
        time.sleep(2)
        
        return AgentDockerController.start_container()
