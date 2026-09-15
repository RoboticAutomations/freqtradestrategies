"""
FreqTrade Manager integration for Multibotdashboard V6
Automated strategy optimization and workflow management
"""

from .config import AppConfig, StrategyConfig
from .state import AppState, ProcessType, ProcessStatus, ProcessInfo
from .workflow import Workflow
from .hyperopt_monitor import HyperoptMonitor

__all__ = [
    "AppConfig",
    "StrategyConfig", 
    "AppState",
    "ProcessType",
    "ProcessStatus",
    "ProcessInfo",
    "Workflow",
    "WorkflowStep",
    "HyperoptMonitor",
]
