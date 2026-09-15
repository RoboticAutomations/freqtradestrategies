"""Configuration management for FreqTrade Manager."""

import os
import yaml
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EpochCriterion:
    field: str
    operator: str
    value: float
    sort: str = ""


@dataclass
class DownloadDataConfig:
    enabled: bool = True
    timeframes: list[str] = field(default_factory=lambda: ["3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "1d"])
    days_back: int = 90
    extra_args: str = ""


@dataclass
class BacktestConfig:
    enabled: bool = True
    timerange_start_days_ago: int = 30
    timerange_end_days_ago: int = 0
    extra_args: str = ""


@dataclass
class HyperoptConfig:
    enabled: bool = True
    timerange_start_days_ago: int = 30
    timerange_end_days_ago: int = 0
    loss_function: str = "SharpeHyperOptLossDaily"
    spaces: str = "sell"
    epochs: int = 2000
    jobs: int = 2
    min_trades: int = 10
    timeframe_detail: str = "3m"
    disable_param_export: bool = True
    extra_args: str = ""
    timeout_minutes: int = 360


@dataclass
class ExtractConfig:
    enabled: bool = True
    on_new_best: bool = False  # False=extract once after hyperopt ends, True=extract on each new best live


@dataclass
class RestartConfig:
    enabled: bool = True


@dataclass
class ScheduleConfig:
    enabled: bool = True
    cron: str = "0 2 * * *"
    interval_hours: float = 0
    cleanup_days: int = 2  # Delete fthypt files older than N days on workflow start (0=disabled)


@dataclass
class StrategyConfig:
    name: str                                          # Display name / identifier
    strategy_name: str = ""                            # Actual --strategy param
    enabled: bool = True

    # Decoupled config files (relative to freqtrade dir)
    trade_config: str = "config.json"
    download_config: str = "config.json"
    backtest_config: str = "config.json"
    hyperopt_config: str = "config.json"

    # freqtrade-client reload config (relative to manager folder)
    reload_client_config: str = ""

    # "hard" = stop+start, "graceful" = freqtrade-client reload_config
    restart_mode: str = "hard"

    download_data: DownloadDataConfig = field(default_factory=DownloadDataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    restart: RestartConfig = field(default_factory=RestartConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    epoch_criteria: list[EpochCriterion] = field(default_factory=list)
    trade_extra_args: str = ""

    def __post_init__(self):
        if not self.strategy_name:
            self.strategy_name = self.name


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 42000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "freqtrade_manager.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5


@dataclass
class MonitoringConfig:
    poll_interval: int = 3
    max_errors: int = 10


@dataclass
class AppConfig:
    freqtrade_dir: str = ""
    venv_path: str = ".venv312"
    strategies: list[StrategyConfig] = field(default_factory=list)
    web: WebConfig = field(default_factory=WebConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    process_stats_interval: int = 2

    # Set at runtime by main.py
    manager_dir: str = ""

    @property
    def freqtrade_exe(self) -> str:
        return os.path.join(self.freqtrade_dir, self.venv_path, "Scripts", "freqtrade.exe")

    @property
    def python_exe(self) -> str:
        return os.path.join(self.freqtrade_dir, self.venv_path, "Scripts", "python.exe")

    def get_strategy(self, name: str) -> StrategyConfig | None:
        for s in self.strategies:
            if s.name == name:
                return s
        return None


def load_config(path: str) -> AppConfig:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig()
    cfg.freqtrade_dir = raw.get("freqtrade", {}).get("directory", "")
    cfg.venv_path = raw.get("freqtrade", {}).get("venv_path", ".venv312")
    cfg.process_stats_interval = raw.get("process_stats_interval", 2)
    cfg.manager_dir = os.path.dirname(os.path.abspath(path))

    # Web
    web_raw = raw.get("web", {})
    cfg.web = WebConfig(
        host=web_raw.get("host", "0.0.0.0"),
        port=web_raw.get("port", 42000),
    )

    # Logging
    log_raw = raw.get("logging", {})
    cfg.logging = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        file=log_raw.get("file", "freqtrade_manager.log"),
        max_bytes=log_raw.get("max_bytes", 10_485_760),
        backup_count=log_raw.get("backup_count", 5),
    )

    # Monitoring
    mon_raw = raw.get("monitoring", {})
    cfg.monitoring = MonitoringConfig(
        poll_interval=mon_raw.get("poll_interval", 3),
        max_errors=mon_raw.get("max_errors", 10),
    )

    # Strategies
    for s_raw in raw.get("strategies", []):
        dl_raw = s_raw.get("download_data", {})
        bt_raw = s_raw.get("backtest", {})
        ho_raw = s_raw.get("hyperopt", {})
        ex_raw = s_raw.get("extract", {})
        rs_raw = s_raw.get("restart", {})
        sc_raw = s_raw.get("schedule", {})

        criteria = []
        for c in s_raw.get("epoch_criteria", []):
            criteria.append(EpochCriterion(
                field=c["field"],
                operator=c["operator"],
                value=float(c["value"]),
                sort=c.get("sort", ""),
            ))

        strat = StrategyConfig(
            name=s_raw["name"],
            strategy_name=s_raw.get("strategy_name", s_raw["name"]),
            enabled=s_raw.get("enabled", True),
            trade_config=s_raw.get("trade_config", "config.json"),
            download_config=s_raw.get("download_config", "config.json"),
            backtest_config=s_raw.get("backtest_config", "config.json"),
            hyperopt_config=s_raw.get("hyperopt_config", "config.json"),
            reload_client_config=s_raw.get("reload_client_config", ""),
            restart_mode=s_raw.get("restart_mode", "hard"),
            download_data=DownloadDataConfig(
                enabled=dl_raw.get("enabled", True),
                timeframes=dl_raw.get("timeframes", ["3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "1d"]),
                days_back=dl_raw.get("days_back", 90),
                extra_args=dl_raw.get("extra_args", ""),
            ),
            backtest=BacktestConfig(
                enabled=bt_raw.get("enabled", True),
                timerange_start_days_ago=bt_raw.get("timerange_start_days_ago", 30),
                timerange_end_days_ago=bt_raw.get("timerange_end_days_ago", 0),
                extra_args=bt_raw.get("extra_args", ""),
            ),
            hyperopt=HyperoptConfig(
                enabled=ho_raw.get("enabled", True),
                timerange_start_days_ago=ho_raw.get("timerange_start_days_ago", 30),
                timerange_end_days_ago=ho_raw.get("timerange_end_days_ago", 0),
                loss_function=ho_raw.get("loss_function", "SharpeHyperOptLossDaily"),
                spaces=ho_raw.get("spaces", "sell"),
                epochs=ho_raw.get("epochs", 2000),
                jobs=ho_raw.get("jobs", 2),
                min_trades=ho_raw.get("min_trades", 10),
                timeframe_detail=ho_raw.get("timeframe_detail", "3m"),
                disable_param_export=ho_raw.get("disable_param_export", True),
                extra_args=ho_raw.get("extra_args", ""),
                timeout_minutes=ho_raw.get("timeout_minutes", 360),
            ),
            extract=ExtractConfig(
                enabled=ex_raw.get("enabled", True),
                on_new_best=ex_raw.get("on_new_best", False),
            ),
            restart=RestartConfig(
                enabled=rs_raw.get("enabled", True),
            ),
            schedule=ScheduleConfig(
                enabled=sc_raw.get("enabled", True),
                cron=sc_raw.get("cron", "0 2 * * *"),
                interval_hours=sc_raw.get("interval_hours", 0),
                cleanup_days=sc_raw.get("cleanup_days", 2),
            ),
            epoch_criteria=criteria,
            trade_extra_args=s_raw.get("trade_extra_args", ""),
        )
        cfg.strategies.append(strat)

    return cfg
