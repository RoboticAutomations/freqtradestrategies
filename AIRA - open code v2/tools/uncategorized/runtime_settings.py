"""Runtime settings loaded from system settings with sane fallbacks."""

import os
from pathlib import Path
from typing import Any

import asyncpg
from sqlalchemy import select

from src.config import settings
from src.models import SystemSetting, async_session_maker


DEFAULT_RUNTIME_SETTINGS: dict[str, str] = {
    "analytics_db_url": os.environ.get("ANALYTICS_DATABASE_URL", ""),
    "finance_db_host": os.environ.get("FINANCE_DB_HOST", os.environ.get("DB_HOST", "")),
    "finance_db_port": os.environ.get("FINANCE_DB_PORT", os.environ.get("DB_PORT", "5432")),
    "finance_db_user": os.environ.get("FINANCE_DB_USER", os.environ.get("DB_USER", "")),
    "finance_db_password": os.environ.get("FINANCE_DB_PASSWORD", os.environ.get("DB_PASSWORD", "")),
    "finance_db_name": os.environ.get("FINANCE_DB_NAME", (os.environ.get("ANALYTICS_DATABASE_URL", "").rsplit("/", 1)[-1] if os.environ.get("ANALYTICS_DATABASE_URL", "") else "")),
    "app_root_dir": "",
    "freqtrade_root_dir": "",
    "freqtrade_data_dir": "",
    "freqtrade_agent_root_dir": "",
    "fred_api_key": "",
    "newsapi_key": "",
}


def _stringify(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _join(base: str, *parts: str) -> str:
    return str(Path(base).joinpath(*parts))


async def get_runtime_settings() -> dict[str, str]:
    """Return runtime settings merged with stored overrides."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(SystemSetting).where(SystemSetting.key.in_(tuple(DEFAULT_RUNTIME_SETTINGS.keys())))
        )
        db_settings = {setting.key: setting.value for setting in result.scalars().all()}

    merged = DEFAULT_RUNTIME_SETTINGS.copy()
    merged.update({key: str(value) for key, value in db_settings.items() if value is not None})
    return merged


async def get_runtime_setting(key: str, default: str = "") -> str:
    """Return one runtime setting with fallback."""
    settings_map = await get_runtime_settings()
    fallback = DEFAULT_RUNTIME_SETTINGS.get(key, default)
    return _stringify(settings_map.get(key), fallback)


async def get_finance_db_config() -> dict[str, Any]:
    """Return asyncpg config for finance and agent modules."""
    runtime = await get_runtime_settings()
    return {
        "host": _stringify(runtime.get("finance_db_host"), DEFAULT_RUNTIME_SETTINGS["finance_db_host"]),
        "port": int(_stringify(runtime.get("finance_db_port"), DEFAULT_RUNTIME_SETTINGS["finance_db_port"])),
        "user": _stringify(runtime.get("finance_db_user"), DEFAULT_RUNTIME_SETTINGS["finance_db_user"]),
        "password": _stringify(
            runtime.get("finance_db_password"), DEFAULT_RUNTIME_SETTINGS["finance_db_password"]
        ),
        "database": _stringify(runtime.get("finance_db_name"), DEFAULT_RUNTIME_SETTINGS["finance_db_name"]),
    }


async def create_finance_db_pool() -> asyncpg.Pool:
    """Create an asyncpg pool for the finance database."""
    return await asyncpg.create_pool(**await get_finance_db_config())


async def get_runtime_paths() -> dict[str, str]:
    """Return resolved filesystem paths for strategy and freqtrade operations."""
    runtime = await get_runtime_settings()

    app_root_dir = _stringify(runtime.get("app_root_dir"), DEFAULT_RUNTIME_SETTINGS["app_root_dir"])
    freqtrade_root_dir = _stringify(
        runtime.get("freqtrade_root_dir"), DEFAULT_RUNTIME_SETTINGS["freqtrade_root_dir"]
    )
    freqtrade_data_dir = _stringify(
        runtime.get("freqtrade_data_dir"), DEFAULT_RUNTIME_SETTINGS["freqtrade_data_dir"]
    )
    freqtrade_agent_root_dir = _stringify(
        runtime.get("freqtrade_agent_root_dir"), DEFAULT_RUNTIME_SETTINGS["freqtrade_agent_root_dir"]
    )

    strategies_dir = _join(app_root_dir, "Strategies")
    if not Path(strategies_dir).exists():
        for fallback_dir in ["./Strategies", "/app/Strategies"]:
            if Path(fallback_dir).exists():
                strategies_dir = fallback_dir
                break

    return {
        "app_root_dir": app_root_dir,
        "scripts_dir": _join(app_root_dir, "scripts"),
        "strategies_dir": strategies_dir,
        "backtest_results_dir": _join(app_root_dir, "results", "backtest"),
        "hyperopt_results_dir": _join(app_root_dir, "results", "hyperopt"),
        "pairlist_results_dir": _join(app_root_dir, "results", "pairlists"),
        "freqtrade_root_dir": freqtrade_root_dir,
        "freqtrade_strategies_dir": _join(freqtrade_root_dir, "user_data", "strategies"),
        "freqtrade_config_dir": _join(freqtrade_root_dir, "user_data", "config"),
        "freqtrade_logs_dir": _join(freqtrade_root_dir, "user_data", "logs"),
        "freqtrade_data_dir": freqtrade_data_dir,
        "freqtrade_agent_root_dir": freqtrade_agent_root_dir,
        "freqtrade_agent_config_dir": _join(freqtrade_agent_root_dir, "user_data", "config"),
        "freqtrade_agent_strategies_dir": _join(freqtrade_agent_root_dir, "user_data", "strategies"),
        "freqtrade_agent_logs_dir": _join(freqtrade_agent_root_dir, "user_data", "logs"),
        "freqtrade_agent_backtest_dir": _join(
            freqtrade_agent_root_dir, "user_data", "backtest_results"
        ),
        "freqtrade_agent_hyperopt_dir": _join(
            freqtrade_agent_root_dir, "user_data", "hyperopt_results"
        ),
    }
