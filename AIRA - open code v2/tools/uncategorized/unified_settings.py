"""Unified Settings API"""
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db, SystemSetting

import os

router = APIRouter(prefix="/settings")

DEFAULT_SETTINGS = {
    "refresh_interval": "30",
    "theme": "dark",
    "language": "en",
    "notifications_enabled": "true",
    "dashboard_layout": "grid",
    "alert_critical": "true",
    "alert_warning": "true",
    "alert_info": "false",
    "discovery_host_ip": "",
    "api_username": "",
    "api_password": "",
    "install_wizard_completed": "false",
    "analytics_db_url": "",
    "finance_db_host": "",
    "finance_db_port": "5432",
    "finance_db_user": "",
    "finance_db_password": "",
    "finance_db_name": "",
    "app_root_dir": "",
    "freqtrade_root_dir": "",
    "freqtrade_data_dir": "",
    "freqtrade_agent_root_dir": "",
    "fred_api_key": "",
    "newsapi_key": "",
    "backtest_url": "",
    "backtest_name": "",
    "backtest_description": "",
}

REQUIRED_SETUP_KEYS = ("discovery_host_ip", "api_username", "api_password")

# Fallback to .env for finance DB settings
ENV_FALLBACKS = {
    "discovery_host_ip": os.environ.get("PUBLIC_HOST", ""),
    "api_username": os.environ.get("FREQTRADE_USERNAME", ""),
    "api_password": os.environ.get("FREQTRADE_PASSWORD", ""),
    "finance_db_host": os.environ.get("FINANCE_DB_HOST", ""),
    "finance_db_port": os.environ.get("FINANCE_DB_PORT", "5432"),
    "finance_db_user": os.environ.get("FINANCE_DB_USER", ""),
    "finance_db_password": os.environ.get("FINANCE_DB_PASSWORD", ""),
    "finance_db_name": os.environ.get("FINANCE_DB_NAME", ""),
}

@router.get("")
async def get_all_settings(session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(SystemSetting))
    db_settings = {s.key: s.value for s in result.scalars().all()}
    settings = DEFAULT_SETTINGS.copy()
    settings.update(db_settings)
    
    # Apply .env fallbacks for empty values
    for key, env_value in ENV_FALLBACKS.items():
        if not str(settings.get(key, "")).strip() and env_value.strip():
            settings[key] = env_value
    
    return {"settings": settings}


@router.get("/setup-status")
async def get_setup_status(session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(SystemSetting))
    db_settings = {s.key: s.value for s in result.scalars().all()}
    
    # Check if wizard was already completed
    wizard_completed = str(db_settings.get("install_wizard_completed", "false")).lower() == "true"
    
    settings = DEFAULT_SETTINGS.copy()
    settings.update(db_settings)
    
    # Apply .env fallbacks for empty values (so setup wizard is pre-populated)
    for key, env_value in ENV_FALLBACKS.items():
        if not str(settings.get(key, "")).strip() and env_value.strip():
            settings[key] = env_value

    missing_keys = [key for key in REQUIRED_SETUP_KEYS if not str(settings.get(key, "")).strip()]
    needs_setup = bool(missing_keys) or not wizard_completed

    return {
        "status": "success",
        "data": {
            "needs_setup": needs_setup,
            "wizard_completed": wizard_completed,
            "missing_keys": missing_keys,
            "settings": settings,
        },
    }

@router.post("/batch")
async def update_settings(settings: Dict[str, Any], session: AsyncSession = Depends(get_db)):
    for key, value in settings.items():
        result = await session.execute(select(SystemSetting).where(SystemSetting.key == key))
        setting = result.scalar_one_or_none()
        if setting:
            setting.value = str(value)
        else:
            session.add(SystemSetting(key=key, value=str(value), description=f"Setting: {key}"))
    await session.commit()
    return {"updated": list(settings.keys())}
