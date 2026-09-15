"""
Settings API for system configuration
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

from src.models import get_db, SystemSetting

router = APIRouter(prefix="/settings", tags=["settings"])

class SettingRequest(BaseModel):
    key: str
    value: str

class SettingResponse(BaseModel):
    key: str
    value: str
    description: Optional[str] = None

from sqlalchemy import select

@router.get("/system/{key}")
async def get_setting(key: str, session: AsyncSession = Depends(get_db)):
    """Get a system setting by key"""
    result = await session.execute(select(SystemSetting).where(SystemSetting.key == key))
    setting = result.scalar_one_or_none()
    if not setting:
        # Return defaults for known keys
        if key == "discovery_host_ip":
            return {"key": key, "value": "", "description": "IP address for Docker bot discovery (for example 127.0.0.1 or your server IP)"}
        if key == "api_username":
            return {"key": key, "value": "", "description": "Username for Freqtrade bot API authentication"}
        if key == "api_password":
            return {"key": key, "value": "", "description": "Password for Freqtrade bot API authentication"}
        raise HTTPException(status_code=404, detail="Setting not found")
    return {"key": setting.key, "value": setting.value, "description": setting.description}

@router.post("/system")
async def set_setting(request: SettingRequest, session: AsyncSession = Depends(get_db)):
    """Set a system setting"""
    result = await session.execute(select(SystemSetting).where(SystemSetting.key == request.key))
    setting = result.scalar_one_or_none()
    if setting:
        setting.value = request.value
    else:
        descriptions = {
            "discovery_host_ip": "IP address for Docker bot discovery (for example 127.0.0.1 or your server IP). Leave empty to use localhost.",
            "api_username": "Username for Freqtrade bot API authentication",
            "api_password": "Password for Freqtrade bot API authentication"
        }
        setting = SystemSetting(
            key=request.key,
            value=request.value,
            description=descriptions.get(request.key)
        )
        session.add(setting)
    await session.commit()
    return {"key": setting.key, "value": setting.value, "description": setting.description}

@router.get("/system")
async def get_all_settings(session: AsyncSession = Depends(get_db)):
    """Get all system settings"""
    from sqlalchemy import select
    result = await session.execute(select(SystemSetting))
    settings = result.scalars().all()
    
    # Ensure all known settings are included
    setting_dict = {s.key: {"key": s.key, "value": s.value, "description": s.description} for s in settings}
    
    defaults = {
        "discovery_host_ip": {
            "key": "discovery_host_ip",
            "value": "",
            "description": "IP address for Docker bot discovery (for example 127.0.0.1 or your server IP). Leave empty to use localhost."
        },
        "api_username": {
            "key": "api_username",
            "value": "",
            "description": "Username for Freqtrade bot API authentication"
        },
        "api_password": {
            "key": "api_password",
            "value": "",
            "description": "Password for Freqtrade bot API authentication (hidden)"
        }
    }
    
    for key, default in defaults.items():
        if key not in setting_dict:
            setting_dict[key] = default
    
    return list(setting_dict.values())
