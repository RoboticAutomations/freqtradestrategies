"""Discovery API endpoints."""

from datetime import datetime


def utc_naive_now() -> datetime:
    """UTC timestamp without tzinfo (safe for TIMESTAMP WITHOUT TIME ZONE columns)."""
    return datetime.utcnow()
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser
from src.models import get_db
from src.models.bot import Bot, BotEnvironment, HealthState, SourceMode
from src.schemas.bot import BotResponse
from src.schemas.discovery import (
    DiscoveryResultData,
    DiscoveryStatusData,
    DiscoveryStatusResponse,
    DiscoveryTriggerResponse,
    ManualBotRequest,
    ManualBotResponse,
)
from src.services.discovery.orchestrator import discovery_orchestrator
from src.services.discovery.scheduler import discovery_scheduler
from src.services.connectors.api import APIConnector

router = APIRouter()


@router.post("/trigger", response_model=DiscoveryTriggerResponse)
async def trigger_discovery(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DiscoveryTriggerResponse:
    """Manually trigger a discovery scan.

    Requires authentication. Scans all configured sources (Docker, filesystem)
    and updates the bot registry.

    Returns:
        Discovery result with counts and discovered bots.
    """
    result = await discovery_scheduler.trigger_manual_scan()

    # Fetch updated bot list
    bots_result = await db.execute(select(Bot))
    bots = [BotResponse.model_validate(bot) for bot in bots_result.scalars()]

    return DiscoveryTriggerResponse(
        data=DiscoveryResultData(
            discovered=result["discovered"],
            new=result["new"],
            updated=result["updated"],
            removed=result["removed"],
            bots=bots,
        )
    )


@router.get("/status", response_model=DiscoveryStatusResponse)
async def get_discovery_status(
    current_user: CurrentUser,
) -> DiscoveryStatusResponse:
    """Get discovery service status.

    Returns information about enabled discovery sources,
    last scan time, and next scheduled scan.

    Returns:
        Discovery service status.
    """
    discovery_status = await discovery_orchestrator.get_status()

    return DiscoveryStatusResponse(
        data=DiscoveryStatusData(
            docker_enabled=discovery_status["docker_enabled"],
            docker_available=discovery_status["docker_available"],
            filesystem_enabled=discovery_status["filesystem_enabled"],
            filesystem_available=discovery_status["filesystem_available"],
            last_scan=discovery_status["last_scan"],
            scan_interval_seconds=discovery_status["scan_interval_seconds"],
            next_scan=discovery_status["next_scan"],
        )
    )


@router.post("/manual", response_model=ManualBotResponse)
async def register_manual_bot(
    request: ManualBotRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ManualBotResponse:
    """Manually register a bot by API URL.

    Attempts to connect to the provided API URL to verify the bot
    and retrieve its configuration.

    Args:
        request: Manual bot registration request with API URL and optional credentials.

    Returns:
        Registered bot information.

    Raises:
        400: Invalid API URL or unable to connect.
        409: Bot already registered.
    """
    # Check if bot with this API URL already exists
    existing = await db.execute(
        select(Bot).where(Bot.api_url == request.api_url)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A bot with this API URL is already registered",
        )

    # Try to connect and get bot info
    import uuid
    temp_bot_id = str(uuid.uuid4())

    connector = APIConnector(
        bot_id=temp_bot_id,
        api_url=request.api_url,
        username=request.username,
        password=request.password,
    )

    try:
        # Check health first
        health_result = await connector.check_health()
        if not health_result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to connect to bot API: {health_result.error}",
            )

        # Get bot status/config
        status_result = await connector.get_status()
        if not status_result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to get bot status: {status_result.error}",
            )

        bot_status = status_result.data

        # Create the bot entry
        bot = Bot(
            id=temp_bot_id,
            name=request.name,
            environment=BotEnvironment.MANUAL,
            api_url=request.api_url,
            host=request.api_url.split("://")[1].split("/")[0] if "://" in request.api_url else request.api_url,
            exchange=bot_status.exchange,
            strategy=bot_status.strategy,
            trading_mode=bot_status.trading_mode,
            is_dryrun=bot_status.is_dryrun,
            health_state=HealthState.HEALTHY,
            source_mode=SourceMode.API,
            discovered_at=datetime.utcnow(),
            last_seen=utc_naive_now(),
        )

        db.add(bot)
        await db.commit()
        await db.refresh(bot)

        return ManualBotResponse(
            data=BotResponse.model_validate(bot),
            message=f"Bot '{request.name}' registered successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register bot: {str(e)}",
        )
    finally:
        await connector.close()
