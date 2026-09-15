"""Alerts API endpoints."""

import json
import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser
from src.models import get_db
from src.models.alert import Alert, AlertType, AlertSeverity
from src.schemas.alert import (
    AlertCreate,
    AlertData,
    AlertResponse,
    AlertListResponse,
    AlertCountResponse,
    AlertMarkReadRequest,
    AlertDismissRequest,
)

router = APIRouter()
ALERTS_DISABLED = os.getenv("DISABLE_ALERTS", "true").lower() == "true"


def _alert_to_data(alert: Alert) -> AlertData:
    """Convert Alert model to AlertData schema."""
    metadata = None
    if alert.metadata_json:
        try:
            metadata = json.loads(alert.metadata_json)
        except json.JSONDecodeError:
            pass

    return AlertData(
        id=alert.id,
        alert_type=alert.alert_type,
        severity=alert.severity,
        title=alert.title,
        message=alert.message,
        bot_id=alert.bot_id,
        bot_name=alert.bot_name,
        is_read=alert.is_read,
        is_dismissed=alert.is_dismissed,
        created_at=alert.created_at,
        metadata=metadata,
    )


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip"),
    unread_only: bool = Query(False, description="Only return unread alerts"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    alert_type: Optional[AlertType] = Query(None, description="Filter by alert type"),
    bot_id: Optional[str] = Query(None, description="Filter by bot ID"),
    include_dismissed: bool = Query(False, description="Include dismissed alerts"),
) -> AlertListResponse:
    """Get list of alerts.

    Returns paginated list of alerts with optional filters.

    Args:
        limit: Maximum number of alerts to return.
        offset: Number of alerts to skip.
        unread_only: Only return unread alerts.
        severity: Filter by severity level.
        alert_type: Filter by alert type.
        bot_id: Filter by bot ID.
        include_dismissed: Include dismissed alerts.

    Returns:
        List of alerts with metadata.
    """
    if ALERTS_DISABLED:
        return AlertListResponse(data=[], total=0, unread_count=0)

    # Build query
    query = select(Alert).order_by(Alert.created_at.desc())

    # Apply filters
    if not include_dismissed:
        query = query.where(Alert.is_dismissed == False)

    if unread_only:
        query = query.where(Alert.is_read == False)

    if severity:
        query = query.where(Alert.severity == severity)

    if alert_type:
        query = query.where(Alert.alert_type == alert_type)

    if bot_id:
        query = query.where(Alert.bot_id == bot_id)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get unread count
    unread_query = select(func.count()).select_from(Alert).where(
        Alert.is_read == False,
        Alert.is_dismissed == False,
    )
    unread_result = await db.execute(unread_query)
    unread_count = unread_result.scalar() or 0

    # Apply pagination
    query = query.offset(offset).limit(limit)

    # Execute query
    result = await db.execute(query)
    alerts = list(result.scalars())

    return AlertListResponse(
        data=[_alert_to_data(alert) for alert in alerts],
        total=total,
        unread_count=unread_count,
    )


@router.get("/rate-limits")
async def get_rate_limit_alerts(
    current_user: CurrentUser,
) -> dict:
    """Get active rate limit alerts.

    Returns currently active rate limits from in-memory tracking.
    Rate limits auto-expire after 10 minutes of no new occurrences.

    Returns:
        Active rate limit count and list.
    """
    from src.services.log_monitor import log_monitor

    active_limits = log_monitor.get_active_rate_limits()
    count = len(active_limits)

    return {
        "status": "success",
        "data": {
            "count": count,
            "has_active": count > 0,
            "alerts": active_limits,
        },
    }


@router.post("/rate-limits/{bot_id}/clear")
async def clear_rate_limit(
    bot_id: str,
    current_user: CurrentUser,
) -> dict:
    """Manually clear a rate limit alert for a bot.

    Args:
        bot_id: Bot ID to clear rate limit for.

    Returns:
        Success status.
    """
    from src.services.log_monitor import log_monitor

    cleared = log_monitor.clear_rate_limit(bot_id)

    if not cleared:
        raise HTTPException(status_code=404, detail="No active rate limit for this bot")

    return {"status": "success", "message": "Rate limit cleared"}


@router.get("/count", response_model=AlertCountResponse)
async def get_alert_counts(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AlertCountResponse:
    """Get alert counts by severity and read status.

    Returns:
        Alert counts broken down by severity and read status.
    """
    if ALERTS_DISABLED:
        return AlertCountResponse(
            data={"total": 0, "unread": 0, "critical": 0, "warning": 0, "info": 0}
        )

    # Total unread
    unread_query = select(func.count()).select_from(Alert).where(
        Alert.is_read == False,
        Alert.is_dismissed == False,
    )
    unread_result = await db.execute(unread_query)
    unread = unread_result.scalar() or 0

    # Critical unread
    critical_query = select(func.count()).select_from(Alert).where(
        Alert.is_read == False,
        Alert.is_dismissed == False,
        Alert.severity == AlertSeverity.CRITICAL,
    )
    critical_result = await db.execute(critical_query)
    critical = critical_result.scalar() or 0

    # Warning unread
    warning_query = select(func.count()).select_from(Alert).where(
        Alert.is_read == False,
        Alert.is_dismissed == False,
        Alert.severity == AlertSeverity.WARNING,
    )
    warning_result = await db.execute(warning_query)
    warning = warning_result.scalar() or 0

    # Info unread
    info_query = select(func.count()).select_from(Alert).where(
        Alert.is_read == False,
        Alert.is_dismissed == False,
        Alert.severity == AlertSeverity.INFO,
    )
    info_result = await db.execute(info_query)
    info = info_result.scalar() or 0

    # Total count
    total_query = select(func.count()).select_from(Alert).where(
        Alert.is_dismissed == False,
    )
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    return AlertCountResponse(
        data={
            "total": total,
            "unread": unread,
            "critical": critical,
            "warning": warning,
            "info": info,
        }
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AlertResponse:
    """Get a specific alert by ID.

    Args:
        alert_id: Alert ID.

    Returns:
        Alert details.
    """
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return AlertResponse(data=_alert_to_data(alert))


@router.post("/mark-read")
async def mark_alerts_read(
    request: AlertMarkReadRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Mark alerts as read.

    Args:
        request: Alert IDs to mark as read.

    Returns:
        Success status.
    """
    await db.execute(
        update(Alert)
        .where(Alert.id.in_(request.alert_ids))
        .values(is_read=True)
    )
    await db.commit()

    return {"status": "success", "message": f"Marked {len(request.alert_ids)} alerts as read"}


@router.post("/mark-all-read")
async def mark_all_alerts_read(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Mark all alerts as read.

    Returns:
        Success status with count of updated alerts.
    """
    result = await db.execute(
        update(Alert)
        .where(Alert.is_read == False, Alert.is_dismissed == False)
        .values(is_read=True)
    )
    await db.commit()

    return {"status": "success", "message": f"Marked {result.rowcount} alerts as read"}


@router.post("/dismiss")
async def dismiss_alerts(
    request: AlertDismissRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Dismiss alerts.

    Args:
        request: Alert IDs to dismiss.

    Returns:
        Success status.
    """
    await db.execute(
        update(Alert)
        .where(Alert.id.in_(request.alert_ids))
        .values(is_dismissed=True)
    )
    await db.commit()

    return {"status": "success", "message": f"Dismissed {len(request.alert_ids)} alerts"}


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Delete an alert.

    Args:
        alert_id: Alert ID to delete.

    Returns:
        Success status.
    """
    result = await db.execute(delete(Alert).where(Alert.id == alert_id))
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"status": "success", "message": "Alert deleted"}


@router.delete("")
async def delete_all_dismissed(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """Delete all dismissed alerts.

    Returns:
        Success status with count of deleted alerts.
    """
    result = await db.execute(delete(Alert).where(Alert.is_dismissed == True))
    await db.commit()

    return {"status": "success", "message": f"Deleted {result.rowcount} dismissed alerts"}


# Alert creation service function (used by health monitor)
async def create_alert(
    db: AsyncSession,
    alert_type: AlertType,
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    bot_id: Optional[str] = None,
    bot_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Alert:
    """Create a new alert.

    This function is called by other services (like health monitor)
    to create alerts.

    Args:
        db: Database session.
        alert_type: Type of alert.
        title: Alert title.
        message: Alert message.
        severity: Alert severity.
        bot_id: Associated bot ID.
        bot_name: Associated bot name.
        metadata: Additional metadata.

    Returns:
        Created alert.
    """
    alert = Alert(
        alert_type=alert_type,
        severity=severity,
        title=title,
        message=message,
        bot_id=bot_id,
        bot_name=bot_name,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert
