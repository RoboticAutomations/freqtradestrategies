"""Historic data API routes (read-only from analytics DB)."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db

router = APIRouter()


def success_response(data: Any) -> dict[str, Any]:
    """Wrap response in standard format."""
    return {"status": "success", "data": data}


@router.get("/bots")
async def list_historic_bots(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List all bot names with their activity status.
    
    V4: Shows ALL bots that ever had data, with last_seen timestamp.
    This preserves historic data for old/disabled bots while showing which are active.
    """
    result = await db.execute(
        text("""
            SELECT 
                bs.bot_name as bot_name,
                MAX(bs.timestamp) as last_seen,
                COUNT(*) as snapshot_count
            FROM bot_snapshots bs
            GROUP BY bs.bot_name
            ORDER BY last_seen DESC
        """)
    )
    
    bots = []
    for row in result.fetchall():
        bot_name, last_seen, snapshot_count = row
        # V4 Fix: Consider active if seen in last 1 hour (not 24h)
        # This better reflects currently running vs stopped/renamed bots
        # last_seen comes from SQLAlchemy and may be naive (timestamp without tz).
        # Treat it as UTC consistently.
        is_active = last_seen and (datetime.utcnow() - last_seen).total_seconds() < 3600
        bots.append({
            "name": bot_name,
            "last_seen": last_seen.isoformat() if last_seen else None,
            "snapshot_count": snapshot_count,
            "is_active": is_active
        })
    
    return success_response({"bots": bots})


@router.get("/bot/{bot_name}/latest")
async def get_latest_bot_snapshot(
    bot_name: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get the most recent snapshot for a specific bot."""
    result = await db.execute(
        text("""
            SELECT 
                bs.timestamp,
                bs.profit_all,
                bs.profit_closed,
                bs.winrate,
                bs.trade_count,
                bs.open_trades,
                bs.balance
            FROM bot_snapshots bs
            WHERE bs.bot_name = :bot_name
            ORDER BY bs.timestamp DESC
            LIMIT 1
        """),
        {"bot_name": bot_name}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No snapshots found for bot: {bot_name}")
    
    return success_response({
        "bot_name": bot_name,
        "timestamp": row[0],
        "profit_all": row[1],
        "profit_closed": row[2],
        "winrate": row[3],
        "trade_count": row[4],
        "open_trades": row[5],
        "balance": row[6],
        "stake_currency": "USDT",
    })


@router.get("/bot/{bot_name}/series")
async def get_bot_time_series(
    bot_name: str,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get time series data for a bot (for charts)."""
    result = await db.execute(
        text("""
            SELECT 
                bs.timestamp,
                bs.profit_all,
                bs.winrate,
                bs.balance,
                bs.trade_count
            FROM bot_snapshots bs
            WHERE bs.bot_name = :bot_name
            ORDER BY bs.timestamp DESC
            LIMIT :limit
        """),
        {"bot_name": bot_name, "limit": limit}
    )
    rows = result.fetchall()
    
    return success_response({
        "bot_name": bot_name,
        "points": [
            {
                "timestamp": row[0],
                "profit_all": row[1],
                "winrate": row[2],
                "balance": row[3],
                "trade_count": row[4],
            }
            for row in rows
        ]
    })
