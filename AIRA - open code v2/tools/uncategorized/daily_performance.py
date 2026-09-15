"""Daily performance API endpoints.

Reads from dashboard DB table: daily_performance.
Used by the main dashboard to show overall daily profit trend.
"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db

router = APIRouter()


@router.get("/daily-profits")
async def get_daily_profits(
    db: Annotated[AsyncSession, Depends(get_db)],
    days: int = Query(30, ge=1, le=365, description="Number of days to return"),
) -> dict:
    """Return daily profit totals for the last N days.

    - daily_performance can contain multiple rows per day (per bot)
    - aggregate profit_abs per date to get overall portfolio daily profit
    - order DESC/limit for performance, then sort ASC for chart rendering
    """

    result = await db.execute(
        text(
            """
            WITH recent AS (
                SELECT
                    date,
                    COALESCE(SUM(profit), 0) AS profit,
                    COALESCE(SUM(win_count), 0) AS win_count
                FROM daily_performance
                GROUP BY date
                ORDER BY date DESC
                LIMIT :limit
            )
            SELECT date, profit, win_count
            FROM recent
            ORDER BY date ASC
            """
        ),
        {"limit": days},
    )

    rows = result.mappings().all()

    return {
        "status": "success",
        "data": [
            {
                "date": (r["date"].isoformat() if isinstance(r["date"], date) else str(r["date"])),
                "profit": float(r["profit"] or 0),
                "win_count": int(r["win_count"] or 0),
            }
            for r in rows
        ],
    }
