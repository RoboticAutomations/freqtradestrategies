"""Replay API endpoints for freqtrade-replay integration."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db
from src.services.replay_service import get_replay_service, ReplayService

router = APIRouter(tags=["replay"])


def success_response(data: Any) -> dict[str, Any]:
    """Wrap response in standard format."""
    return {"status": "success", "data": data}


@router.post("/run")
async def trigger_replay(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    replay_svc: ReplayService = Depends(get_replay_service),
) -> dict[str, Any]:
    """Trigger a new freqtrade-replay Docker run.

    Payload fields:
        - strategy (str, required): Strategy name
        - timerange (str, required): e.g. "20240101-20241231"
        - bot_id (str, optional): Associated bot UUID
        - timeframe (str, optional): e.g. "15m"
        - generate_report (bool, optional): Generate HTML report
    """
    strategy = payload.get("strategy")
    timerange = payload.get("timerange")
    if not strategy or not timerange:
        raise HTTPException(status_code=400, detail="strategy and timerange are required")

    bot_id = payload.get("bot_id")
    timeframe = payload.get("timeframe")

    # Insert pending row
    result = await db.execute(
        text(
            """
            INSERT INTO replay_runs
                (bot_id, strategy_name, timerange, timeframe, status, notes)
            VALUES
                (:bot_id, :strategy_name, :timerange, :timeframe, 'pending', :notes)
            RETURNING id
            """
        ),
        {
            "bot_id": bot_id,
            "strategy_name": strategy,
            "timerange": timerange,
            "timeframe": timeframe,
            "notes": payload.get("notes", ""),
        },
    )
    row = result.fetchone()
    replay_id = row[0] if row else None
    await db.commit()

    # Kick off async background execution
    import asyncio

    asyncio.create_task(
        replay_svc.run_replay(
            db=db,
            replay_id=replay_id,
            strategy_name=strategy,
            timerange=timerange,
            bot_id=bot_id,
            timeframe=timeframe,
        )
    )

    return success_response(
        {
            "replay_id": replay_id,
            "status": "pending",
            "strategy": strategy,
            "timerange": timerange,
            "timeframe": timeframe,
            "message": "Replay run queued and will execute in background",
        }
    )


@router.get("/results")
async def list_replay_results(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    strategy_name: Optional[str] = Query(None),
    bot_id: Optional[str] = Query(None),
) -> dict[str, Any]:
    """List replay runs with optional filtering."""
    where_clauses: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if status:
        where_clauses.append("status = :status")
        params["status"] = status
    if strategy_name:
        where_clauses.append("strategy_name = :strategy_name")
        params["strategy_name"] = strategy_name
    if bot_id:
        where_clauses.append("bot_id = :bot_id")
        params["bot_id"] = bot_id

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Total count
    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM replay_runs {where_sql}"), params
    )
    total = count_result.scalar() or 0

    # Paginated rows
    rows_result = await db.execute(
        text(
            f"""
            SELECT
                id, bot_id, strategy_name, timerange, timeframe,
                status, started_at, completed_at,
                profit_pct, profit_abs, win_rate, total_trades,
                max_drawdown_pct, sharpe, profit_factor,
                docker_container_id, docker_log_path, report_html_path,
                report_json, error_message, notes
            FROM replay_runs
            {where_sql}
            ORDER BY started_at DESC NULLS LAST
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    )

    rows = []
    for row in rows_result.fetchall():
        rows.append(
            {
                "id": row[0],
                "bot_id": str(row[1]) if row[1] else None,
                "strategy_name": row[2],
                "timerange": row[3],
                "timeframe": row[4],
                "status": row[5],
                "started_at": row[6].isoformat() if row[6] else None,
                "completed_at": row[7].isoformat() if row[7] else None,
                "profit_pct": float(row[8]) if row[8] is not None else None,
                "profit_abs": float(row[9]) if row[9] is not None else None,
                "win_rate": float(row[10]) if row[10] is not None else None,
                "total_trades": row[11],
                "max_drawdown_pct": float(row[12]) if row[12] is not None else None,
                "sharpe": float(row[13]) if row[13] is not None else None,
                "profit_factor": float(row[14]) if row[14] is not None else None,
                "docker_container_id": row[15],
                "docker_log_path": row[16],
                "report_html_path": row[17],
                "report_json": row[18],
                "error_message": row[19],
                "notes": row[20],
            }
        )

    return success_response(
        {
            "results": rows,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    )


@router.get("/{replay_id}")
async def get_replay_detail(
    replay_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get single replay run details."""
    result = await db.execute(
        text(
            """
            SELECT
                id, bot_id, strategy_name, timerange, timeframe,
                status, started_at, completed_at,
                profit_pct, profit_abs, win_rate, total_trades,
                max_drawdown_pct, sharpe, profit_factor,
                docker_container_id, docker_log_path, report_html_path,
                report_json, error_message, notes
            FROM replay_runs
            WHERE id = :id
            """
        ),
        {"id": replay_id},
    )

    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Replay run {replay_id} not found")

    return success_response(
        {
            "id": row[0],
            "bot_id": str(row[1]) if row[1] else None,
            "strategy_name": row[2],
            "timerange": row[3],
            "timeframe": row[4],
            "status": row[5],
            "started_at": row[6].isoformat() if row[6] else None,
            "completed_at": row[7].isoformat() if row[7] else None,
            "profit_pct": float(row[8]) if row[8] is not None else None,
            "profit_abs": float(row[9]) if row[9] is not None else None,
            "win_rate": float(row[10]) if row[10] is not None else None,
            "total_trades": row[11],
            "max_drawdown_pct": float(row[12]) if row[12] is not None else None,
            "sharpe": float(row[13]) if row[13] is not None else None,
            "profit_factor": float(row[14]) if row[14] is not None else None,
            "docker_container_id": row[15],
            "docker_log_path": row[16],
            "report_html_path": row[17],
            "report_json": row[18],
            "error_message": row[19],
            "notes": row[20],
        }
    )


@router.delete("/{replay_id}")
async def delete_replay(
    replay_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Delete a replay run by ID."""
    # Check existence
    result = await db.execute(
        text("SELECT id FROM replay_runs WHERE id = :id"), {"id": replay_id}
    )
    if not result.fetchone():
        raise HTTPException(status_code=404, detail=f"Replay run {replay_id} not found")

    # Prevent deleting running replays
    result = await db.execute(
        text("SELECT status FROM replay_runs WHERE id = :id"), {"id": replay_id}
    )
    status_row = result.fetchone()
    if status_row and status_row[0] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running replay")

    await db.execute(text("DELETE FROM replay_runs WHERE id = :id"), {"id": replay_id})
    await db.commit()

    return success_response({"message": f"Replay run {replay_id} deleted successfully"})
