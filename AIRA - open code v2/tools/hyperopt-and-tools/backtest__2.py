"""Backtest results API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db

router = APIRouter()


def success_response(data: Any) -> dict[str, Any]:
    """Wrap response in standard format."""
    return {"status": "success", "data": data}


@router.get("")
async def list_backtest_results(
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    """List all backtest results sorted by profit."""
    
    where_clause = ""
    params: dict = {"limit": limit, "offset": offset}
    
    if strategy_name:
        where_clause = "WHERE strategy_name = :strategy_name"
        params["strategy_name"] = strategy_name
    
    result = await db.execute(
        text(f"""
            SELECT * FROM backtest_results
            {where_clause}
            ORDER BY total_profit_pct DESC
            LIMIT :limit OFFSET :offset
        """),
        params
    )
    
    rows = []
    for row in result.fetchall():
        rows.append({
            "id": row[0],
            "strategy_name": row[1],
            "timeframe": row[2],
            "timerange": row[3],
            "start_balance": float(row[4]) if row[4] else None,
            "final_balance": float(row[5]) if row[5] else None,
            "total_profit_pct": float(row[6]) if row[6] else None,
            "total_profit_abs": float(row[7]) if row[7] else None,
            "total_trades": row[8],
            "win_rate": float(row[9]) if row[9] else None,
            "avg_profit_pct": float(row[10]) if row[10] else None,
            "max_drawdown_pct": float(row[11]) if row[11] else None,
            "max_drawdown_abs": float(row[12]) if row[12] else None,
            "sharpe": float(row[13]) if row[13] else None,
            "sortino": float(row[14]) if row[14] else None,
            "calmar": float(row[15]) if row[15] else None,
            "sqn": float(row[16]) if row[16] else None,
            "profit_factor": float(row[17]) if row[17] else None,
            "expectancy": float(row[18]) if row[18] else None,
            "avg_trade_duration": row[19],
            "best_pair": row[20],
            "best_pair_profit": float(row[21]) if row[21] else None,
            "worst_pair": row[22],
            "worst_pair_profit": float(row[23]) if row[23] else None,
            "market_change": float(row[24]) if row[24] else None,
            "cagr_pct": float(row[25]) if row[25] else None,
            "backtest_date": row[26].isoformat() if row[26] else None,
        })
    
    return success_response({
        "results": rows,
        "total": len(rows),
        "limit": limit,
        "offset": offset
    })


@router.get("/strategies")
async def get_strategy_names(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get list of unique strategy names."""
    result = await db.execute(
        text("SELECT DISTINCT strategy_name FROM backtest_results ORDER BY strategy_name")
    )
    strategies = [row[0] for row in result.fetchall()]
    return success_response(strategies)


@router.get("/summary")
async def get_backtest_summary(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get summary statistics of all backtests."""
    result = await db.execute(
        text("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN total_profit_pct > 0 THEN 1 END) as profitable,
                COUNT(CASE WHEN total_profit_pct <= 0 THEN 1 END) as unprofitable,
                AVG(total_profit_pct) as avg_profit,
                MAX(total_profit_pct) as best_profit,
                MIN(total_profit_pct) as worst_profit,
                AVG(win_rate) as avg_winrate,
                SUM(total_trades) as total_trades
            FROM backtest_results
        """)
    )
    
    row = result.fetchone()
    return success_response({
        "total_strategies": row[0],
        "profitable": row[1],
        "unprofitable": row[2],
        "avg_profit_pct": float(row[3]) if row[3] else 0,
        "best_profit_pct": float(row[4]) if row[4] else 0,
        "worst_profit_pct": float(row[5]) if row[5] else 0,
        "avg_win_rate": float(row[6]) if row[6] else 0,
        "total_trades": row[7] or 0,
    })


@router.delete("/{backtest_id}")
async def delete_backtest_result(
    backtest_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Delete a backtest result by ID."""
    # Check if exists
    result = await db.execute(
        text("SELECT id FROM backtest_results WHERE id = :id"),
        {"id": backtest_id}
    )
    
    if not result.fetchone():
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    
    # Delete
    await db.execute(
        text("DELETE FROM backtest_results WHERE id = :id"),
        {"id": backtest_id}
    )
    await db.commit()
    
    return success_response({"message": f"Backtest {backtest_id} deleted successfully"})


@router.get("/{strategy_name}")
async def get_backtest_detail(
    strategy_name: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get detailed backtest result for a specific strategy."""
    result = await db.execute(
        text("""
            SELECT * FROM backtest_results
            WHERE strategy_name = :strategy_name
            ORDER BY backtest_date DESC
            LIMIT 1
        """),
        {"strategy_name": strategy_name}
    )
    
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {strategy_name}")
    
    return success_response({
        "id": row[0],
        "strategy_name": row[1],
        "timeframe": row[2],
        "timerange": row[3],
        "start_balance": float(row[4]) if row[4] else None,
        "final_balance": float(row[5]) if row[5] else None,
        "total_profit_pct": float(row[6]) if row[6] else None,
        "total_profit_abs": float(row[7]) if row[7] else None,
        "total_trades": row[8],
        "win_rate": float(row[9]) if row[9] else None,
        "avg_profit_pct": float(row[10]) if row[10] else None,
        "max_drawdown_pct": float(row[11]) if row[11] else None,
        "max_drawdown_abs": float(row[12]) if row[12] else None,
        "sharpe": float(row[13]) if row[13] else None,
        "sortino": float(row[14]) if row[14] else None,
        "calmar": float(row[15]) if row[15] else None,
        "sqn": float(row[16]) if row[16] else None,
        "profit_factor": float(row[17]) if row[17] else None,
        "expectancy": float(row[18]) if row[18] else None,
        "avg_trade_duration": row[19],
        "best_pair": row[20],
        "best_pair_profit": float(row[21]) if row[21] else None,
        "worst_pair": row[22],
        "worst_pair_profit": float(row[23]) if row[23] else None,
        "market_change": float(row[24]) if row[24] else None,
        "cagr_pct": float(row[25]) if row[25] else None,
        "backtest_date": row[26].isoformat() if row[26] else None,
    })
