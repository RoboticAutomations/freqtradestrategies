"""
Pairlist Results API - View completed pairlist optimization results
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db, PairlistJob, PairlistResult, PairlistPairResult

router = APIRouter(prefix="/pairlist-results", tags=["pairlist-results"])

@router.get("/jobs")
async def list_pairlist_jobs(
    strategy: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    session: AsyncSession = Depends(get_db)
):
    """List all completed pairlist jobs"""
    query = select(PairlistJob).order_by(desc(PairlistJob.created_at)).limit(limit)
    
    if strategy:
        query = query.where(PairlistJob.strategy == strategy)
    
    result = await session.execute(query)
    jobs = result.scalars().all()
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "strategy": job.strategy,
                "mode": job.mode,
                "status": job.status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            for job in jobs
        ]
    }

@router.get("/jobs/{job_id}")
async def get_pairlist_job(
    job_id: str,
    session: AsyncSession = Depends(get_db)
):
    """Get detailed result for a specific job"""
    
    # Get job by job_id (string), not id (integer)
    result = await session.execute(
        select(PairlistJob).where(PairlistJob.job_id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get result summary
    result_query = select(PairlistResult).where(PairlistResult.job_id == job_id)
    result = await session.execute(result_query)
    summary = result.scalar_one_or_none()
    
    # Get top pairs
    pairs_query = (
        select(PairlistPairResult)
        .where(PairlistPairResult.job_id == job_id)
        .order_by(PairlistPairResult.rank)
        .limit(50)
    )
    pairs_result = await session.execute(pairs_query)
    pairs = pairs_result.scalars().all()
    
    return {
        "job": {
            "job_id": job.job_id,
            "strategy": job.strategy,
            "mode": job.mode,
            "status": job.status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
        },
        "summary": {
            "strategy": summary.strategy if summary else job.strategy,
            "timeframe": summary.timeframe if summary else None,
            "evaluation_mode": summary.evaluation_mode if summary else job.mode,
            "total_pairs": summary.total_pairs if summary else 0,
            "best_pair": summary.best_pair if summary else None,
            "best_profit": summary.best_profit if summary else 0,
            "best_sharpe": summary.best_sharpe if summary else 0,
            "avg_profit": summary.avg_profit if summary else 0,
            "avg_win_rate": summary.avg_win_rate if summary else 0,
        } if summary else None,
        "pairs": [
            {
                "rank": pair.rank,
                "pair": pair.pair,
                "profit_total": pair.profit_total,
                "win_rate": pair.win_rate,
                "max_drawdown": pair.max_drawdown,
                "sharpe_ratio": pair.sharpe_ratio,
                "trade_count": pair.trade_count,
                "score": pair.score,
            }
            for pair in pairs
        ]
    }

@router.get("/strategies")
async def list_strategies_with_results(
    session: AsyncSession = Depends(get_db)
):
    """Get list of strategies that have pairlist results"""
    
    query = (
        select(PairlistJob.strategy)
        .where(PairlistJob.status == 'completed')
        .distinct()
    )
    result = await session.execute(query)
    strategies = [row[0] for row in result.all()]
    
    return {"strategies": strategies}
