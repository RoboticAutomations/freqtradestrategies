"""
Agent API - Multibotdashboard V8
Dynamic Weight Trading Agent Endpoints
Uses asyncpg like finance.py
"""

from datetime import datetime, timedelta
from typing import List, Optional
import asyncpg
import json

from fastapi import APIRouter, HTTPException, Query

from src.services.agent_docker import AgentDockerController
from src.services.runtime_settings import create_finance_db_pool

router = APIRouter(prefix="/agent", tags=["agent"])

# Database connection - SAME as finance.py
async def get_db_pool():
    """Get database connection pool for financial_data."""
    return await create_finance_db_pool()


@router.get("/weights")
async def get_all_weights():
    """Get signal weights for all market regimes"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT regime, price_momentum_weight, volume_weight, sentiment_weight,
                   macro_weight, orderbook_weight, total_trades, win_rate, last_updated
            FROM signal_weights
            ORDER BY win_rate DESC
        """)
    await pool.close()
    
    return [{
        "regime": row['regime'],
        "price_momentum_weight": row['price_momentum_weight'],
        "volume_weight": row['volume_weight'],
        "sentiment_weight": row['sentiment_weight'],
        "macro_weight": row['macro_weight'],
        "orderbook_weight": row['orderbook_weight'],
        "total_weight": round(sum([row['price_momentum_weight'], row['volume_weight'], 
                                   row['sentiment_weight'], row['macro_weight'], 
                                   row['orderbook_weight']]), 2),
        "win_rate": row['win_rate'],
        "total_trades": row['total_trades'],
        "last_updated": row['last_updated'].isoformat() if row['last_updated'] else None
    } for row in rows]


@router.get("/weights/{regime}")
async def get_weights_by_regime(regime: str):
    """Get signal weights for specific regime"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM signal_weights WHERE regime = $1
        """, regime)
    await pool.close()
    
    if not row:
        raise HTTPException(status_code=404, detail=f"Regime '{regime}' not found")
    
    total = sum([row['price_momentum_weight'], row['volume_weight'], 
                 row['sentiment_weight'], row['macro_weight'], row['orderbook_weight']])
    
    return {
        "regime": row['regime'],
        "price_momentum_weight": row['price_momentum_weight'],
        "volume_weight": row['volume_weight'],
        "sentiment_weight": row['sentiment_weight'],
        "macro_weight": row['macro_weight'],
        "orderbook_weight": row['orderbook_weight'],
        "total_weight": round(total, 2),
        "win_rate": row['win_rate'],
        "total_trades": row['total_trades'],
        "last_updated": row['last_updated'].isoformat() if row['last_updated'] else None
    }


@router.put("/weights/{regime}")
async def update_weights(regime: str, update: dict):
    """Update signal weights for a regime (manual override)"""
    total = sum([update.get('price_momentum_weight', 0), update.get('volume_weight', 0),
                 update.get('sentiment_weight', 0), update.get('macro_weight', 0),
                 update.get('orderbook_weight', 0)])
    
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail=f"Weights must sum to 1.0, got {total}")
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE signal_weights 
            SET price_momentum_weight = $1, volume_weight = $2, sentiment_weight = $3,
                macro_weight = $4, orderbook_weight = $5, last_updated = NOW()
            WHERE regime = $6
        """, update['price_momentum_weight'], update['volume_weight'],
           update['sentiment_weight'], update['macro_weight'], update['orderbook_weight'],
           regime)
    await pool.close()
    
    return {"status": "success", "message": f"Weights updated for {regime}"}


@router.get("/trades")
async def get_agent_trades(status: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    """Get agent trade history"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch("""
                SELECT * FROM agent_trades 
                WHERE status = $1
                ORDER BY timestamp DESC LIMIT $2
            """, status, limit)
        else:
            rows = await conn.fetch("""
                SELECT * FROM agent_trades 
                ORDER BY timestamp DESC LIMIT $1
            """, limit)
    await pool.close()
    
    return [{
        "id": row['id'],
        "timestamp": row['timestamp'].isoformat() if row['timestamp'] else None,
        "pair": row['pair'],
        "direction": row['direction'],
        "confidence": row['confidence'],
        "stake_amount": row['stake_amount'],
        "entry_price": row['entry_price'],
        "status": row['status'],
        "final_profit": row['final_profit'],
        "signals": row['signals']
    } for row in rows]


@router.get("/performance")
async def get_performance(days: int = Query(30, ge=1, le=365)):
    """Get daily performance metrics"""
    since = datetime.utcnow() - timedelta(days=days)
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                DATE_TRUNC('day', timestamp) as date,
                COUNT(*) as total_signals,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                ROUND(AVG(profit_pct), 4) as avg_profit,
                ROUND(SUM(profit_pct), 4) as total_profit
            FROM signal_performance
            WHERE timestamp >= $1 AND outcome IS NOT NULL
            GROUP BY DATE_TRUNC('day', timestamp)
            ORDER BY date DESC
        """, since)
    await pool.close()
    
    return [{
        "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10],
        "total_signals": row['total_signals'],
        "wins": row['wins'],
        "losses": row['losses'],
        "win_rate": round((row['wins'] or 0) / row['total_signals'] * 100, 2) if row['total_signals'] > 0 else 0,
        "avg_profit": row['avg_profit'] or 0,
        "total_profit": row['total_profit'] or 0
    } for row in rows]


@router.get("/performance/by-regime")
async def get_performance_by_regime():
    """Get win rate by regime"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                regime,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(profit_pct), 4) as avg_profit
            FROM signal_performance
            WHERE outcome IS NOT NULL
            GROUP BY regime
            ORDER BY wins DESC
        """)
    await pool.close()
    
    return [{
        "regime": row['regime'],
        "total_trades": row['total_trades'],
        "wins": row['wins'],
        "win_rate": round((row['wins'] or 0) / row['total_trades'] * 100, 2) if row['total_trades'] > 0 else 0,
        "avg_profit": row['avg_profit'] or 0
    } for row in rows]


@router.get("/config")
async def get_agent_config():
    """Get agent configuration"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT key, value FROM agent_config")
    await pool.close()
    return {row['key']: row['value'] for row in rows}


@router.put("/config/{key}")
async def update_agent_config(key: str, value: str):
    """Update agent configuration"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_config (key, value, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE 
            SET value = $2, updated_at = NOW()
        """, key, value)
    await pool.close()
    return {"status": "success", "key": key, "value": value}


@router.get("/regime/current")
async def get_current_regime():
    """Get current market regime"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM regime_history 
            ORDER BY timestamp DESC LIMIT 1
        """)
    await pool.close()
    
    if not row:
        return {"regime": "ranging", "timestamp": datetime.utcnow().isoformat()}
    
    return {
        "regime": row['regime'],
        "timestamp": row['timestamp'].isoformat() if row['timestamp'] else None,
        "btc_price": row['btc_price'],
        "btc_sma50": row['btc_sma50'],
        "btc_sma200": row['btc_sma200'],
        "atr_14": row['atr_14']
    }


@router.get("/status")
async def get_agent_status():
    """Get agent status and stats"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Get enabled config
        enabled_row = await conn.fetchrow(
            "SELECT value FROM agent_config WHERE key = 'enabled'"
        )
        enabled = enabled_row['value'] == 'true' if enabled_row else False
        
        # Get paper trading config
        paper_row = await conn.fetchrow(
            "SELECT value FROM agent_config WHERE key = 'paper_trading'"
        )
        paper_trading = paper_row['value'] == 'true' if paper_row else True
        
        # Get today's trades count
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        trades_row = await conn.fetchrow("""
            SELECT COUNT(*) FROM agent_trades WHERE timestamp >= $1
        """, today)
        today_trades = trades_row['count'] if trades_row else 0
        
        # Get today's signals
        signals_rows = await conn.fetch("""
            SELECT outcome FROM signal_performance 
            WHERE timestamp >= $1 AND outcome IS NOT NULL
        """, today)
        
        wins = len([r for r in signals_rows if r['outcome'] == 'win'])
        total = len(signals_rows)
    await pool.close()
    
    # Check container status
    container_running = AgentDockerController.is_container_running()
    
    return {
        "enabled": enabled,
        "container_running": container_running,
        "paper_trading": paper_trading,
        "today_trades": today_trades,
        "today_signals": total,
        "today_wins": wins,
        "today_win_rate": round(wins / total * 100, 2) if total > 0 else 0,
        "current_regime": (await get_current_regime())['regime']
    }


@router.post("/trades")
async def create_trade(trade: dict):
    """Create a new trade record from Freqtrade"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_trades (
                pair, direction, confidence, stake_amount, entry_price,
                status, final_profit, signals, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        """,
            trade.get('pair'),
            trade.get('direction'),
            trade.get('confidence'),
            trade.get('stake_amount'),
            trade.get('entry_price'),
            trade.get('status', 'pending'),
            trade.get('final_profit'),
            json.dumps(trade.get('signals', {}))
        )
    await pool.close()
    return {"status": "success", "message": "Trade logged"}


@router.post("/signals")
async def create_signal_performance(signal: dict):
    """Create signal performance record for AI learning"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO signal_performance (
                trade_id, pair, regime, direction, price_signal, volume_signal,
                sentiment_signal, macro_signal, orderbook_signal, combined_score,
                outcome, profit_pct, duration_minutes, executed, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW())
        """,
            signal.get('trade_id'),
            signal.get('pair'),
            signal.get('regime'),
            signal.get('direction'),
            signal.get('price_signal', 0),
            signal.get('volume_signal', 0),
            signal.get('sentiment_signal', 0),
            signal.get('macro_signal', 0),
            signal.get('orderbook_signal', 0),
            signal.get('combined_score', 0),
            signal.get('outcome'),
            signal.get('profit_pct'),
            signal.get('duration_minutes'),
            signal.get('executed', True)
        )
        
        # Update win count for the regime
        if signal.get('outcome') == 'win':
            await conn.execute("""
                UPDATE signal_weights 
                SET win_count = win_count + 1, total_trades = total_trades + 1
                WHERE regime = $1
            """, signal.get('regime'))
        else:
            await conn.execute("""
                UPDATE signal_weights 
                SET total_trades = total_trades + 1
                WHERE regime = $1
            """, signal.get('regime'))
    await pool.close()
    return {"status": "success", "message": "Signal logged"}


@router.post("/enable")
async def enable_agent():
    """Enable agent trading and start container"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_config (key, value, updated_at)
            VALUES ('enabled', 'true', NOW())
            ON CONFLICT (key) DO UPDATE SET value = 'true', updated_at = NOW()
        """)
    await pool.close()
    
    # Start Docker container
    success, message = AgentDockerController.start_container()
    
    if success:
        return {"status": "success", "enabled": True, "container": "started"}
    else:
        return {"status": "partial", "enabled": True, "container": "failed", "error": message}


@router.post("/disable")
async def disable_agent():
    """Disable agent trading and stop container"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_config (key, value, updated_at)
            VALUES ('enabled', 'false', NOW())
            ON CONFLICT (key) DO UPDATE SET value = 'false', updated_at = NOW()
        """)
    await pool.close()
    
    # Stop Docker container
    success, message = AgentDockerController.stop_container()
    
    if success:
        return {"status": "success", "enabled": False, "container": "stopped"}
    else:
        return {"status": "partial", "enabled": False, "container": "failed", "error": message}
