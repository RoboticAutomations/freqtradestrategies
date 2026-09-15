"""
FinanceData API routes for MultibotdashboardV7
Integrates AlexFinanceData into the dashboard
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime, timedelta
import asyncpg
from pydantic import BaseModel

from src.services.runtime_settings import create_finance_db_pool
from src.services.finance_collectors.stock_collector import stock_collector

# Auth removed for public access

router = APIRouter(prefix="/finance", tags=["finance"])

# Database connection
async def get_db_pool():
    """Get database connection pool."""
    return await create_finance_db_pool()

# Pydantic models
class CryptoPrice(BaseModel):
    id: int
    coin_id: str
    symbol: str
    name: str
    price_usd: float
    price_eur: Optional[float]
    price_btc: Optional[float]
    market_cap: Optional[int]
    volume_24h: Optional[int]
    change_24h_pct: Optional[float]
    timestamp: datetime
    source: str

class Stock(BaseModel):
    id: int
    symbol: str
    name: str
    price: float
    change: Optional[float]
    change_percent: Optional[float]
    volume: Optional[int]
    market_cap: Optional[int]
    pe_ratio: Optional[float]
    sector: Optional[str]
    industry: Optional[str]
    timestamp: datetime

class NewsItem(BaseModel):
    id: int
    title: str
    source: str
    url: Optional[str]
    symbol: Optional[str]
    category: Optional[str]
    published_at: Optional[datetime]
    sentiment_score: Optional[float]
    timestamp: datetime

class EconomicIndicator(BaseModel):
    id: int
    indicator_id: str
    name: str
    value: float
    date: Optional[datetime]
    timestamp: datetime

class BybitOrderbook(BaseModel):
    id: int
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    spread_pct: Optional[float]
    bid_depth: Optional[float]
    ask_depth: Optional[float]
    imbalance: Optional[float]
    timestamp: datetime


class BinanceOrderbook(BaseModel):
    id: int
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    spread_pct: Optional[float]
    bid_depth: Optional[float]
    ask_depth: Optional[float]
    imbalance: Optional[float]
    timestamp: datetime


class StockWatchlistItem(BaseModel):
    symbol: str
    created_at: datetime


class StockWatchlistCreate(BaseModel):
    symbol: str

# API Routes

@router.get("/crypto/prices", response_model=List[CryptoPrice])
async def get_crypto_prices(
    # Removed auth for public access
    limit: int = 100,
):
    """Get latest crypto prices."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (coin_id) *
            FROM crypto_prices
            ORDER BY coin_id, timestamp DESC
            LIMIT $1
            """,
            limit
        )
    await pool.close()
    
    # Sort: bitcoin and ethereum first, then by market cap
    sorted_rows = sorted(rows, key=lambda x: (
        0 if x['coin_id'] == 'bitcoin' else (1 if x['coin_id'] == 'ethereum' else 2),
        -x['market_cap'] if x['market_cap'] else 0
    ))
    
    return [dict(row) for row in sorted_rows]

@router.get("/crypto/movers")
async def get_crypto_movers(
    # Removed auth for public access
    category: str = "gainers",
    limit: int = 10,
):
    """Get top crypto gainers or losers."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM crypto_movers
            WHERE category = $1
            ORDER BY timestamp DESC, change_24h_pct DESC
            LIMIT $2
            """,
            category, limit
        )
    await pool.close()
    return [dict(row) for row in rows]

@router.get("/stocks", response_model=List[Stock])
async def get_stocks(
    # Removed auth for public access
    limit: int = 100,
    sector: Optional[str] = None,
):
    """Get latest stock data."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if sector:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) *
                FROM stocks
                WHERE sector = $1
                ORDER BY symbol, timestamp DESC
                LIMIT $2
                """,
                sector, limit
            )
        else:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) *
                FROM stocks
                ORDER BY symbol, timestamp DESC
                LIMIT $1
                """,
                limit
            )
    await pool.close()
    return [dict(row) for row in rows]


@router.get("/stocks/watchlist", response_model=List[StockWatchlistItem])
async def get_stock_watchlist():
    """Get tracked stock symbols."""
    await stock_collector.ensure_watchlist_table()
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT symbol, created_at
            FROM stock_watchlist
            ORDER BY symbol ASC
            """
        )
    await pool.close()
    return [dict(row) for row in rows]


@router.post("/stocks/watchlist")
async def add_stock_watchlist_item(payload: StockWatchlistCreate):
    """Add a stock symbol to the tracked watchlist."""
    symbol = (payload.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    await stock_collector.ensure_watchlist_table()
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO stock_watchlist (symbol)
            VALUES ($1)
            ON CONFLICT (symbol) DO NOTHING
            """,
            symbol,
        )
    await pool.close()

    await stock_collector.run()
    return {"message": "Stock added", "symbol": symbol}


@router.delete("/stocks/watchlist/{symbol}")
async def delete_stock_watchlist_item(symbol: str):
    """Remove a stock symbol from the tracked watchlist."""
    normalized_symbol = (symbol or "").strip().upper()
    if not normalized_symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    await stock_collector.ensure_watchlist_table()
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM stock_watchlist
            WHERE symbol = $1
            """,
            normalized_symbol,
        )
        await conn.execute(
            """
            DELETE FROM stocks
            WHERE symbol = $1
            """,
            normalized_symbol,
        )
    await pool.close()

    if result.endswith("0"):
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    return {"message": "Stock removed", "symbol": normalized_symbol}

@router.get("/news", response_model=List[NewsItem])
async def get_news(
    # Removed auth for public access
    limit: int = 50,
    category: Optional[str] = None,
    symbol: Optional[str] = None,
):
    """Get latest news."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        query = "SELECT * FROM news WHERE 1=1"
        params = []
        
        if category:
            params.append(category)
            query += f" AND category = ${len(params)}"
        
        if symbol:
            params.append(symbol)
            query += f" AND symbol = ${len(params)}"
        
        query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
    await pool.close()
    return [dict(row) for row in rows]

@router.get("/economic", response_model=List[EconomicIndicator])
async def get_economic_indicators(
    # Public access
):
    """Get latest economic indicators."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (indicator_id) *
            FROM economic_indicators
            ORDER BY indicator_id, timestamp DESC
            """
        )
    await pool.close()
    return [dict(row) for row in rows]

@router.get("/bybit/orderbook", response_model=List[BybitOrderbook])
async def get_bybit_orderbook(
    # Removed auth for public access
    symbol: Optional[str] = None,
):
    """Get Bybit orderbook data."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if symbol:
            rows = await conn.fetch(
                """
                SELECT *
                FROM bybit_orderbook
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol
            )
        else:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) *
                FROM bybit_orderbook
                ORDER BY symbol, timestamp DESC
                """
            )
    await pool.close()
    return [dict(row) for row in rows]


@router.get("/binance/orderbook", response_model=List[BinanceOrderbook])
async def get_binance_orderbook(
    symbol: Optional[str] = None,
):
    """Get Binance orderbook data."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if symbol:
            rows = await conn.fetch(
                """
                SELECT *
                FROM binance_orderbook
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol
            )
        else:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) *
                FROM binance_orderbook
                ORDER BY symbol, timestamp DESC
                """
            )
    await pool.close()
    return [dict(row) for row in rows]

@router.get("/portfolio/summary")
async def get_portfolio_summary(
    # Public access
):
    """Get portfolio summary."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
    await pool.close()
    return dict(row) if row else {}

@router.get("/sync/status")
async def get_sync_status(
    # Public access
):
    """Get data sync status."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (source) *
            FROM sync_log
            ORDER BY source, created_at DESC
            """
        )
    await pool.close()
    return {row['source']: dict(row) for row in rows}

@router.post("/sync/trigger")
async def trigger_sync(
    source: str,
    # Public access
):
    """Trigger data sync for a source."""
    # This would trigger the AlexFinanceData collector
    # For now, just return success
    return {"status": "triggered", "source": source}
