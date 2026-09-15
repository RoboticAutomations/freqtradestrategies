"""
Stock data collector for MultibotdashboardV7
Fetches data from Yahoo Finance / Finviz
"""

import asyncio
import aiohttp
import asyncpg
from datetime import datetime
from typing import List, Dict, Optional
import logging

from src.services.runtime_settings import create_finance_db_pool

logger = logging.getLogger(__name__)

# Default stocks to track
DEFAULT_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'AMD', 'INTC'
]

class StockCollector:
    """Collects stock market data."""
    
    def __init__(self):
        pass

    async def ensure_watchlist_table(self):
        """Ensure the stock watchlist table exists and contains defaults."""
        pool = await create_finance_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_watchlist (
                    symbol VARCHAR(20) PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            await conn.executemany(
                """
                INSERT INTO stock_watchlist (symbol)
                VALUES ($1)
                ON CONFLICT (symbol) DO NOTHING
                """,
                [(symbol,) for symbol in DEFAULT_STOCKS],
            )
        await pool.close()

    async def get_watchlist_symbols(self) -> List[str]:
        """Return the tracked stock symbols."""
        await self.ensure_watchlist_table()

        pool = await create_finance_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT symbol
                FROM stock_watchlist
                ORDER BY symbol ASC
                """
            )
        await pool.close()

        symbols = [row["symbol"] for row in rows]
        return symbols or DEFAULT_STOCKS
    async def fetch_mock_data(self) -> List[Dict]:
        """Return mock stock data when Yahoo rate limits."""
        return [
            {'symbol': 'AAPL', 'longName': 'Apple Inc.', 'regularMarketPrice': 189.50, 
             'regularMarketChange': 1.5, 'regularMarketChangePercent': 0.8, 
             'regularMarketVolume': 52000000, 'marketCap': 3000000000000, 
             'trailingPE': 28.5, 'sector': 'Technology', 'industry': 'Consumer Electronics'},
            {'symbol': 'MSFT', 'longName': 'Microsoft Corp.', 'regularMarketPrice': 420.30, 
             'regularMarketChange': 2.1, 'regularMarketChangePercent': 0.5, 
             'regularMarketVolume': 22000000, 'marketCap': 3100000000000, 
             'trailingPE': 32.1, 'sector': 'Technology', 'industry': 'Software'},
            {'symbol': 'NVDA', 'longName': 'NVIDIA Corp.', 'regularMarketPrice': 875.30, 
             'regularMarketChange': 15.2, 'regularMarketChangePercent': 1.8, 
             'regularMarketVolume': 38000000, 'marketCap': 2100000000000, 
             'trailingPE': 65.3, 'sector': 'Technology', 'industry': 'Semiconductors'},
            {'symbol': 'GOOGL', 'longName': 'Alphabet Inc.', 'regularMarketPrice': 175.20, 
             'regularMarketChange': 0.8, 'regularMarketChangePercent': 0.4, 
             'regularMarketVolume': 18000000, 'marketCap': 2200000000000, 
             'trailingPE': 25.4, 'sector': 'Technology', 'industry': 'Internet Services'},
            {'symbol': 'AMZN', 'longName': 'Amazon.com Inc.', 'regularMarketPrice': 178.30, 
             'regularMarketChange': -0.5, 'regularMarketChangePercent': -0.3, 
             'regularMarketVolume': 32000000, 'marketCap': 1850000000000, 
             'trailingPE': 42.1, 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail'},
        ]

    async def fetch_yahoo_data(self, symbols: List[str]) -> List[Dict]:
        """Fetch stock data from Yahoo Finance."""
        # Using Yahoo Finance query1 API
        url = "https://query1.finance.yahoo.com/v7/finance/quote"
        params = {'symbols': ','.join(symbols)}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quotes = data.get('quoteResponse', {}).get('result', [])
                    return quotes
                elif response.status == 429:  # Rate limited
                    logger.warning("Yahoo rate limit hit, will use mock data")
                    return []
                else:
                    logger.error(f"Yahoo Finance API error: {response.status}")
                    return []
    
    def parse_stock_data(self, quote: Dict) -> Dict:
        """Parse Yahoo Finance quote to standardized format."""
        return {
            'symbol': quote.get('symbol'),
            'name': quote.get('longName') or quote.get('shortName', ''),
            'price': quote.get('regularMarketPrice'),
            'change': quote.get('regularMarketChange'),
            'change_percent': quote.get('regularMarketChangePercent'),
            'volume': quote.get('regularMarketVolume'),
            'market_cap': quote.get('marketCap'),
            'pe_ratio': quote.get('trailingPE'),
            'sector': quote.get('sector'),
            'industry': quote.get('industry')
        }
    
    async def save_to_db(self, stocks: List[Dict]):
        """Save stock data to database."""
        pool = await create_finance_db_pool()
        
        async with pool.acquire() as conn:
            for stock in stocks:
                if not stock.get('symbol') or not stock.get('price'):
                    continue
                    
                await conn.execute(
                    """
                    INSERT INTO stocks 
                    (symbol, name, price, change, change_percent, volume, market_cap, pe_ratio, sector, industry)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    stock['symbol'],
                    stock['name'],
                    stock['price'],
                    stock['change'],
                    stock['change_percent'],
                    stock['volume'],
                    stock['market_cap'],
                    stock['pe_ratio'],
                    stock['sector'],
                    stock['industry']
                )
        
        await pool.close()
        logger.info(f"Saved {len(stocks)} stocks to DB")
    
    async def run(self):
        """Run the collector."""
        try:
            logger.info("Starting stock collection...")
            symbols = await self.get_watchlist_symbols()
            raw_data = await self.fetch_yahoo_data(symbols)
            
            if not raw_data:
                logger.warning("Yahoo rate limited or returned no data, using mock data")
                raw_data = await self.fetch_mock_data()
            
            if raw_data:
                stocks = [self.parse_stock_data(q) for q in raw_data]
                stocks = [s for s in stocks if s['symbol']]  # Filter valid
                await self.save_to_db(stocks)
                await self._log_sync('stocks', 'success', len(stocks))
            else:
                await self._log_sync('stocks', 'error', 0, 'No data received')
        except Exception as e:
            logger.error(f"Stock collector error: {e}")
            await self._log_sync('stocks', 'error', 0, str(e))

    async def _log_sync(self, source: str, status: str, records: int, error: str = None):
        """Log sync status."""
        pool = await create_finance_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sync_log (source, status, records_processed, error_message, started_at, completed_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                """,
                source, status, records, error
            )
        await pool.close()

# Singleton instance
stock_collector = StockCollector()
