"""
Bybit orderbook collector for MultibotdashboardV7
Fetches L2 orderbook data from Bybit API
"""

import asyncio
import aiohttp
import asyncpg
from datetime import datetime
from typing import List, Dict, Optional
import logging

from src.services.runtime_settings import create_finance_db_pool

logger = logging.getLogger(__name__)

# Default trading pairs
DEFAULT_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'DOTUSDT',
    'LINKUSDT', 'AVAXUSDT', 'TRXUSDT', 'MATICUSDT', 'TONUSDT', 'ICPUSDT', 'SHIBUSDT',
    'LTCUSDT', 'BCHUSDT', 'ETCUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'LEOUSDT',
    'ATOMUSDT', 'FILUSDT', 'IMXUSDT', 'RNDRUSDT', 'HBARUSDT', 'ARBUSDT', 'OPUSDT',
    'MKRUSDT', 'AAVEUSDT', 'GRTUSDT', 'STXUSDT', 'INJUSDT', 'EGLDUSDT', 'FTMUSDT',
    'ALGOUSDT', 'THETAUSDT', 'MANAUSDT', 'SANDUSDT', 'FLOWUSDT', 'XTZUSDT', 'NEOUSDT',
    'KAVAUSDT', 'CRVUSDT', 'CHZUSDT', 'GALAUSDT', 'DYDXUSDT', 'SUIUSDT', 'PEPEUSDT',
    'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT', 'JASMYUSDT', 'XECUSDT', 'CSPRUSDT', 'MINAUSDT',
    'CAKEUSDT', '1INCHUSDT', 'LRCUSDT', 'ZILUSDT', 'ENJUSDT', 'BATUSDT', 'GMTUSDT',
    'ZRXUSDT', 'COMPUSDT', 'YFIUSDT', 'SUSHIUSDT', 'BALUSDT', 'RENUSDT', 'OCEANUSDT',
    'LPTUSDT', 'API3USDT', 'MASKUSDT', 'CELRUSDT', 'SKLUSDT', 'STORJUSDT', 'ANKRUSDT',
    'AUDIOUSDT', 'RLCUSDT', 'NMRUSDT', 'BANDUSDT', 'CVCUSDT', 'OXTUSDT', 'REPUSDT',
    'UMAUSDT', 'GNOUSDT', 'LSKUSDT', 'WAVESUSDT', 'KSMUSDT', 'QTUMUSDT', 'ICXUSDT',
    'DASHUSDT', 'ZENUSDT', 'XEMUSDT', 'ONTUSDT', 'IOTXUSDT', 'RVNUSDT', 'SXPUSDT',
    'VETUSDT', 'ONEUSDT', 'HOTUSDT', 'BTTUSDT', 'WINUSDT', 'DENTUSDT', 'COTIUSDT'
]


class BybitCollector:
    """Collects orderbook data from Bybit."""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
    
    async def fetch_orderbook(self, symbol: str) -> Optional[Dict]:
        """Fetch orderbook for a symbol."""
        url = f"{self.base_url}/v5/market/orderbook"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': 50
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('retCode') == 0:
                        result = data.get('result', {})
                        bids = result.get('b', [])
                        asks = result.get('a', [])
                        
                        if bids and asks:
                            best_bid = float(bids[0][0])
                            best_ask = float(asks[0][0])
                            mid_price = (best_bid + best_ask) / 2
                            spread = best_ask - best_bid
                            spread_pct = (spread / mid_price) * 100
                            
                            # Calculate depth
                            bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10])
                            ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10])
                            total_depth = bid_depth + ask_depth
                            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
                            
                            return {
                                'symbol': symbol,
                                'best_bid': best_bid,
                                'best_ask': best_ask,
                                'mid_price': mid_price,
                                'spread': spread,
                                'spread_pct': spread_pct,
                                'bid_depth': bid_depth,
                                'ask_depth': ask_depth,
                                'imbalance': imbalance
                            }
                else:
                    logger.warning(f"Bybit API error for {symbol}: {response.status}")
        return None
    
    async def fetch_all_orderbooks(self) -> List[Dict]:
        """Fetch orderbooks for all default pairs."""
        results = []
        for symbol in DEFAULT_PAIRS:
            data = await self.fetch_orderbook(symbol)
            if data:
                results.append(data)
            await asyncio.sleep(0.1)  # Rate limiting
        return results
    
    async def save_to_db(self, orderbooks: List[Dict]):
        """Save orderbook data to database."""
        pool = await create_finance_db_pool()
        
        async with pool.acquire() as conn:
            for ob in orderbooks:
                await conn.execute(
                    """
                    INSERT INTO bybit_orderbook 
                    (symbol, best_bid, best_ask, mid_price, spread, spread_pct, bid_depth, ask_depth, imbalance)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    ob['symbol'],
                    ob['best_bid'],
                    ob['best_ask'],
                    ob['mid_price'],
                    ob['spread'],
                    ob['spread_pct'],
                    ob['bid_depth'],
                    ob['ask_depth'],
                    ob['imbalance']
                )
        
        await pool.close()
        logger.info(f"Saved {len(orderbooks)} orderbooks to DB")
    
    async def run(self):
        """Run the collector."""
        try:
            logger.info("Starting Bybit orderbook collection...")
            orderbooks = await self.fetch_all_orderbooks()
            
            if orderbooks:
                await self.save_to_db(orderbooks)
                await self._log_sync('bybit_orderbook', 'success', len(orderbooks))
            else:
                await self._log_sync('bybit_orderbook', 'error', 0, 'No data received')
        except Exception as e:
            logger.error(f"Bybit collector error: {e}")
            await self._log_sync('bybit_orderbook', 'error', 0, str(e))
    
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
bybit_collector = BybitCollector()
