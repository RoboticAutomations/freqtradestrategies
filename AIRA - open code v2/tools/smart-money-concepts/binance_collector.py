"""
Binance orderbook collector for MultibotdashboardV7
Fetches L2 orderbook data from Binance API
"""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp

from src.services.finance_collectors.bybit_collector import DEFAULT_PAIRS
from src.services.runtime_settings import create_finance_db_pool

logger = logging.getLogger(__name__)


class BinanceCollector:
    """Collects orderbook data from Binance."""

    def __init__(self):
        self.base_url = "https://api.binance.com"

    async def fetch_supported_symbols(self) -> List[str]:
        """Fetch supported spot symbols once and intersect with the default watchlist."""
        url = f"{self.base_url}/api/v3/exchangeInfo"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Binance exchangeInfo request failed: {response.status}")
                    return DEFAULT_PAIRS

                data = await response.json()
                supported = {
                    symbol_info.get("symbol")
                    for symbol_info in data.get("symbols", [])
                    if symbol_info.get("status") == "TRADING"
                }

        filtered_symbols = [symbol for symbol in DEFAULT_PAIRS if symbol in supported]
        return filtered_symbols or DEFAULT_PAIRS

    async def fetch_orderbook(self, symbol: str) -> Optional[Dict]:
        """Fetch orderbook for a symbol."""
        url = f"{self.base_url}/api/v3/depth"
        params = {
            "symbol": symbol,
            "limit": 50,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Binance API error for {symbol}: {response.status}")
                    return None

                data = await response.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])

                if not bids or not asks:
                    return None

                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                spread_pct = (spread / mid_price) * 100 if mid_price else 0
                bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:10])
                ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:10])
                total_depth = bid_depth + ask_depth
                imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

                return {
                    "symbol": symbol,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price": mid_price,
                    "spread": spread,
                    "spread_pct": spread_pct,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "imbalance": imbalance,
                }

    async def fetch_all_orderbooks(self) -> List[Dict]:
        """Fetch orderbooks for all default pairs."""
        results: List[Dict] = []
        symbols = await self.fetch_supported_symbols()
        for symbol in symbols:
            data = await self.fetch_orderbook(symbol)
            if data:
                results.append(data)
            await asyncio.sleep(0.05)
        return results

    async def save_to_db(self, orderbooks: List[Dict]):
        """Save orderbook data to database."""
        pool = await create_finance_db_pool()

        async with pool.acquire() as conn:
            for ob in orderbooks:
                await conn.execute(
                    """
                    INSERT INTO binance_orderbook
                    (symbol, best_bid, best_ask, mid_price, spread, spread_pct, bid_depth, ask_depth, imbalance)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    ob["symbol"],
                    ob["best_bid"],
                    ob["best_ask"],
                    ob["mid_price"],
                    ob["spread"],
                    ob["spread_pct"],
                    ob["bid_depth"],
                    ob["ask_depth"],
                    ob["imbalance"],
                )

        await pool.close()
        logger.info(f"Saved {len(orderbooks)} Binance orderbooks to DB")

    async def run(self):
        """Run the collector."""
        try:
            logger.info("Starting Binance orderbook collection...")
            orderbooks = await self.fetch_all_orderbooks()

            if orderbooks:
                await self.save_to_db(orderbooks)
                await self._log_sync("binance_orderbook", "success", len(orderbooks))
            else:
                await self._log_sync("binance_orderbook", "error", 0, "No data received")
        except Exception as e:
            logger.error(f"Binance collector error: {e}")
            await self._log_sync("binance_orderbook", "error", 0, str(e))

    async def _log_sync(self, source: str, status: str, records: int, error: str = None):
        """Log sync status."""
        pool = await create_finance_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sync_log (source, status, records_processed, error_message, started_at, completed_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                """,
                source,
                status,
                records,
                error,
            )
        await pool.close()


binance_collector = BinanceCollector()
