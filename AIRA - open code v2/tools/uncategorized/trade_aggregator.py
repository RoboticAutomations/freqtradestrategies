import ccxt
import asyncio
import websockets
import json
import sqlite3
import logging
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    timestamp: float
    price: float
    quantity: float
    is_buyer_maker: bool  # True = Sell aggressor, False = Buy aggressor

    @property
    def is_buy(self) -> bool:
        return not self.is_buyer_maker

    @property
    def is_sell(self) -> bool:
        return self.is_buyer_maker


@dataclass
class Candle:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    delta: float
    buy_vol: float
    sell_vol: float
    trade_count: int

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': int(self.timestamp),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'delta': self.delta,
            'buy_vol': self.buy_vol,
            'sell_vol': self.sell_vol,
            'trade_count': self.trade_count
        }


class SQLiteStorage:
    def __init__(self, db_path: str = "orderbook_cache.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._init_database()
        logger.info(f"SQLite storage initialized: {db_path}")

    def _init_database(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    delta REAL NOT NULL,
                    buy_vol REAL NOT NULL,
                    sell_vol REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
                ON trade_candles(symbol, timestamp DESC)
            """)

            conn.commit()
            conn.close()

    def insert_candle(self, candle: Candle) -> bool:
        """Inserează un candle în database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO trade_candles 
                    (symbol, timestamp, open, high, low, close, volume, 
                     delta, buy_vol, sell_vol, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candle.symbol,
                    int(candle.timestamp),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.delta,
                    candle.buy_vol,
                    candle.sell_vol,
                    candle.trade_count
                ))

                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error inserting candle: {e}")
            return False

    def get_candles(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[int] = None
    ) -> List[Candle]:

        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                if start_time:
                    cursor.execute("""
                        SELECT symbol, timestamp, open, high, low, close, 
                               volume, delta, buy_vol, sell_vol, trade_count
                        FROM trade_candles
                        WHERE symbol = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (symbol, start_time, limit))
                else:
                    cursor.execute("""
                        SELECT symbol, timestamp, open, high, low, close, 
                               volume, delta, buy_vol, sell_vol, trade_count
                        FROM trade_candles
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (symbol, limit))

                rows = cursor.fetchall()
                conn.close()

                candles = [
                    Candle(
                        symbol=row[0],
                        timestamp=row[1],
                        open=row[2],
                        high=row[3],
                        low=row[4],
                        close=row[5],
                        volume=row[6],
                        delta=row[7],
                        buy_vol=row[8],
                        sell_vol=row[9],
                        trade_count=row[10]
                    )
                    for row in rows
                ]

                return candles
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

    def get_statistics(self, symbol: str) -> dict:
        """Statistici despre candles stocate"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT 
                        COUNT(*) as count,
                        MIN(timestamp) as first_timestamp,
                        MAX(timestamp) as last_timestamp,
                        SUM(volume) as total_volume,
                        SUM(delta) as cumulative_delta
                    FROM trade_candles
                    WHERE symbol = ?
                """, (symbol,))

                row = cursor.fetchone()
                conn.close()

                return {
                    'symbol': symbol,
                    'total_candles': row[0],
                    'first_timestamp': row[1],
                    'last_timestamp': row[2],
                    'total_volume': row[3] or 0,
                    'cumulative_delta': row[4] or 0
                }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


class CandleAggregator:

    def __init__(
        self,
        symbol: str,
        candle_interval: int = 60,  # Secunde
        storage: Optional[SQLiteStorage] = None
    ):
        self.symbol = symbol
        self.candle_interval = candle_interval
        self.storage = storage
        self.current_trades: List[Trade] = []
        self.current_candle_start: Optional[float] = None
        self.total_trades = 0
        self.total_candles = 0

        logger.info(f"CandleAggregator initialized for {symbol} ({candle_interval}s)")

    def add_trade(self, trade: Trade) -> Optional[Candle]:

        candle_timestamp = (int(trade.timestamp) // self.candle_interval) * self.candle_interval

        if (self.current_candle_start is not None and 
            candle_timestamp > self.current_candle_start):
            completed_candle = self._build_candle()

            self.current_trades = [trade]
            self.current_candle_start = candle_timestamp
            self.total_trades += 1

            return completed_candle

        if self.current_candle_start is None:
            self.current_candle_start = candle_timestamp

        self.current_trades.append(trade)
        self.total_trades += 1

        return None

    def _build_candle(self) -> Optional[Candle]:
        if not self.current_trades:
            return None

        prices = np.array([t.price for t in self.current_trades], dtype=np.float64)
        quantities = np.array([t.quantity for t in self.current_trades], dtype=np.float64)
        is_buy = np.array([t.is_buy for t in self.current_trades], dtype=bool)

        # OHLC
        open_price = float(prices[0])
        high_price = float(np.max(prices))
        low_price = float(np.min(prices))
        close_price = float(prices[-1])
        total_volume = float(np.sum(quantities))
        buy_volume = float(np.sum(quantities[is_buy]))
        sell_volume = float(np.sum(quantities[~is_buy]))
        delta = buy_volume - sell_volume
        candle = Candle(
            symbol=self.symbol,
            timestamp=self.current_candle_start,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            delta=delta,
            buy_vol=buy_volume,
            sell_vol=sell_volume,
            trade_count=len(self.current_trades)
        )

        if self.storage:
            self.storage.insert_candle(candle)

        self.total_candles += 1

        logger.info(
            f"Candle completed: {self.symbol} | "
            f"Delta: {delta:.2f} | "
            f"Buy: {buy_volume:.2f} | "
            f"Sell: {sell_volume:.2f} | "
            f"Trades: {len(self.current_trades)}"
        )
        return candle

    def force_complete(self) -> Optional[Candle]:
        return self._build_candle()


class BinanceWebSocketClient:
    def __init__(
        self,
        symbol: str,
        aggregator: CandleAggregator,
        testnet: bool = False
    ):
        self.symbol = symbol.lower().replace('/', '')  # btcusdt format
        self.aggregator = aggregator
        self.testnet = testnet

        if testnet:
            self.ws_url = f"wss://stream.binancefuture.com/ws/{self.symbol}@aggTrade"
        else:
            self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@aggTrade"

        self.is_running = False
        self.reconnect_delay = 5
        self.trade_count = 0

        logger.info(f"BinanceWebSocketClient initialized for {symbol}")

    async def _handle_message(self, message: dict):
        try:


            trade = Trade(
                timestamp=message['T'] / 1000.0,
                price=float(message['p']),
                quantity=float(message['q']),
                is_buyer_maker=message['m']
            )


            completed_candle = self.aggregator.add_trade(trade)

            self.trade_count += 1
            if self.trade_count % 100 == 0:
                logger.debug(f"Processed {self.trade_count} trades for {self.symbol}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _connect(self):
        """Conectare WebSocket cu reconnect logic"""
        while self.is_running:
            try:
                logger.info(f"Connecting to {self.ws_url}")

                async with websockets.connect(self.ws_url) as ws:
                    logger.info(f"WebSocket connected: {self.symbol}")

                    async for message in ws:
                        if not self.is_running:
                            break

                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Message processing error: {e}")

            except Exception as e:
                logger.error(f"WebSocket error: {e}")

                if self.is_running:
                    logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)

    async def start(self):
        self.is_running = True
        await self._connect()

    def stop(self):
        self.is_running = False
        logger.info(f"WebSocket stopped: {self.symbol}")


class MultiSymbolTradeCollector:
    def __init__(
        self,
        symbols: List[str],
        candle_interval: int = 60,
        storage: Optional[SQLiteStorage] = None,
        testnet: bool = False
    ):
        self.symbols = symbols
        self.candle_interval = candle_interval
        self.storage = storage or SQLiteStorage()
        self.testnet = testnet

        self.aggregators: Dict[str, CandleAggregator] = {}

        self.ws_clients: Dict[str, BinanceWebSocketClient] = {}

        self._init_components()

        logger.info(f"MultiSymbolTradeCollector initialized with {len(symbols)} symbols")

    def _init_components(self):
        for symbol in self.symbols:
            aggregator = CandleAggregator(
                symbol=symbol,
                candle_interval=self.candle_interval,
                storage=self.storage
            )
            self.aggregators[symbol] = aggregator
            ws_client = BinanceWebSocketClient(
                symbol=symbol,
                aggregator=aggregator,
                testnet=self.testnet
            )
            self.ws_clients[symbol] = ws_client

    async def start_all(self):
        tasks = [
            asyncio.create_task(client.start())
            for client in self.ws_clients.values()
        ]
        logger.info("Starting WebSocket connections for all symbols...")
        await asyncio.gather(*tasks)

    def stop_all(self):
        for client in self.ws_clients.values():
            client.stop()

        logger.info("All WebSocket connections stopped")

    def get_statistics(self) -> Dict[str, dict]:
        stats = {}
        for symbol in self.symbols:
            stats[symbol] = self.storage.get_statistics(symbol)
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    storage = SQLiteStorage("orderbook_cache.db")

    symbols = ["BTC/USDT", "ETH/USDT"]
    collector = MultiSymbolTradeCollector(
        symbols=symbols,
        candle_interval=120,
        storage=storage,
        testnet=False
    )


    print("="*120)
    print("Starting Trade Collection (Ctrl+C to stop)")
    print("="*120)

    try:
        asyncio.run(collector.start_all())
    except KeyboardInterrupt:
        print("\n Stopping...")
        collector.stop_all()


        print("\n" + "="*120)
        print("Statistics:")
        print("="*120)
        stats = collector.get_statistics()
        for symbol, data in stats.items():
            print(f"\n{symbol}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
