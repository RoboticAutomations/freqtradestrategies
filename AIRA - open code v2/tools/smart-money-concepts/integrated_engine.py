"""
Integrated Engine - Combină Order Book și Trade Data
Oferă o viziune completă: Order Book Imbalance + Trade Delta
"""

import logging
import time
import asyncio
from typing import Dict, Optional, List
from threading import Thread
from dataclasses import dataclass

from orderbook_engine import (
    EnhancedOrderBookEngine, 
    MultiSymbolOrderBookManager,
    OrderBookMetrics
)
from trade_aggregator import (
    SQLiteStorage,
    MultiSymbolTradeCollector,
    CandleAggregator,
    Candle
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedMetrics:
    symbol: str
    timestamp: float

    # Order Book metrics
    ob_delta: float  # Order Book Delta (bid_vol - ask_vol)
    ob_obi: float  # Order Book Imbalance
    ob_cvd: float  # Cumulative Volume Delta (order book)
    ob_pressure: float  # Order Book Pressure
    spread: float
    mid_price: float
    liquidity_score: float

    # Trade metrics
    trade_delta: Optional[float] = None  # Buy Vol - Sell Vol
    trade_volume: Optional[float] = None
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None
    candle_open: Optional[float] = None
    candle_close: Optional[float] = None
    @property
    def total_delta(self) -> float:
        if self.trade_delta is not None:
            return self.ob_delta + self.trade_delta
        return self.ob_delta

    @property
    def delta_divergence(self) -> Optional[float]:
        if self.trade_delta is not None:
            return self.ob_delta - self.trade_delta
        return None

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'ob_delta': self.ob_delta,
            'ob_obi': self.ob_obi,
            'ob_cvd': self.ob_cvd,
            'ob_pressure': self.ob_pressure,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'liquidity_score': self.liquidity_score,
            'trade_delta': self.trade_delta,
            'trade_volume': self.trade_volume,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'candle_open': self.candle_open,
            'candle_close': self.candle_close,
            'total_delta': self.total_delta,
            'delta_divergence': self.delta_divergence
        }


class IntegratedEngine:
    def __init__(
        self,
        symbols: List[str],
        exchange_name: str = "bybit",
        candle_interval: int = 60,
        enable_trades: bool = True,
        db_path: str = "orderbook_cache.db",
        depth: int = 25,
        cvd_reset_hours: int | None = None,
        max_history: int = 1000
    ):
        self.symbols = symbols
        self.exchange_name = exchange_name
        self.candle_interval = candle_interval
        self.enable_trades = enable_trades
        self.ob_manager = MultiSymbolOrderBookManager(
            symbols=symbols,
            depth=depth,
            exchange_name=exchange_name,
            cvd_reset_hours=cvd_reset_hours,
            max_history=max_history
        )

        self.trade_collector: Optional[MultiSymbolTradeCollector] = None
        self.storage: Optional[SQLiteStorage] = None

        if enable_trades:
            self.storage = SQLiteStorage(db_path)
            self.trade_collector = MultiSymbolTradeCollector(
                symbols=symbols,
                candle_interval=candle_interval,
                storage=self.storage,
                testnet=False
            )
            self._start_trade_collection()

        logger.info(f"IntegratedEngine initialized for {len(symbols)} symbols")

    def _start_trade_collection(self):
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.trade_collector.start_all())
            except Exception as e:
                logger.error(f"Trade collection error: {e}")
            finally:
                loop.close()

        thread = Thread(target=run_async_loop, daemon=True)
        thread.start()
        logger.info("Trade collection started in background")

    def get_integrated_metrics(self, symbol: str) -> Optional[IntegratedMetrics]:

        engine = self.ob_manager.get_engine(symbol)
        if not engine or not engine.metrics_history:
            return None

        ob_metrics: OrderBookMetrics = engine.metrics_history[-1]

        trade_delta = None
        trade_volume = None
        buy_volume = None
        sell_volume = None
        candle_open = None
        candle_close = None

        if self.storage:
            candles = self.storage.get_candles(symbol, limit=1)
            if candles:
                latest_candle = candles[0]
                trade_delta = latest_candle.delta
                trade_volume = latest_candle.volume
                buy_volume = latest_candle.buy_vol
                sell_volume = latest_candle.sell_vol
                candle_open = latest_candle.open
                candle_close = latest_candle.close

        integrated = IntegratedMetrics(
            symbol=symbol,
            timestamp=ob_metrics.timestamp,
            ob_delta=ob_metrics.delta,
            ob_obi=ob_metrics.obi,
            ob_cvd=ob_metrics.cvd,
            ob_pressure=ob_metrics.order_book_pressure,
            spread=ob_metrics.spread,
            mid_price=ob_metrics.mid_price,
            liquidity_score=ob_metrics.liquidity_score,
            trade_delta=trade_delta,
            trade_volume=trade_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            candle_open=candle_open,
            candle_close=candle_close
        )

        return integrated

    def get_all_integrated_metrics(self) -> Dict[str, Optional[IntegratedMetrics]]:
        results = {}
        for symbol in self.symbols:
            results[symbol] = self.get_integrated_metrics(symbol)
        return results

    def update_orderbook(self, symbol: str):

        return self.ob_manager.update_symbol(symbol)

    def update_all_orderbooks(self):

        return self.ob_manager.update_all()

    def get_candle_history(self, symbol: str, limit: int = 100) -> List[Candle]:

        if not self.storage:
            return []
        return self.storage.get_candles(symbol, limit=limit)

    def get_statistics(self) -> Dict[str, dict]:

        stats = {}

        for symbol in self.symbols:

            ob_stats = self.ob_manager.get_engine(symbol).get_statistics()

            trade_stats = {}
            if self.storage:
                trade_stats = self.storage.get_statistics(symbol)

            stats[symbol] = {
                'orderbook': ob_stats,
                'trades': trade_stats
            }

        return stats

    def stop(self):
        if self.trade_collector:
            self.trade_collector.stop_all()
        logger.info("IntegratedEngine stopped")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    symbols = ["BTC/USDT", "ETH/USDT"]

    engine = IntegratedEngine(
        symbols=symbols,
        exchange_name="bybit",
        candle_interval=60,
        enable_trades=True
    )

    print("="*60)
    print("🚀 Integrated Engine Started")
    print("="*60)
    print("Collecting Order Book + Trade Data...")
    print("Press Ctrl+C to stop\n")

    try:

        while True:

            engine.update_all_orderbooks()

            metrics = engine.get_all_integrated_metrics()

            for symbol, data in metrics.items():
                if data:
                    print(f"\n{symbol}:")
                    print(f"  OB Delta: {data.ob_delta:.2f}")
                    print(f"  Trade Delta: {data.trade_delta:.2f if data.trade_delta else 'N/A'}")
                    print(f"  Total Delta: {data.total_delta:.2f}")
                    print(f"  Divergence: {data.delta_divergence:.2f if data.delta_divergence else 'N/A'}")
                    print(f"  OBI: {data.ob_obi:.4f}")
                    print(f"  Pressure: {data.ob_pressure:.4f}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n⏹ Stopping...")
        engine.stop()

        print("\n" + "="*60)
        print("📊 Final Statistics:")
        print("="*60)
        stats = engine.get_statistics()
        for symbol, data in stats.items():
            print(f"\n{symbol}:")
            print(f"  Order Book Updates: {data['orderbook']['total_updates']}")
            if 'trades' in data and data['trades']:
                print(f"  Total Candles: {data['trades'].get('total_candles', 0)}")
                print(f"  Total Volume: {data['trades'].get('total_volume', 0):.2f}")
                print(f"  Cumulative Delta: {data['trades'].get('cumulative_delta', 0):.2f}")
