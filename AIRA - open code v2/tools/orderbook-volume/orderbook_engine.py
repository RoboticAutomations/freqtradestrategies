import ccxt
import logging
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderBookMetrics:
    timestamp: float
    symbol: str
    bids: np.ndarray
    asks: np.ndarray
    bid_volume: float
    ask_volume: float
    delta: float
    obi: float
    cvd: float
    spread: float
    spread_percentage: float
    mid_price: float
    vwap_bid: float
    vwap_ask: float
    best_bid_volume: float
    best_ask_volume: float
    depth_imbalance: np.ndarray
    order_book_pressure: float
    liquidity_score: float = 0.0
    volume_concentration: float = 0.0
    imbalance_velocity: float = 0.0
    order_book_entropy: float = 0.0
    weighted_mid_price: float = 0.0
    liquidity_imbalance_5: float = 0.0
    liquidity_imbalance_10: float = 0.0
    spread_volatility: float = 0.0
    smart_money_index: float = 0.0
    price_level_momentum: float = 0.0
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    depth_asymmetry: float = 0.0
    liquidity_fragmentation: float = 0.0
    vwap_spread: float = 0.0
    vwap_spread_percentage: float = 0.0

    bid_depth_slope: float = 0.0
    ask_depth_slope: float = 0.0
    trade_delta: float | None = None
    trade_volume: float | None = None
    buy_volume: float | None = None
    sell_volume: float | None = None

    def __post_init__(self):
        if not isinstance(self.bids, np.ndarray):
            self.bids = np.array(self.bids, dtype=np.float64)
        if not isinstance(self.asks, np.ndarray):
            self.asks = np.array(self.asks, dtype=np.float64)
        if not isinstance(self.depth_imbalance, np.ndarray):
            self.depth_imbalance = np.array(self.depth_imbalance, dtype=np.float64)

class EnhancedOrderBookEngine:

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        depth: int = 25,
        max_history: int = 1000,
        cvd_reset_hours: int | None = None,  # None = unlimited, no reset
        exchange_name: str = "bybit"
    ):
        self.symbol = symbol
        self.depth = depth
        # CVD nunca se resetează automat dacă cvd_reset_hours este None
        self.cvd_reset_seconds = cvd_reset_hours * 3600 if cvd_reset_hours else float('inf')


        exchange_class = getattr(ccxt, exchange_name, None)
        if not exchange_class:
            raise ValueError(f"Exchange {exchange_name} not found in CCXT")

        self.exchange = exchange_class({
            "enableRateLimit": True,
            "timeout": 10000,
        })

        self._validate_symbol()
        self.metrics_history = deque(maxlen=max_history)
        self.cvd = 0.0
        self.cvd_reset_time = time.time()
        self.last_ob_hash = None
        self.cached_metrics = None


        self.update_count = 0
        self.error_count = 0

        logger.info(f"OrderBookEngine initialized for {symbol} on {exchange_name}")

    def _validate_symbol(self):
        try:
            markets = self.exchange.load_markets()
            if self.symbol not in markets:
                raise ValueError(f"Symbol {self.symbol} not available on {self.exchange.name}")
        except Exception as e:
            logger.error(f"Failed to validate symbol: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def fetch_orderbook(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            ob = self.exchange.fetch_order_book(
                self.symbol,
                limit=self.depth
            )

            if not isinstance(ob, dict):
                raise ValueError("Invalid order book response type")

            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if not self._validate_orderbook_structure(bids, asks):
                logger.warning("Invalid order book structure received")
                return np.zeros((self.depth, 2)), np.zeros((self.depth, 2))
            bids_array = self._pad_orderbook_levels(bids[:self.depth], is_bid=True)
            asks_array = self._pad_orderbook_levels(asks[:self.depth], is_bid=False)

            return bids_array, asks_array

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching order book: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.error_count += 1
            raise

    def _validate_orderbook_structure(self, bids, asks) -> bool:
        try:

            if not isinstance(bids, list) or not isinstance(asks, list):
                return False

            def validate_levels(levels):
                if not levels:
                    return True
                for level in levels:
                    if (not isinstance(level, list) or
                        len(level) != 2):
                        return False
                    price, volume = level
                    if not (isinstance(price, (int, float)) and
                            isinstance(volume, (int, float))):
                        return False
                    if price <= 0 or volume < 0:
                        return False
                return True

            return validate_levels(bids) and validate_levels(asks)

        except Exception:
            return False

    def _pad_orderbook_levels(self, levels: list[list[float]], is_bid: bool) -> np.ndarray:

        if len(levels) >= self.depth:
            return np.array(levels[:self.depth], dtype=np.float64)

        padded = np.zeros((self.depth, 2), dtype=np.float64)

        num_levels = len(levels)
        if num_levels > 0:
            padded[:num_levels] = np.array(levels, dtype=np.float64)
            last_price = levels[-1][0]
            if last_price > 0:
                remaining = self.depth - num_levels
                # Generare vectorizată a prețurilor
                multiplier = 0.999 if is_bid else 1.001
                price_factors = np.power(multiplier, np.arange(1, remaining + 1))
                padded[num_levels:, 0] = last_price * price_factors


        return padded

    def compute_all_metrics(self, bids: np.ndarray, asks: np.ndarray) -> OrderBookMetrics:
        timestamp = time.time()
        bid_prices = bids[:, 0]
        bid_volumes = bids[:, 1]
        ask_prices = asks[:, 0]
        ask_volumes = asks[:, 1]
        bid_volume = np.sum(bid_volumes)
        ask_volume = np.sum(ask_volumes)

        # Delta și OBI - calcule atomice
        delta = bid_volume - ask_volume
        total_volume = bid_volume + ask_volume
        obi = delta / total_volume if total_volume > 0 else 0.0

        # CVD acumulare continuă (fără reset automat)
        # Doar dacă cvd_reset_seconds este finit (nu inf), verificăm reset
        if self.cvd_reset_seconds != float('inf'):
            if timestamp - self.cvd_reset_time > self.cvd_reset_seconds:
                self.cvd = 0.0
                self.cvd_reset_time = timestamp
                logger.info("CVD reset due to time interval")

        self.cvd += delta

        best_bid = bid_prices[0] if bid_prices[0] > 0 else 0.0
        best_ask = ask_prices[0] if ask_prices[0] > 0 else 0.0

        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0.0
        spread_percentage = (spread / best_bid * 100.0) if best_bid > 0 else 0.0

        mid_price = (best_bid + best_ask) * 0.5 if best_bid > 0 and best_ask > 0 else 0.0
        # VWAP bid/ask - primele 10 nivele
        n_vwap = 20
        bid_slice = bids[:n_vwap]
        ask_slice = asks[:n_vwap]

        bid_values = np.sum(bid_slice[:, 0] * bid_slice[:, 1])
        bid_total_vol = np.sum(bid_slice[:, 1])
        vwap_bid = bid_values / bid_total_vol if bid_total_vol > 0 else 0.0
        ask_values = np.sum(ask_slice[:, 0] * ask_slice[:, 1])
        ask_total_vol = np.sum(ask_slice[:, 1])
        vwap_ask = ask_values / ask_total_vol if ask_total_vol > 0 else 0.0
        min_depth = min(len(bid_volumes), len(ask_volumes))
        bid_vol_slice = bid_volumes[:min_depth]
        ask_vol_slice = ask_volumes[:min_depth]

        total_vol_per_level = bid_vol_slice + ask_vol_slice
        depth_imbalance = np.where(
            total_vol_per_level > 0,
            (bid_vol_slice - ask_vol_slice) / total_vol_per_level,
            0.0
        )

        order_book_pressure = self._calculate_order_book_pressure_vectorized(
            bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
        )

        liquidity_score = self._calculate_liquidity_score(bid_volumes, ask_volumes)

        volume_concentration = self._calculate_volume_concentration(bid_volumes, ask_volumes)

        order_book_entropy = self._calculate_order_book_entropy(bid_volumes, ask_volumes)

        weighted_mid_price = self._calculate_weighted_mid_price(
            bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
        )

        liquidity_imbalance_5, liquidity_imbalance_10 = self._calculate_liquidity_imbalance_ratios(
            bid_volumes, ask_volumes
        )

        smart_money_index = self._calculate_smart_money_index(bid_volumes, ask_volumes)

        support_strength, resistance_strength = self._calculate_support_resistance_strength(
            bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price
        )

        depth_asymmetry = self._calculate_depth_asymmetry(bid_volumes, ask_volumes)

        liquidity_fragmentation = self._calculate_liquidity_fragmentation(bid_volumes, ask_volumes)

        vwap_spread, vwap_spread_percentage = self._calculate_vwap_spread(
            vwap_bid, vwap_ask, mid_price
        )

        bid_depth_slope, ask_depth_slope = self._calculate_depth_curve_slopes(
            bid_volumes, ask_volumes
        )

        imbalance_velocity = self._calculate_imbalance_velocity(obi, timestamp)

        spread_volatility = self._calculate_spread_volatility(spread, timestamp)

        price_level_momentum = self._calculate_price_level_momentum(
            bid_volumes, ask_volumes, timestamp
        )

        metrics = OrderBookMetrics(
            timestamp=timestamp,
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            bid_volume=float(bid_volume),
            ask_volume=float(ask_volume),
            delta=float(delta),
            obi=float(obi),
            cvd=float(self.cvd),
            spread=float(spread),
            spread_percentage=float(spread_percentage),
            mid_price=float(mid_price),
            vwap_bid=float(vwap_bid),
            vwap_ask=float(vwap_ask),
            best_bid_volume=float(bid_volumes[0]),
            best_ask_volume=float(ask_volumes[0]),
            depth_imbalance=depth_imbalance,
            order_book_pressure=float(order_book_pressure),
            liquidity_score=float(liquidity_score),
            imbalance_velocity=float(imbalance_velocity),
            order_book_entropy=float(order_book_entropy),
            weighted_mid_price=float(weighted_mid_price),
            liquidity_imbalance_5=float(liquidity_imbalance_5),
            liquidity_imbalance_10=float(liquidity_imbalance_10),
            spread_volatility=float(spread_volatility),
            smart_money_index=float(smart_money_index),
            price_level_momentum=float(price_level_momentum),
            support_strength=float(support_strength),
            resistance_strength=float(resistance_strength),
            depth_asymmetry=float(depth_asymmetry),
            liquidity_fragmentation=float(liquidity_fragmentation),
            vwap_spread=float(vwap_spread),
            vwap_spread_percentage=float(vwap_spread_percentage),
            bid_depth_slope=float(bid_depth_slope),
            ask_depth_slope=float(ask_depth_slope)
        )

        return metrics

    def _calculate_order_book_pressure_vectorized(
        self, 
        bid_prices: np.ndarray, 
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray, 
        ask_volumes: np.ndarray,
        mid_price: float
    ) -> float:

        if mid_price <= 0:
            return 0.0

        n = 20  # Primele 20 nivele
        bid_prices_slice = bid_prices[:n]
        bid_volumes_slice = bid_volumes[:n]
        ask_prices_slice = ask_prices[:n]
        ask_volumes_slice = ask_volumes[:n]
        bid_distance_ratio = (mid_price - bid_prices_slice) / mid_price
        ask_distance_ratio = (ask_prices_slice - mid_price) / mid_price
        bid_decay = np.maximum(0, 1 - np.abs(bid_distance_ratio) * 10)
        ask_decay = np.maximum(0, 1 - np.abs(ask_distance_ratio) * 10)
        position_weights = 1.0 / (np.arange(n) + 1)

        bid_pressure = np.sum(bid_volumes_slice * bid_decay * position_weights[:len(bid_volumes_slice)])
        ask_pressure = np.sum(ask_volumes_slice * ask_decay * position_weights[:len(ask_volumes_slice)])

        total_pressure = bid_pressure + ask_pressure
        if total_pressure == 0:
            return 0.0

        normalized_pressure = (bid_pressure - ask_pressure) / total_pressure

        return normalized_pressure

    def _calculate_liquidity_score(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:

        top_n = 10
        top_bid_vol = np.sum(bid_volumes[:top_n])
        top_ask_vol = np.sum(ask_volumes[:top_n])
        liquidity = top_bid_vol + top_ask_vol
        return float(liquidity)

    def _calculate_volume_concentration(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
        top_n = 20
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        total_volume = total_bid + total_ask

        if total_volume == 0:
            return 0.0

        top_bid = np.sum(bid_volumes[:top_n])
        top_ask = np.sum(ask_volumes[:top_n])
        top_volume = top_bid + top_ask

        concentration = top_volume / total_volume
        return float(concentration)

    def _calculate_order_book_entropy(
        self, 
        bid_volumes: np.ndarray, 
        ask_volumes: np.ndarray
    ) -> float:
        all_volumes = np.concatenate([bid_volumes, ask_volumes])

        all_volumes = all_volumes[all_volumes > 0]

        if len(all_volumes) == 0:
            return 0.0

        total_volume = np.sum(all_volumes)
        probabilities = all_volumes / total_volume

        entropy = -np.sum(
            np.where(probabilities > 0, probabilities * np.log2(probabilities), 0)
        )

        max_entropy = np.log2(len(all_volumes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def _calculate_weighted_mid_price(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float
    ) -> float:

        if mid_price <= 0:
            return 0.0

        n = 20
        bid_liquidity = np.sum(bid_volumes[:n])
        ask_liquidity = np.sum(ask_volumes[:n])

        total_liquidity = bid_liquidity + ask_liquidity
        if total_liquidity == 0:
            return mid_price

        bid_vwap = np.sum(bid_prices[:n] * bid_volumes[:n]) / bid_liquidity if bid_liquidity > 0 else 0
        ask_vwap = np.sum(ask_prices[:n] * ask_volumes[:n]) / ask_liquidity if ask_liquidity > 0 else 0

        weighted_mid = (bid_vwap * ask_liquidity + ask_vwap * bid_liquidity) / total_liquidity

        return float(weighted_mid)

    def _calculate_liquidity_imbalance_ratios(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> tuple[float, float]:

        bid_liq_5 = np.sum(bid_volumes[:10])
        ask_liq_5 = np.sum(ask_volumes[:10])
        ratio_5 = bid_liq_5 / ask_liq_5 if ask_liq_5 > 0 else 0.0

        bid_liq_10 = np.sum(bid_volumes[:20])
        ask_liq_10 = np.sum(ask_volumes[:20])
        ratio_10 = bid_liq_10 / ask_liq_10 if ask_liq_10 > 0 else 0.0

        return float(ratio_5), float(ratio_10)

    def _calculate_smart_money_index(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> float:

        all_volumes = np.concatenate([bid_volumes, ask_volumes])
        all_volumes = all_volumes[all_volumes > 0]

        if len(all_volumes) < 5:
            return 0.0

        median_vol = np.median(all_volumes)
        max_vol = np.max(all_volumes)
        std_vol = np.std(all_volumes)

        if std_vol == 0:
            return 0.0

        z_score = (max_vol - median_vol) / std_vol

        smi = 1.0 / (1.0 + np.exp(-z_score))

        return float(smi)

    def _calculate_support_resistance_strength(
        self,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
        mid_price: float
    ) -> tuple[float, float]:
        if len(bid_volumes) > 0 and np.sum(bid_volumes) > 0:
            max_bid_vol = np.max(bid_volumes)
            avg_bid_vol = np.mean(bid_volumes[bid_volumes > 0])
            support_strength = max_bid_vol / avg_bid_vol if avg_bid_vol > 0 else 1.0
        else:
            support_strength = 0.0

        if len(ask_volumes) > 0 and np.sum(ask_volumes) > 0:
            max_ask_vol = np.max(ask_volumes)
            avg_ask_vol = np.mean(ask_volumes[ask_volumes > 0])
            resistance_strength = max_ask_vol / avg_ask_vol if avg_ask_vol > 0 else 1.0
        else:
            resistance_strength = 0.0

        return float(support_strength), float(resistance_strength)

    def _calculate_depth_asymmetry(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> float:
        bid_cumsum = np.cumsum(bid_volumes)
        ask_cumsum = np.cumsum(ask_volumes)

        bid_total = bid_cumsum[-1] if len(bid_cumsum) > 0 else 1.0
        ask_total = ask_cumsum[-1] if len(ask_cumsum) > 0 else 1.0

        bid_cumsum_norm = bid_cumsum / bid_total if bid_total > 0 else np.zeros_like(bid_cumsum)
        ask_cumsum_norm = ask_cumsum / ask_total if ask_total > 0 else np.zeros_like(ask_cumsum)

        n_bid = len(bid_volumes)
        n_ask = len(ask_volumes)

        if n_bid == 0 or n_ask == 0:
            return 0.0

        bid_gini = 1.0 - 2.0 * np.sum(bid_cumsum_norm) / n_bid if n_bid > 0 else 0.0

        ask_gini = 1.0 - 2.0 * np.sum(ask_cumsum_norm) / n_ask if n_ask > 0 else 0.0

        asymmetry = bid_gini - ask_gini

        asymmetry = np.clip(asymmetry, -1.0, 1.0)

        return float(asymmetry)

    def _calculate_liquidity_fragmentation(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> float:
        all_volumes = np.concatenate([bid_volumes, ask_volumes])
        all_volumes = all_volumes[all_volumes > 0]

        if len(all_volumes) < 2:
            return 0.0

        total_volume = np.sum(all_volumes)
        shares = all_volumes / total_volume
        hhi = np.sum(shares ** 2)
        n = len(all_volumes)
        hhi_min = 1.0 / n
        hhi_normalized = (hhi - hhi_min) / (1.0 - hhi_min) if n > 1 else 0.0
        fragmentation = 1.0 - hhi_normalized

        return float(fragmentation)

    def _calculate_vwap_spread(
        self,
        vwap_bid: float,
        vwap_ask: float,
        mid_price: float
    ) -> tuple[float, float]:
        if vwap_bid <= 0 or vwap_ask <= 0:
            return 0.0, 0.0

        vwap_spread_abs = vwap_ask - vwap_bid
        vwap_spread_pct = (vwap_spread_abs / mid_price * 100.0) if mid_price > 0 else 0.0

        return float(vwap_spread_abs), float(vwap_spread_pct)

    def _calculate_depth_curve_slopes(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> tuple[float, float]:
        bid_cumsum = np.cumsum(bid_volumes)
        ask_cumsum = np.cumsum(ask_volumes)

        if len(bid_cumsum) > 1:
            x_bid = np.arange(len(bid_cumsum))
            bid_slope, _ = np.polyfit(x_bid, bid_cumsum, 1)
            bid_slope = bid_slope / bid_cumsum[-1] if bid_cumsum[-1] > 0 else 0.0
        else:
            bid_slope = 0.0

        # Slope ask
        if len(ask_cumsum) > 1:
            x_ask = np.arange(len(ask_cumsum))
            ask_slope, _ = np.polyfit(x_ask, ask_cumsum, 1)
            # Normalizare
            ask_slope = ask_slope / ask_cumsum[-1] if ask_cumsum[-1] > 0 else 0.0
        else:
            ask_slope = 0.0

        return float(bid_slope), float(ask_slope)

    def _calculate_imbalance_velocity(self, current_obi: float, timestamp: float) -> float:
        if len(self.metrics_history) < 2:
            return 0.0
        previous_metrics = self.metrics_history[-1]
        previous_obi = previous_metrics.obi
        previous_timestamp = previous_metrics.timestamp
        delta_time = timestamp - previous_timestamp

        if delta_time <= 0:
            return 0.0

        # Delta OBI
        delta_obi = current_obi - previous_obi

        # Velocity = ΔOBI / Δtime
        velocity = delta_obi / delta_time

        return float(velocity)

    def _calculate_spread_volatility(self, current_spread: float, timestamp: float) -> float:

        if len(self.metrics_history) < 5:
            return 0.0

        window_size = min(20, len(self.metrics_history))
        recent_spreads = [m.spread for m in list(self.metrics_history)[-window_size:]]
        recent_spreads.append(current_spread)
        spread_volatility = float(np.std(recent_spreads))

        return spread_volatility

    def _calculate_price_level_momentum(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
        timestamp: float
    ) -> float:
        if len(self.metrics_history) < 2:
            return 0.0

        # Previous volumes
        previous_metrics = self.metrics_history[-1]
        prev_bid_volumes = previous_metrics.bids[:, 1]
        prev_ask_volumes = previous_metrics.asks[:, 1]

        # Calculare delta volumes
        min_bid_len = min(len(bid_volumes), len(prev_bid_volumes))
        min_ask_len = min(len(ask_volumes), len(prev_ask_volumes))

        # Delta bid
        bid_delta = np.sum(bid_volumes[:min_bid_len] - prev_bid_volumes[:min_bid_len])

        # Delta ask
        ask_delta = np.sum(ask_volumes[:min_ask_len] - prev_ask_volumes[:min_ask_len])
        total_volume = np.sum(bid_volumes) + np.sum(ask_volumes)

        if total_volume == 0:
            return 0.0

        # Momentum = (bid_delta - ask_delta) / total_volume
        # Normalizare la [-1, 1]
        momentum = (bid_delta - ask_delta) / total_volume
        momentum = np.clip(momentum, -1.0, 1.0)

        return float(momentum)

    def _calculate_order_book_pressure(self, bids, asks) -> float:
        logger.warning("Using deprecated _calculate_order_book_pressure. Use vectorized version.")
        return 0.0

    def update(self) -> OrderBookMetrics | None:
        try:

            bids, asks = self.fetch_orderbook()

            if bids.size == 0 or asks.size == 0:
                logger.warning("Empty order book received")
                return None
            metrics = self.compute_all_metrics(bids, asks)
            self.metrics_history.append(metrics)
            self.update_count += 1
            if self.update_count % 100 == 0:
                logger.info(f"Processed {self.update_count} updates. "
                          f"Errors: {self.error_count}. "
                          f"Current CVD: {self.cvd:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            self.error_count += 1
            return None

    def get_statistics(self) -> dict:
        return {
            "total_updates": self.update_count,
            "error_count": self.error_count,
            "success_rate": (self.update_count - self.error_count) / self.update_count 
                          if self.update_count > 0 else 0,
            "current_cvd": self.cvd,
            "history_size": len(self.metrics_history),
            "symbol": self.symbol,
            "depth": self.depth,
        }

    def reset_cvd(self):
        self.cvd = 0.0
        self.cvd_reset_time = time.time()
        logger.info("CVD manually reset")

    def get_historical_data(self, metric_name: str, window: int | None = None) -> list:
        if not hasattr(OrderBookMetrics, metric_name):
            raise ValueError(f"Metric {metric_name} not found")
        data = [getattr(metric, metric_name) for metric in self.metrics_history]
        if window:
            data = data[-window:]

        return data

class MultiSymbolOrderBookManager:
    def __init__(
        self,
        symbols: list[str] | None = None,
        depth: int = 25,
        max_history: int = 1000,
        cvd_reset_hours: int | None = None,  # None = unlimited capture
        exchange_name: str = "bybit"
    ):
        if symbols is None:
            symbols = ["BTC/USDT"]

        self.symbols = symbols
        self.depth = depth
        self.max_history = max_history
        self.cvd_reset_hours = cvd_reset_hours
        self.exchange_name = exchange_name
        self.engines: dict[str, EnhancedOrderBookEngine] = {}
        self._initialize_engines()

        logger.info(f"MultiSymbolOrderBookManager initialized with {len(self.symbols)} symbols")

    def _initialize_engines(self):
        for symbol in self.symbols:
            try:
                engine = EnhancedOrderBookEngine(
                    symbol=symbol,
                    depth=self.depth,
                    max_history=self.max_history,
                    cvd_reset_hours=self.cvd_reset_hours,
                    exchange_name=self.exchange_name
                )
                self.engines[symbol] = engine
                logger.info(f"Engine initialized for {symbol}")
            except Exception as e:
                logger.error(f"Failed to initialize engine for {symbol}: {e}")

    def add_symbol(self, symbol: str) -> bool:
        if symbol in self.engines:
            logger.warning(f"Symbol {symbol} already exists")
            return False

        try:
            engine = EnhancedOrderBookEngine(
                symbol=symbol,
                depth=self.depth,
                max_history=self.max_history,
                cvd_reset_hours=self.cvd_reset_hours,
                exchange_name=self.exchange_name
            )
            self.engines[symbol] = engine
            self.symbols.append(symbol)
            logger.info(f"Symbol {symbol} added successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to add symbol {symbol}: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        if symbol not in self.engines:
            logger.warning(f"Symbol {symbol} not found")
            return False

        del self.engines[symbol]
        self.symbols.remove(symbol)
        logger.info(f"Symbol {symbol} removed")
        return True

    def update_all(self) -> dict[str, OrderBookMetrics | None]:
        results = {}
        for symbol, engine in self.engines.items():
            try:
                metrics = engine.update()
                results[symbol] = metrics
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                results[symbol] = None

        return results

    def update_symbol(self, symbol: str) -> OrderBookMetrics | None:
        if symbol not in self.engines:
            logger.error(f"Symbol {symbol} not found")
            return None

        return self.engines[symbol].update()

    def get_engine(self, symbol: str) -> EnhancedOrderBookEngine | None:
        return self.engines.get(symbol)

    def get_all_statistics(self) -> dict[str, dict]:
        stats = {}
        for symbol, engine in self.engines.items():
            stats[symbol] = engine.get_statistics()
        return stats

    def reset_all_cvd(self):
        for engine in self.engines.values():
            engine.reset_cvd()
        logger.info("All CVDs reset")

    def get_available_symbols(self) -> list[str]:
        return self.symbols.copy()

if __name__ == "__main__":
    print("="*60)
    print("EXEMPLU 1: Single Symbol Engine")
    print("="*60)

    engine = EnhancedOrderBookEngine(
        symbol="BTC/USDT",
        depth=25,
        max_history=1000,
        cvd_reset_hours=24,
        exchange_name="bybit"
    )

    for i in range(5):
        metrics = engine.update()

        if metrics:
            print(f"\nUpdate {i + 1}:")
            print(f"  Symbol: {metrics.symbol}")
            print(f"  Delta: {metrics.delta:.2f}")
            print(f"  OBI: {metrics.obi:.4f}")
            print(f"  CVD: {metrics.cvd:.2f}")
            print(f"  Spread: {metrics.spread:.2f} ({metrics.spread_percentage:.3f}%)")
            print(f"  Mid Price: {metrics.mid_price:.2f}")
            print(f"  Order Book Pressure: {metrics.order_book_pressure:.4f}")
            print(f"  Liquidity Score: {metrics.liquidity_score:.2f}")
            print(f"  Volume Concentration: {metrics.volume_concentration:.4f}")

        time.sleep(1)  # Rate limiting


    print("\n" + "="*60)
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n" + "="*60)
    print("EX2: Multi Symbol Manager")
    print("="*60)


    manager = MultiSymbolOrderBookManager(
        symbols=["BTC/USDT", "ETH/USDT"],
        depth=25,
        exchange_name="bybit"
    )
    print("\nUpdate all symbols:")
    results = manager.update_all()

    for symbol, metrics in results.items():
        if metrics:
            print(f"\n{symbol}:")
            print(f"  Delta: {metrics.delta:.2f}")
            print(f"  CVD: {metrics.cvd:.2f}")
            print(f"  OBI: {metrics.obi:.4f}")
            print(f"  Pressure: {metrics.order_book_pressure:.4f}")

    print("\n" + "="*60)
    print("Statistics for all symbols:")
    all_stats = manager.get_all_statistics()
    for symbol, stats in all_stats.items():
        print(f"\n{symbol}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")