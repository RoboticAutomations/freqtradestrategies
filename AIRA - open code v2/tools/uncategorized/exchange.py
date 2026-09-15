"""
ReplayExchange — thin Exchange subclass that replaces all live data I/O.

Patch surface (methods overridden here):
  _init_ccxt          → return MagicMock so no real exchange connects
  reload_markets      → no-op (called every process() loop)
  validate_config     → no-op (pairs/timeframes validated by data_store)
  exchange_has        → returns True only for capabilities we implement
  refresh_latest_ohlcv→ serve from ReplayDataStore gated by VirtualClock
  fetch_ticker        → synthesise from last known candle close ± spread
  fetch_l2_order_book → synthesise bid/ask used by _dry_is_price_crossed()
  get_funding_fees    → 0.0  (known divergence #1 — see design doc)
  set_leverage        → no-op
  set_margin_mode     → no-op
  get_positions       → {} (futures position fetching not needed in dry-run)

Everything else — dry_run order creation/filling, fee calculation, DCA,
virtual wallet — is inherited unchanged from Exchange.
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock

from freqtrade.exchange import Exchange

from .clock import VirtualClock
from .data_store import ReplayDataStore

logger = logging.getLogger(__name__)

_SUPPORTED_CAPABILITIES = {
    "fetchL2OrderBook",
    "fetchOHLCV",
    "fetchTicker",
    "createOrder",
    "cancelOrder",
    "fetchOrder",
    "fetchOrders",
    "fetchClosedOrders",
    "fetchBalance",
    "setLeverage",
    "setMarginMode",
}


class ReplayExchange(Exchange):

    def __init__(
        self,
        config: dict,
        store: ReplayDataStore,
        clock: VirtualClock,
        slippage_pct: float = 0.0005,
    ) -> None:
        # Store before super().__init__ because _init_ccxt is called from there
        self._replay_store = store
        self._replay_clock = clock
        self._slippage_pct = slippage_pct
        super().__init__(config, validate=False)
        # Pre-populate markets so get_markets() / verify_whitelist() works
        self._markets = {pair: self._make_market(pair) for pair in store._pairs}

    @staticmethod
    def _make_market(pair: str) -> dict:
        """Minimal market dict satisfying Freqtrade's market structure."""
        base, quote = pair.split("/")[0], pair.split("/")[1].split(":")[0]
        return {
            "id": pair.replace("/", "").replace(":", ""),
            "symbol": pair,
            "base": base,
            "quote": quote,
            "active": True,
            "spot": False,
            "swap": True,
            "future": True,
            "linear": True,
            "type": "swap",
            "contract": True,
            "contractSize": 1.0,
            "taker": 0.0002,
            "maker": 0.0002,
            "precision": {"amount": 8, "price": 2},
            "limits": {
                "amount": {"min": 0.00001, "max": None},
                "cost": {"min": 1.0, "max": None},
                "price": {"min": None, "max": None},
                "market": {"min": 0, "max": None},
                "leverage": {"min": 1, "max": None},
            },
            "info": {},
        }

    # ------------------------------------------------------------------
    # Mock ccxt so no network connection is attempted
    # ------------------------------------------------------------------

    def _init_ccxt(self, exchange_conf, is_sync=True, ccxt_config=None):
        mock = MagicMock()
        mock.options = {}
        mock.markets = {}
        mock.name = exchange_conf.get("name", "binance")
        mock.id = exchange_conf.get("name", "binance").lower()
        mock.timeframes = {tf: tf for tf in ["1m", "5m", "15m", "1h", "4h"]}
        mock.precisionMode = 2  # DECIMAL_PLACES — read directly by Exchange.__init__
        mock.describe.return_value = {
            "has": {cap: True for cap in _SUPPORTED_CAPABILITIES},
            "timeframes": mock.timeframes,
            "precisionMode": 2,
        }
        return mock

    # ------------------------------------------------------------------
    # No-ops that would otherwise hit the live exchange
    # ------------------------------------------------------------------

    def reload_markets(self, force: bool = False, **kwargs) -> None:
        pass

    def validate_config(self, config: dict) -> None:
        pass

    def exchange_has(self, endpoint: str) -> bool:
        return endpoint in _SUPPORTED_CAPABILITIES

    def get_markets(self, base_currencies=None, quote_currencies=None, tradable_only=True,
                    active_only=True, spot_only=False, margin_only=False, futures_only=False):
        return self._markets

    def get_positions(self, *args, **kwargs):
        return {}

    # ------------------------------------------------------------------
    # OHLCV data feed — replaces live ccxt fetch
    # ------------------------------------------------------------------

    def refresh_latest_ohlcv(
        self,
        pair_list: list,
        *,
        since_ms: int | None = None,
        cache: bool = True,
        drop_incomplete: bool | None = None,
    ) -> dict:
        from freqtrade.enums import CandleType
        now = self._replay_clock.now()
        results: dict = {}
        for item in pair_list:
            pair, tf, c_type = item
            df = self._replay_store.get_candles(pair, tf, up_to=now)
            if cache and not df.empty:
                self._klines[(pair, tf, c_type)] = df
                # Strategies calling get_pair_dataframe() without candle_type resolve
                # to CandleType.SPOT (empty string). Mirror data there so informative
                # pairs (e.g. BTC/USDT:USDT 15m) are found regardless of how the
                # strategy requests them.
                if c_type != CandleType.SPOT:
                    self._klines[(pair, tf, CandleType.SPOT)] = df
            results[(pair, tf, c_type)] = df
        return results

    # ------------------------------------------------------------------
    # Price feeds — drive entry/exit pricing and dry-run order fills
    # ------------------------------------------------------------------

    def fetch_ticker(self, pair: str) -> dict:
        price = self._replay_store.get_last_price(pair, self._replay_clock.now())
        half_spread = price * self._slippage_pct / 2
        ts = int(self._replay_clock.now().timestamp() * 1000)
        return {
            "symbol": pair,
            "last": price,
            "bid": price - half_spread,
            "ask": price + half_spread,
            "high": price,
            "low": price,
            "open": price,
            "close": price,
            "baseVolume": 0.0,
            "quoteVolume": 0.0,
            "timestamp": ts,
            "datetime": self._replay_clock.now().isoformat(),
            "info": {},
        }

    def fetch_l2_order_book(self, pair: str, limit: int = 1) -> dict:
        """
        Synthetic order book consumed by _dry_is_price_crossed() for limit-order fills.
        ask = last_close + spread/2  →  buy limit fills when limit_price >= ask
        bid = last_close - spread/2  →  sell limit fills when limit_price <= bid
        """
        price = self._replay_store.get_last_price(pair, self._replay_clock.now())
        half_spread = price * self._slippage_pct / 2
        ts = int(self._replay_clock.now().timestamp() * 1000)
        return {
            "asks": [[price + half_spread, 999_999.0]],
            "bids": [[price - half_spread, 999_999.0]],
            "timestamp": ts,
            "datetime": self._replay_clock.now().isoformat(),
            "nonce": ts,
        }

    # ------------------------------------------------------------------
    # Intra-candle order fill (stop-loss / take-profit accuracy)
    # ------------------------------------------------------------------

    def check_dry_limit_order_filled(
        self,
        order: dict,
        immediate: bool = False,
        orderbook: dict | None = None,
    ) -> dict:
        """
        For deferred checks (existing open orders), replace the close±spread
        synthetic order book with the last completed candle's high/low so that
        stop-losses and limit exits trigger whenever the candle's range touched
        the order price — matching backtesting behaviour.

        For immediate checks (new orders just placed), delegate to the parent
        so that market orders fill at the current close price as expected.
        """
        if immediate or order.get("status") == "closed":
            return super().check_dry_limit_order_filled(
                order, immediate=immediate, orderbook=orderbook
            )

        pair = order["symbol"]
        tf = self._config.get("timeframe", "1h")
        candle = self._replay_store.get_candle_ohlc(pair, tf, self._replay_clock.now())
        if candle is None:
            return super().check_dry_limit_order_filled(
                order, immediate=False, orderbook=orderbook
            )

        low, high = candle["low"], candle["high"]
        half_spread = low * self._slippage_pct / 2

        # Intra-candle book:
        #   ask = candle low  — market fell this low  → fills buy limits & sell stops
        #   bid = candle high — market rose this high → fills sell limits & buy stops
        candle_book: dict = {
            "asks": [[low + half_spread, 999_999.0]],
            "bids": [[high - half_spread, 999_999.0]],
            "timestamp": int(self._replay_clock.now().timestamp() * 1000),
            "datetime": self._replay_clock.now().isoformat(),
            "nonce": 0,
        }

        # Stoploss orders need special fill-price handling: the parent would use
        # get_dry_market_fill_price(bids=[high]) which fills at candle-high — wrong.
        # Fill at order["price"] instead (the stop price with slippage already applied).
        if order.get("ft_order_type") == "stoploss" and order.get("status") != "closed":
            from freqtrade.misc import safe_value_fallback
            stop_trigger = safe_value_fallback(
                order, self._ft_has["stop_price_prop"], "price"
            )
            crossed = self._dry_is_price_crossed(
                pair, order["side"], stop_trigger, candle_book, is_stop=True
            )
            if crossed:
                fill_price = order["price"]
                order.update({
                    "status": "closed",
                    "filled": order["amount"],
                    "remaining": 0,
                    "average": fill_price,
                    "cost": order["amount"] * fill_price,
                })
                self.add_dry_order_fee(pair, order, "maker")
            return order

        # Regular limit orders: use candle book for crossing; fill price stays at order["price"]
        return super().check_dry_limit_order_filled(
            order, immediate=False, orderbook=candle_book
        )

    # ------------------------------------------------------------------
    # Futures no-ops
    # ------------------------------------------------------------------

    def get_fee(
        self,
        symbol: str,
        order_type: str = "",
        side: str = "",
        amount: float = 1,
        price: float = 1,
        taker_or_maker: str = "maker",
    ) -> float:
        return 0.0002  # Binance futures taker/maker fee

    def get_funding_fees(
        self, pair: str, amount: float, is_short: bool, open_date: datetime
    ) -> float:
        return 0.0

    def get_max_leverage(self, pair: str, stake_amount: float | None) -> float:
        return 125.0  # Binance futures max; lets strategy.leverage() value pass through unchanged

    def set_leverage(
        self, leverage: float, pair: str | None = None, accept_fail: bool = False
    ) -> None:
        pass

    def set_margin_mode(
        self,
        pair: str,
        marginMode,
        accept_fail: bool = False,
        params: dict | None = None,
    ) -> None:
        pass
