from datetime import datetime
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade


class RSI_Futures(IStrategy):

    INTERFACE_VERSION = 3

    _MAX_PROFIT_KEY = "max_profit"

    can_short = True

    timeframe = "4h"

    process_only_new_candles = True

    startup_candle_count = 100

    stoploss = -0.50

    use_exit_signal = True

    position_adjustment_enable = True

    max_entry_position_adjustment = 2

    minimal_roi = {
        "0": 100
    }

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        dataframe["donchian_upper"] = (
            dataframe["high"].rolling(20).max()
        )
        dataframe["donchian_lower"] = (
            dataframe["low"].rolling(20).min()
        )

        dataframe["exit_lower"] = (
            dataframe["low"].rolling(10).min()
        )
        dataframe["exit_upper"] = (
            dataframe["high"].rolling(10).max()
        )

        dataframe["volume_ma"] = (
            dataframe["volume"].rolling(20).mean()
        )
        dataframe["volume_ratio"] = (
            dataframe["volume"] / dataframe["volume_ma"].replace(0, 1)
        )

        dataframe["ema50_slope"] = (
            dataframe["ema50"] - dataframe["ema50"].shift(3)
        ) / dataframe["ema50"].shift(3)

        return dataframe

    def populate_entry_trend(self, dataframe, metadata):

        long_conditions = (
            (dataframe["ema50"] > dataframe["ema200"])
            &
            (dataframe["adx"] > 25)
            &
            (dataframe["close"] > dataframe["donchian_upper"].shift(1))
            &
            (dataframe["volume_ratio"] > 1.0)
        )

        dataframe.loc[
            long_conditions,
            ["enter_long", "enter_tag"]
        ] = (1, "donchian_long")

        short_conditions = (
            (dataframe["ema50"] < dataframe["ema200"])
            &
            (dataframe["ema50_slope"] < 0)
            &
            (dataframe["adx"] > 25)
            &
            (dataframe["close"] < dataframe["donchian_lower"].shift(1))
            &
            (dataframe["volume_ratio"] > 1.0)
            &
            (dataframe["rsi"] < 50)
        )

        dataframe.loc[
            short_conditions,
            ["enter_short", "enter_tag"]
        ] = (1, "donchian_short")

        return dataframe

    def populate_exit_trend(self, dataframe, metadata):

        dataframe.loc[
            (dataframe["close"] < dataframe["exit_lower"].shift(1)),
            "exit_long"
        ] = 1

        dataframe.loc[
            (dataframe["close"] > dataframe["exit_upper"].shift(1)),
            "exit_short"
        ] = 1

        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        if side == "long":
            return min(5.0, max_leverage)
        else:
            return min(1.0, max_leverage)

    def custom_stake_amount(self, current_time, current_rate, proposed_stake, min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        pair = kwargs.get("pair", "")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return max_stake * 0.05 if side == "long" else max_stake * 0.02
        last = dataframe.iloc[-1]
        atr_pct = float(last.get("atr_pct", 0.02) or 0.02)

        if side == "long":
            pct = 0.05
        else:
            pct = 0.02

        if atr_pct > 0.04:
            pct *= 0.75

        stake = max_stake * max(0.02, min(pct, 0.05))
        min_allowed = max(min_stake or 0, max_stake * 0.02)
        return max(min_allowed, min(stake, max_stake * 0.05))

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs
    ) -> Optional[float]:
        """
        Profit-based pyramiding: only add when already in profit.
        Long only (shorts have minimal stake, not worth pyramiding).
        """
        entries = trade.nr_of_successful_entries

        # Only pyramid longs
        if trade.trade_direction != "long":
            return None

        # Pyramid 1: at +10% profit, add 30% more
        if entries == 1 and current_profit > 0.10:
            add_pct = 0.30
            add_stake = trade.stake_amount * add_pct
            # Ensure min_stake requirement
            if min_stake and add_stake < min_stake:
                return None
            # Cap additional stake at 3% of wallet (per-entry 2-5% requirement)
            max_additional = max_stake * 0.03
            add_stake = min(add_stake, max_additional)
            return (add_stake, "pyramid_10pct")

        # Pyramid 2: at +20% profit, add 25% more
        if entries == 2 and current_profit > 0.20:
            add_pct = 0.25
            add_stake = trade.stake_amount * add_pct
            if min_stake and add_stake < min_stake:
                return None
            max_additional = max_stake * 0.03
            add_stake = min(add_stake, max_additional)
            return (add_stake, "pyramid_20pct")

        return None

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        max_profit = trade.get_custom_data(
            key=self._MAX_PROFIT_KEY, default=current_profit
        )
        if current_profit > max_profit:
            max_profit = current_profit
            trade.set_custom_data(key=self._MAX_PROFIT_KEY, value=max_profit)

        if max_profit > 0.08:
            if (max_profit - current_profit) > 0.03:
                return "profit_pullback_3pct"
        if max_profit > 0.15:
            if (max_profit - current_profit) > 0.05:
                return "profit_pullback_5pct"
        if max_profit > 0.25:
            if (max_profit - current_profit) > 0.08:
                return "profit_pullback_8pct"
        return None
