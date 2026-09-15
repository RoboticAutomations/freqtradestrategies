import logging
import talib.abstract as ta
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class NOTankAi_15_unlookahead_v5(IStrategy):
    """
    NOTankAi strategy - ULTRA SIMPLE VERSION - FIXED
    ** ZERO LOOKAHEAD BIAS - NO EXTREMA DETECTION **
    ** FIXED NUMPY/PANDAS SHIFT ISSUE **

    This version uses index-based shifting instead of .shift() method
    to avoid numpy/pandas compatibility issues.

    Trading Mode: SPOT ONLY (no leverage)
    Position Type: LONG ONLY
    Timeframe: 15 minutes
    """

    # Basic strategy settings
    exit_profit_only = True
    trailing_stop = False
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1
    process_only_new_candles = True
    can_short = False  # LONG ONLY
    use_exit_signal = True
    startup_candle_count: int = 100
    stoploss = -0.15
    timeframe = "15m"

    # Spot trading settings
    margin_mode = ""
    trading_mode = "spot"

    # Simple Parameters
    rsi_buy_threshold = IntParameter(
        20, 35, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(
        65, 80, default=70, space="buy", optimize=True)

    ema_short_period = IntParameter(
        8, 15, default=10, space="buy", optimize=True)
    ema_long_period = IntParameter(
        20, 50, default=30, space="buy", optimize=True)

    volume_multiplier = DecimalParameter(
        1.1, 2.0, default=1.5, space="buy", optimize=True)
    volume_period = IntParameter(
        10, 30, default=20, space="buy", optimize=True)

    # DCA Parameters
    safety_order_volume_scale = DecimalParameter(
        1.2, 2.0, default=1.5, space="buy", optimize=True)
    dca_delay_minutes = IntParameter(
        15, 60, default=30, space="buy", optimize=True)

    # Protection Parameters
    cooldown_lookback = IntParameter(
        6, 24, default=12, space="protection", optimize=True)
    stop_duration = IntParameter(
        24, 120, default=48, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(
        default=True, space="protection", optimize=True)

    # Performance tracking
    recent_trades = []
    recent_win_rate = 0.5
    consecutive_losses = 0

    # Simple ROI
    minimal_roi = {
        "0": 0.08,
        "30": 0.05,
        "60": 0.03,
        "120": 0.02,
        "240": 0.01,
        "480": 0.005,
        "720": 0
    }

    @property
    def protections(self):
        """Simple protection system"""
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })

        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 3,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })

        return prot

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Simple position sizing"""
        try:
            base_stake = proposed_stake / self.max_dca_multiplier

            # Reduce size after consecutive losses
            if self.consecutive_losses >= 2:
                base_stake *= 0.7
            elif self.consecutive_losses >= 3:
                base_stake *= 0.5

            return max(min_stake or 0, min(base_stake, max_stake))

        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return proposed_stake / self.max_dca_multiplier

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Simple exit confirmation"""
        profit_ratio = trade.calc_profit_ratio(rate)

        if exit_reason == "partial_exit" and profit_ratio < 0:
            return False

        if exit_reason == "stoploss":
            self.update_performance_metrics(profit_ratio)
            return True

        if exit_reason == "roi" and profit_ratio < 0.01:
            return False

        if profit_ratio > 0:
            self.update_performance_metrics(profit_ratio)

        return True

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
        **kwargs,
    ) -> Optional[float]:
        """Simple DCA logic"""
        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            count_of_entries = trade.nr_of_successful_entries

            trade_age_minutes = (
                current_time - trade.open_date_utc).total_seconds() / 60

            # Simple profit taking
            if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
                return -(trade.stake_amount / 4)

            if trade_age_minutes < self.dca_delay_minutes.value:
                return None

            # Simple DCA thresholds
            if current_profit > -0.08 and count_of_entries == 1:
                return None
            if current_profit > -0.18 and count_of_entries == 2:
                return None
            if count_of_entries >= 3:
                return None

            base_stake = filled_entries[0].cost
            dca_size = base_stake * self.safety_order_volume_scale.value

            return dca_size

        except Exception as e:
            logger.error(f"Error in adjust_trade_position: {e}")
            return None

    def leverage(
        self,
        pair: str,
        current_time: "datetime",
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        """Spot trading - no leverage"""
        return 1.0

    def safe_shift(self, data, shift_periods=1):
        """
        Safely shift data whether it's numpy array or pandas Series
        """
        if isinstance(data, np.ndarray):
            # For numpy arrays, create a new array with NaN padding
            result = np.empty_like(data, dtype=float)
            result[:] = np.nan
            if shift_periods > 0:
                result[shift_periods:] = data[:-shift_periods]
            else:
                result[:shift_periods] = data[-shift_periods:]
            return result
        else:
            # For pandas Series, use built-in shift
            return data.shift(shift_periods)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        ✅ ULTRA SIMPLE indicators with FIXED shifting and error handling
        Uses safe_shift method to handle both numpy and pandas data
        """

        try:
            # ✅ BASIC RSI
            rsi_raw = ta.RSI(dataframe, timeperiod=14)
            shifted_rsi = self.safe_shift(rsi_raw, 1)
            if shifted_rsi is not None:
                dataframe['rsi'] = shifted_rsi
            else:
                dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14).shift(1)

            # ✅ BASIC EMAs
            ema_short_raw = ta.EMA(
                dataframe, timeperiod=self.ema_short_period.value)
            ema_long_raw = ta.EMA(
                dataframe, timeperiod=self.ema_long_period.value)

            shifted_ema_short = self.safe_shift(ema_short_raw, 1)
            shifted_ema_long = self.safe_shift(ema_long_raw, 1)

            if shifted_ema_short is not None:
                dataframe['ema_short'] = shifted_ema_short
            else:
                dataframe['ema_short'] = ta.EMA(
                    dataframe, timeperiod=self.ema_short_period.value).shift(1)

            if shifted_ema_long is not None:
                dataframe['ema_long'] = shifted_ema_long
            else:
                dataframe['ema_long'] = ta.EMA(
                    dataframe, timeperiod=self.ema_long_period.value).shift(1)

            # ✅ BASIC SMA for trend
            sma_trend_raw = ta.SMA(dataframe, timeperiod=50)
            shifted_sma_trend = self.safe_shift(sma_trend_raw, 1)
            if shifted_sma_trend is not None:
                dataframe['sma_trend'] = shifted_sma_trend
            else:
                dataframe['sma_trend'] = ta.SMA(
                    dataframe, timeperiod=50).shift(1)

            # ✅ BASIC Volume analysis
            volume_sma_raw = ta.SMA(
                dataframe['volume'], timeperiod=self.volume_period.value)
            shifted_volume_sma = self.safe_shift(volume_sma_raw, 1)
            if shifted_volume_sma is not None:
                dataframe['volume_sma'] = shifted_volume_sma
            else:
                dataframe['volume_sma'] = ta.SMA(
                    dataframe['volume'], timeperiod=self.volume_period.value).shift(1)

            # Calculate volume ratio using shifted data
            volume_shifted = dataframe['volume'].shift(1)
            dataframe['volume_ratio'] = (
                volume_shifted / dataframe['volume_sma']).fillna(1)

            # ✅ BASIC MACD with error handling
            try:
                macd_result = ta.MACD(dataframe)
                if isinstance(macd_result, tuple) and len(macd_result) == 3:
                    macd_raw, macd_signal_raw, macd_hist_raw = macd_result

                    shifted_macd = self.safe_shift(macd_raw, 1)
                    shifted_macd_signal = self.safe_shift(macd_signal_raw, 1)
                    shifted_macd_hist = self.safe_shift(macd_hist_raw, 1)

                    dataframe['macd'] = shifted_macd if shifted_macd is not None else pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['macd_signal'] = shifted_macd_signal if shifted_macd_signal is not None else pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['macd_hist'] = shifted_macd_hist if shifted_macd_hist is not None else pd.Series(
                        index=dataframe.index, dtype=float)
                else:
                    # Fallback: create empty series
                    dataframe['macd'] = pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['macd_signal'] = pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['macd_hist'] = pd.Series(
                        index=dataframe.index, dtype=float)
                    logger.warning(
                        "MACD calculation returned unexpected format")
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}")
                dataframe['macd'] = pd.Series(
                    index=dataframe.index, dtype=float)
                dataframe['macd_signal'] = pd.Series(
                    index=dataframe.index, dtype=float)
                dataframe['macd_hist'] = pd.Series(
                    index=dataframe.index, dtype=float)

            # ✅ BASIC Bollinger Bands with error handling
            try:
                bb_result = ta.BBANDS(dataframe, timeperiod=20)
                if isinstance(bb_result, tuple) and len(bb_result) == 3:
                    bb_upper_raw, bb_middle_raw, bb_lower_raw = bb_result

                    shifted_bb_upper = self.safe_shift(bb_upper_raw, 1)
                    shifted_bb_middle = self.safe_shift(bb_middle_raw, 1)
                    shifted_bb_lower = self.safe_shift(bb_lower_raw, 1)

                    dataframe['bb_upper'] = shifted_bb_upper if shifted_bb_upper is not None else pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['bb_middle'] = shifted_bb_middle if shifted_bb_middle is not None else pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['bb_lower'] = shifted_bb_lower if shifted_bb_lower is not None else pd.Series(
                        index=dataframe.index, dtype=float)
                else:
                    # Fallback: create empty series
                    dataframe['bb_upper'] = pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['bb_middle'] = pd.Series(
                        index=dataframe.index, dtype=float)
                    dataframe['bb_lower'] = pd.Series(
                        index=dataframe.index, dtype=float)
                    logger.warning(
                        "BBANDS calculation returned unexpected format")
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}")
                dataframe['bb_upper'] = pd.Series(
                    index=dataframe.index, dtype=float)
                dataframe['bb_middle'] = pd.Series(
                    index=dataframe.index, dtype=float)
                dataframe['bb_lower'] = pd.Series(
                    index=dataframe.index, dtype=float)

            # ✅ BASIC ATR
            atr_raw = ta.ATR(dataframe, timeperiod=14)
            shifted_atr = self.safe_shift(atr_raw, 1)
            if shifted_atr is not None:
                dataframe['atr'] = shifted_atr
            else:
                dataframe['atr'] = ta.ATR(dataframe, timeperiod=14).shift(1)

            # ✅ BASIC trend determination (using shifted data)
            dataframe['trend_up'] = (
                (dataframe['ema_short'] > dataframe['ema_long']) &
                (dataframe['close'].shift(1) > dataframe['sma_trend'])
            )

            # ✅ BASIC price position (using shifted data)
            dataframe['price_above_ema'] = dataframe['close'].shift(
                1) > dataframe['ema_short']
            dataframe['price_below_bb_lower'] = dataframe['close'].shift(
                1) < dataframe['bb_lower']
            dataframe['price_above_bb_upper'] = dataframe['close'].shift(
                1) > dataframe['bb_upper']

            # Fill any NaN values
            dataframe = dataframe.ffill().fillna(0)

        except Exception as e:
            logger.error(f"Error in populate_indicators: {e}")
            # Return dataframe with basic indicators as fallback
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14).shift(1)
            dataframe['ema_short'] = ta.EMA(
                dataframe, timeperiod=self.ema_short_period.value).shift(1)
            dataframe['ema_long'] = ta.EMA(
                dataframe, timeperiod=self.ema_long_period.value).shift(1)
            dataframe = dataframe.ffill().fillna(0)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        ✅ ULTRA SIMPLE entry logic with volume checks using shift(2)
        """

        # ✅ STRATEGY 1: Simple RSI Oversold
        df.loc[
            (
                (df['rsi'] < self.rsi_buy_threshold.value) &
                (df['volume_ratio'] > self.volume_multiplier.value) &
                (df['volume'].shift(2) > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "RSI_Oversold")

        # ✅ STRATEGY 2: Trend Following
        df.loc[
            (
                (df['trend_up']) &
                (df['rsi'] < 45) &
                (df['price_above_ema']) &
                (df['volume_ratio'] > 1.2) &
                (df['volume'].shift(2) > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "Trend_Follow")

        # ✅ STRATEGY 3: Bollinger Band Bounce
        df.loc[
            (
                (df['price_below_bb_lower']) &
                (df['rsi'] < 35) &
                (df['volume_ratio'] > 1.3) &
                (df['volume'].shift(2) > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "BB_Bounce")

        # ✅ STRATEGY 4: MACD Bullish
        df.loc[
            (
                (df['macd'] > df['macd_signal']) &
                (df['macd_hist'] > 0) &
                (df['rsi'] < 50) &
                (df['trend_up']) &
                (df['volume_ratio'] > 1.1) &
                (df['volume'].shift(2) > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "MACD_Bull")

        # ✅ STRATEGY 5: EMA Cross
        df.loc[
            (
                (df['ema_short'] > df['ema_long']) &
                (df['ema_short'].shift(1) <= df['ema_long'].shift(1)) &
                (df['rsi'] < 60) &
                (df['volume_ratio'] > 1.2) &
                (df['volume'].shift(2) > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "EMA_Cross")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        ✅ ULTRA SIMPLE exit logic with volume checks using shift(2)
        """

        # ✅ EXIT 1: Simple RSI Overbought
        df.loc[
            (
                (df['rsi'] > self.rsi_sell_threshold.value) &
                (df['volume'].shift(2) > 0)
            ),
            ["exit_long", "exit_tag"]
        ] = (1, "RSI_Overbought")

        # ✅ EXIT 2: Trend Change
        df.loc[
            (
                (~df['trend_up']) &
                (df['rsi'] > 60) &
                (df['volume'].shift(2) > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "Trend_Change")

        # ✅ EXIT 3: Bollinger Band Top
        df.loc[
            (
                (df['price_above_bb_upper']) &
                (df['rsi'] > 65) &
                (df['volume'].shift(2) > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "BB_Top")

        # ✅ EXIT 4: MACD Bearish
        df.loc[
            (
                (df['macd'] < df['macd_signal']) &
                (df['macd_hist'] < 0) &
                (df['rsi'] > 55) &
                (df['volume'].shift(2) > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "MACD_Bear")

        # ✅ EXIT 5: EMA Cross Down
        df.loc[
            (
                (df['ema_short'] < df['ema_long']) &
                (df['ema_short'].shift(1) >= df['ema_long'].shift(1)) &
                (df['rsi'] > 50) &
                (df['volume'].shift(2) > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "EMA_Cross_Down")

        return df

    def update_performance_metrics(self, trade_result: float):
        """Update performance tracking"""
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)

        wins = sum(1 for trade in self.recent_trades if trade > 0)
        self.recent_win_rate = wins / \
            len(self.recent_trades) if self.recent_trades else 0.5

        if trade_result < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
