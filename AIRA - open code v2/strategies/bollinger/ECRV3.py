from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from functools import reduce
from datetime import datetime
from typing import Optional, Union
from freqtrade.persistence import Trade

# Correct import for timeframe_to_prev_date
from freqtrade.exchange import timeframe_to_prev_date

class ECRV3(IStrategy):
    """
    Enhanced Crash-Resistant (ECR) Martingale DCA Strategy (V2)

    Changes vs V1:
    - ✅ Enable SHORT signals (populate_entry_trend + populate_exit_trend)
    - ✅ Make custom_exit ATR TP/SL work for both long and short
    - ✅ Keep DCA logic for longs only (short DCA disabled by default for safety)

    Notes:
    - This strategy is still martingale/DCA style. Use with care.
    """

    INTERFACE_VERSION = 3
    can_short = True  # V2: enable shorts
    timeframe = '5m'
    startup_candle_count: int = 200

# ── ROI: realistic for 5m small-profit strategy ──────────
    minimal_roi = {
        "0": 0.99  # Disabled — custom_exit handles everything
    }
    use_custom_exits = True
    # ── Trailing: start early, trail tight ───────────────────
    #trailing_stop = True
    #trailing_stop_positive = 0.005          # trail 0.5%
    #trailing_stop_positive_offset = 0.008   # start trailing at 0.8%
    #trailing_only_offset_is_reached = True
    stoploss = -0.12  # Keep wide for martingale room

    use_exit_signal = False  # Rely primarily on custom_exit + trailing
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Position adjustment (DCA)
    position_adjustment_enable = True
    max_entry_position_adjustment = 3  # Max 3 safety orders → total 4 entries max

    # Martingale DCA settings
    martingale_factor = 2.0             # Classic doubling (tune between 1.5–2.5)
    dca_trigger_pct = -0.05             # Trigger DCA at -5% unrealized profit or worse
    dca_rsi_threshold = 32              # RSI must be ≤ this for DCA
    dca_volume_multiplier = 1.2         # Volume ≥ EMA × this
    max_total_exposure_multiplier = 15.0  # Hard cap: don't exceed ~15× initial stake total

    # ATR-based fixed take-profit / stop-loss
    atr_distance = 2.0                  # Risk distance multiplier
    risk_reward_ratio = 2.0             # Reward:risk ratio for TP

    # Plot configuration for nice charts in `freqtrade plot-dataframe`
    plot_config = {
        'main_plot': {
            'bb_upperband': {
                'color': 'orange',
                'fill_to': 'bb_lowerband',
                'fill_color': 'rgba(255, 165, 0, 0.15)'
            },
            'bb_lowerband': {'color': 'orange'},
            'bb_middleband': {'color': 'gray', 'linestyle': 'dash'},
            'ema50': {'color': 'purple'},
            'ema200': {'color': 'blue', 'linewidth': 2}
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'red'},
                'hline': [
                    {'value': 70, 'color': 'red', 'linestyle': 'dash'},
                    {'value': 30, 'color': 'green', 'linestyle': 'dash'}
                ]
            },
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
                'macdhist': {'type': 'bar', 'color': 'gray'}
            },
            "ROC": {
                'roc': {'color': 'purple'}
            },
            "ATR": {
                'atr': {'color': 'teal'}
            }
        }
    }
    # ── ADD HERE ─────────────────────────────────────────────
    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self.profit_peaks = {}
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return -1
        candle = dataframe.iloc[-1]
        atr_multiplier = 1.5 if candle['rsi'] > 60 else 2.0
        sl_offset = atr_multiplier * candle['atr']
        sl_price = trade.open_rate_avg - sl_offset
        return (sl_price - current_rate) / current_rate

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        actual_profit = trade.calc_profit_ratio(current_rate)
        current_profit = actual_profit
        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60

        # ── PROFIT PEAK TRACKING ─────────────────────────────
        trade_id = f"{pair}_{trade.open_date_utc.timestamp()}"
        if trade_id not in self.profit_peaks:
            self.profit_peaks[trade_id] = current_profit
        elif current_profit > self.profit_peaks[trade_id]:
            self.profit_peaks[trade_id] = current_profit
        profit_peak = self.profit_peaks[trade_id]

        def log_and_exit(reason):
            if trade_id in self.profit_peaks:
                del self.profit_peaks[trade_id]
            return reason

        # ── ATR ───────────────────────────────────────────────
        atr_pct = 0.008  # fallback for 5m
        try:
            if len(dataframe) > 1:
                last = dataframe.iloc[-2]
                atr_pct = max(0.002, min(last['atr'] / last['close'], 0.03))
        except:
            pass
        atr_unit = atr_pct * trade.leverage

        # ── ZONE 0: UNDER 1% ──────────────────────────────────
        if current_profit < 0.01:
            # Had profit, now pulling back — protect it
            if profit_peak >= 0.005 and current_profit < (profit_peak - 0.003):
                return log_and_exit(f"MicroTrail_Peak_{profit_peak*100:.1f}%")
            # Was green, now red — cut it
            if profit_peak >= 0.004 and current_profit < 0.0:
                return log_and_exit(f"MicroTrail_WentNeg_{profit_peak*100:.1f}%")
            # Stagnation
            if trade_duration_minutes >= 120 and profit_peak < 0.003:
                return log_and_exit(f"Stagnant_2HR")
            if trade_duration_minutes >= 360 and current_profit < 0.005:
                return log_and_exit(f"Stagnant_6HR_{current_profit*100:.1f}%")
            return None

        # ── ZONE 1: 1-2% ──────────────────────────────────────
        if current_profit < 0.02:
            pullback = 1.5 * atr_unit
            trail_stop = max(profit_peak - pullback, 0.005)
            if current_profit < trail_stop:
                return log_and_exit(f"Trail_1pct_Peak_{profit_peak*100:.1f}%")
            return None

        # ── ZONE 2: 2-4% ──────────────────────────────────────
        if current_profit < 0.04:
            pullback = 1.2 * atr_unit
            trail_stop = max(profit_peak - pullback, 0.010)
            if current_profit < trail_stop:
                return log_and_exit(f"Trail_2pct_Peak_{profit_peak*100:.1f}%")
            return None

        # ── ZONE 3: 4%+ ───────────────────────────────────────
        pullback = 1.0 * atr_unit
        trail_stop = max(profit_peak - pullback, 0.020)
        if current_profit < trail_stop:
            return log_and_exit(f"Trail_4pct_Peak_{profit_peak*100:.1f}%")
        return None
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = ta.BBANDS(
            dataframe['close'],
            timeperiod=20,
            nbdevup=2.5,
            nbdevdn=2.5,
            matype=0
        )
        dataframe['bb_upperband'] = bollinger[0]
        dataframe['bb_middleband'] = bollinger[1]
        dataframe['bb_lowerband'] = bollinger[2]

        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['volume_ema'] = ta.EMA(dataframe['volume'], timeperiod=20)
        
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # LONG entries
        long_conditions = [
            (dataframe['rsi'] < 40),
            (dataframe['close'] < dataframe['bb_lowerband'] * 1.02),
            (dataframe['macd'] > dataframe['macdsignal']),
            (dataframe['volume'] > dataframe['volume_ema'] * 0.8),
            (dataframe['volume'] > 0)
        ]
        long_signal = reduce(lambda x, y: x & y, long_conditions)
        dataframe.loc[long_signal, ['enter_long', 'enter_tag']] = (1, 'long_entry')

        # SHORT entries (V2)
        if self.can_short:
            short_conditions = [
                (dataframe['rsi'] > 60),
                (dataframe['close'] > dataframe['bb_upperband'] * 0.98),
                (dataframe['macd'] < dataframe['macdsignal']),
                (dataframe['volume'] > dataframe['volume_ema'] * 0.8),
                (dataframe['volume'] > 0)
            ]
            short_signal = reduce(lambda x, y: x & y, short_conditions)
            dataframe.loc[short_signal, ['enter_short', 'enter_tag']] = (1, 'short_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # Exit long when short signal fires (same candle)
        dataframe.loc[dataframe['enter_short'] == 1, 'exit_long'] = 1
        dataframe.loc[dataframe['enter_short'] == 1, 'exit_tag'] = 'Reversal_Short_Signal'

        # Exit short when long signal fires (same candle)
        dataframe.loc[dataframe['enter_long'] == 1, 'exit_short'] = 1
        dataframe.loc[dataframe['enter_long'] == 1, 'exit_tag'] = 'Reversal_Long_Signal'

        return dataframe


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """Martingale DCA: multiplicatively increase stake on drawdowns (LONG only in V2)."""
        # V2 safety: disable DCA for shorts
        if trade.is_short:
            return None

        if current_profit > self.dca_trigger_pct:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty or len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]

        if last_candle['rsi'] >= self.dca_rsi_threshold:
            return None
        if last_candle['volume'] <= last_candle['volume_ema'] * self.dca_volume_multiplier:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = len(filled_entries)

        if count_of_entries >= (self.max_entry_position_adjustment + 1):
            return None

        if count_of_entries > 0:
            last_entry = filled_entries[-1]
            previous_stake = last_entry.stake_amount
        else:
            previous_stake = trade.stake_amount

        dca_stake = previous_stake * self.martingale_factor

        total_stake_so_far = sum(e.stake_amount for e in filled_entries)
        projected_total = total_stake_so_far + dca_stake
        initial_stake = trade.stake_amount
        if projected_total > initial_stake * self.max_total_exposure_multiplier:
            return None

        if min_stake is not None and dca_stake < min_stake:
            dca_stake = min_stake
        if max_stake is not None and dca_stake > max_stake:
            dca_stake = max_stake

        return dca_stake

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        return proposed_stake