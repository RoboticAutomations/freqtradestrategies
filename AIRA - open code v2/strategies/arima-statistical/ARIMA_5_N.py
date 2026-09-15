# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from pandas import DataFrame
import warnings
import pandas as pd
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, informative
from freqtrade.strategy import DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter
import technical.indicators as ftt
import math
import logging
from scipy.signal import find_peaks, find_peaks_cwt
import warnings
from math import ceil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error
import time
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.set_option('display.float_format', lambda x: '%.7f' % x)
logger = logging.getLogger(__name__)

class ARIMA_5_N(IStrategy):
    INTERFACE_VERSION = 3
    # Stoploss:
    stoploss = -0.99
    # Trailing stop:
    use_custom_stoploss = True
    # Initialize dicts for arima storage
    last_run_time = {}
    arima_model = {}
    last_run_time_1h = {}
    arima_model_1h = {}
    last_run_time_4h = {}
    arima_model_4h = {}
    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    can_short: bool = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    ## Optional order time in force.
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}
    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count = 400
    process_only_new_candles = True
    # Custom Entry
    last_entry_price = None
    # Hyper-opt parameters
    base_nb_candles_buy = IntParameter(150, 200, default=184, space='buy', optimize=True, load=True)
    up = DecimalParameter(low=1.02, high=1.025, default=1.02, decimals=3, space='buy', optimize=True, load=True)
    dn = DecimalParameter(low=0.983, high=0.987, default=0.984, decimals=3, space='buy', optimize=True, load=True)
    increment = DecimalParameter(low=1.0005, high=1.001, default=1.0007, decimals=4, space='buy', optimize=True, load=True)
    atr_length = IntParameter(5, 30, default=5, space='buy', optimize=True, load=True)
    window = IntParameter(10, 30, default=16, space='buy', optimize=True, load=True)
    window_1h = IntParameter(10, 30, default=8, space='buy', optimize=True, load=True)
    window_4h = IntParameter(2, 30, default=2, space='buy', optimize=True, load=True)
    x = DecimalParameter(low=1.2, high=1.75, default=1.6, decimals=2, space='buy', optimize=True, load=True)
    x_1h = DecimalParameter(low=1.2, high=1.75, default=1.5, decimals=2, space='buy', optimize=True, load=True)
    x_4h = DecimalParameter(low=1.2, high=1.75, default=1.3, decimals=2, space='buy', optimize=True, load=True)
    ### trailing stop loss optimiziation ###
    tsl_target3 = DecimalParameter(low=0.1, high=0.15, default=0.15, decimals=2, space='sell', optimize=True, load=True)
    ts3 = DecimalParameter(low=0.025, high=0.04, default=0.035, decimals=3, space='sell', optimize=True, load=True)
    tsl_target2 = DecimalParameter(low=0.06, high=0.1, default=0.1, decimals=3, space='sell', optimize=True, load=True)
    ts2 = DecimalParameter(low=0.015, high=0.03, default=0.02, decimals=3, space='sell', optimize=True, load=True)
    tsl_target1 = DecimalParameter(low=0.04, high=0.08, default=0.06, decimals=3, space='sell', optimize=True, load=True)
    ts1 = DecimalParameter(low=0.01, high=0.016, default=0.013, decimals=3, space='sell', optimize=True, load=True)
    tsl_target0 = DecimalParameter(low=0.03, high=0.06, default=0.04, decimals=3, space='sell', optimize=True, load=True)
    ts0 = DecimalParameter(low=0.005, high=0.012, default=0.01, decimals=3, space='sell', optimize=True, load=True)
    moon = IntParameter(80, 90, default=85, space='sell', optimize=True)

    @property
    def protections(self):
        return [{'method': 'CooldownPeriod', 'stop_duration_candles': 5}, {'method': 'MaxDrawdown', 'lookback_period_candles': 48, 'trade_limit': 20, 'stop_duration_candles': 4, 'max_allowed_drawdown': 0.2}, {'method': 'StoplossGuard', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 2, 'only_per_pair': False}, {'method': 'LowProfitPairs', 'lookback_period_candles': 6, 'trade_limit': 2, 'stop_duration_candles': 60, 'required_profit': 0.02}, {'method': 'LowProfitPairs', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 2, 'required_profit': 0.01}]
    ### Trailing Stop ###

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        if current_candle['max_l'] > 0.0035:
            if current_profit > self.tsl_target3.value:
                return self.ts3.value
            if current_profit > self.tsl_target2.value:
                return self.ts2.value
            if current_profit > self.tsl_target1.value:
                return self.ts1.value
            if current_profit > self.tsl_target0.value:
                return self.ts0.value
        elif current_profit > self.tsl_target0.value:
            return 0.99
        return self.stoploss

    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}")
        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value  # Increment by 0.2%
            logger.info(f'{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.')
        # Update the last entry price
        self.last_entry_price = entry_price
        return entry_price

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        if exit_reason == 'roi' and last_candle['min_l'] > last_candle['max_l'] * 3:
            return False
        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        # Apply rolling window operation to the 'OHLC4' column
        rolling_window = dataframe['OHLC4'].rolling(self.window.value)  # 5.25 hrs
        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        dataframe['move_mean'] = dataframe['move'].mean()
        dataframe['move_mean_x'] = dataframe['move'].mean() * self.x.value
        dataframe['atr_pcnt'] = ta.ATR(dataframe, timeperiod=self.atr_length.value) / dataframe['OHLC4']
        dataframe['vol_z_score'] = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        dataframe['vol_anomaly'] = np.where(dataframe['vol_z_score'] > 3, 1, 0)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        dataframe['sma'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}']
        dataframe['sma_up'] = dataframe['sma'] * self.up.value
        dataframe['sma_dn'] = dataframe['sma'] * self.dn.value
        dataframe['max_l'] = dataframe['OHLC4'].rolling(120).max() / dataframe['OHLC4'] - 1
        dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(120).min() / dataframe['OHLC4'] - 1)
        dataframe['max'] = dataframe['OHLC4'].rolling(4).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(4).min() / dataframe['OHLC4'] - 1)

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # Hammer: values [0, 100]
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # Inverted Hammer: values [0, 100]
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # Dragonfly Doji: values [0, 100]
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # Piercing Line: values [0, 100]
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # Morningstar: values [0, 100]
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # Three White Soldiers: values [0, 100]
        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # Hanging Man: values [0, 100]
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # Shooting Star: values [0, 100]
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # Gravestone Doji: values [0, 100]
        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # Dark Cloud Cover: values [0, 100]
        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # Evening Doji Star: values [0, 100]
        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # Evening Star: values [0, 100]
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # Three Line Strike: values [0, -100, 100]
        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # Spinning Top: values [0, -100, 100]
        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # Engulfing: values [0, -100, 100]
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # Harami: values [0, -100, 100]
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # Three Outside Up/Down: values [0, -100, 100]
        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # Three Inside Up/Down: values [0, -100, 100]
        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condition1 = (dataframe['move'] >= dataframe['move_mean']) & (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma_dn']) & (dataframe['sma_dn'].shift() > dataframe['sma_dn']) & (dataframe['volume'] > 0)
        dataframe.loc[condition1, 'enter_long'] = 1
        dataframe.loc[condition1, 'enter_tag'] = 'long: Up Trend Soon below sma_dn'
        condition2 = (dataframe['move'] >= dataframe['move_mean_x']) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma_dn']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['volume'] > 0)
        dataframe.loc[condition2, 'enter_long'] = 1
        dataframe.loc[condition2, 'enter_tag'] = 'long: Move Mean Fib below sma_dn'
        condition3 = (dataframe['move'] >= dataframe['move_mean']) & (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['OHLC4'] < dataframe['sma']) & (dataframe['volume'] > 0)
        dataframe.loc[condition3, 'enter_long'] = 1
        dataframe.loc[condition3, 'enter_tag'] = 'long: Up Trend Soon below sma'
        condition4 = (dataframe['move'] >= dataframe['move_mean_x']) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['volume'] > 0)
        dataframe.loc[condition4, 'enter_long'] = 1
        dataframe.loc[condition4, 'enter_tag'] = 'long: Move Mean Fib below sma'
        condition5150 = (dataframe['move'] >= dataframe['move_mean']) & (dataframe['min'] < dataframe['max']) & (dataframe['OHLC4'] > dataframe['sma']) & (dataframe['sma_up'].shift() < dataframe['sma']) & (dataframe['volume'] > 0)
        dataframe.loc[condition5150, 'enter_long'] = 1
        dataframe.loc[condition5150, 'enter_tag'] = 'long: Hope this works...'

        condition5 =  (dataframe['move'] >= dataframe['move_mean']) & (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) & (dataframe['min'] > dataframe['max']) & (dataframe['min_l'] > dataframe['max_l']) & (dataframe['volume'] > 0)
        dataframe.loc[condition5, 'enter_short'] = 1
        dataframe.loc[condition5, 'enter_tag'] = 'short: Down Trend Soon'
        condition6 =  (dataframe['move'] >= dataframe['move_mean_x']) & (dataframe['move'].shift(3) >= dataframe['move_mean_x'].shift(3)) & (dataframe['min'] > dataframe['max']) & (dataframe['min_l'] > dataframe['max_l']) & (dataframe['volume'] > 0)
        dataframe.loc[condition6, 'enter_short'] = 1
        dataframe.loc[condition6, 'enter_tag'] = 'short: Move Mean Fib'
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        condition1 = (dataframe['move'] >= dataframe['move_mean']) & (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma_dn']) & (dataframe['sma_dn'].shift() > dataframe['sma_dn']) & (dataframe['volume'] > 0)
        dataframe.loc[condition1, 'exit_short'] = 1
        dataframe.loc[condition1, 'exit_tag'] = 'Up Trend Soon below sma_dn'
        condition2 = (dataframe['move'] >= dataframe['move_mean_x']) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma_dn']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['volume'] > 0)
        dataframe.loc[condition2, 'exit_short'] = 1
        dataframe.loc[condition2, 'exit_tag'] = 'Move Mean Fib below sma_dn'
        condition3 = (dataframe['move'] >= dataframe['move_mean']) & (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['OHLC4'] < dataframe['sma']) & (dataframe['volume'] > 0)
        dataframe.loc[condition3, 'exit_short'] = 1
        dataframe.loc[condition3, 'exit_tag'] = 'Up Trend Soon below sma'
        condition4 = (dataframe['move'] >= dataframe['move_mean_x']) & (dataframe['min'] < dataframe['max']) & (dataframe['min_l'] < dataframe['max_l']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['OHLC4'] < dataframe['sma']) & (dataframe['max_l'] < dataframe['atr_pcnt']) & (dataframe['volume'] > 0)
        dataframe.loc[condition4, 'exit_short'] = 1
        dataframe.loc[condition4, 'exit_tag'] = 'Move Mean Fib below sma'
        condition5150 = (dataframe['move'] >= dataframe['move_mean']) & (dataframe['min'] < dataframe['max']) & (dataframe['OHLC4'] > dataframe['sma']) & (dataframe['sma_up'].shift() < dataframe['sma']) & (dataframe['volume'] > 0)
        dataframe.loc[condition5150, 'exit_short'] = 1
        dataframe.loc[condition5150, 'exit_tag'] = 'Hope this works...'


        condition5 =  (dataframe['move'] >= dataframe['move_mean']) & (dataframe['move'].shift(6) < dataframe['move_mean'].shift(6)) & (dataframe['min'] > dataframe['max']) & (dataframe['min_l'] > dataframe['max_l']) & (dataframe['volume'] > 0)
        dataframe.loc[condition5, 'exit_long'] = 1
        dataframe.loc[condition5, 'exit_tag'] = 'Down Trend Soon'
        condition6 =  (dataframe['move'] >= dataframe['move_mean_x']) & (dataframe['move'].shift(3) >= dataframe['move_mean_x'].shift(3)) & (dataframe['min'] > dataframe['max']) & (dataframe['min_l'] > dataframe['max_l']) & (dataframe['volume'] > 0)
        dataframe.loc[condition6, 'exit_long'] = 1
        dataframe.loc[condition6, 'exit_tag'] = 'Move Mean Fib'
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        # Ensure max_leverage is not None and is a float
        if max_leverage is None:
            raise ValueError('Max leverage cannot be None')
        if not isinstance(max_leverage, (int, float)):
            raise ValueError('Max leverage must be a number')
        base_leverage = 1
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        bullish_patterns = ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 'CDLPIERCING', 'CDLMORNINGSTAR', 'CDL3WHITESOLDIERS']
        bearish_patterns = ['CDLHANGINGMAN', 'CDLSHOOTINGSTAR', 'CDLGRAVESTONEDOJI', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR']
        bullish_bearish_patterns = ['CDL3LINESTRIKE', 'CDLSPINNINGTOP', 'CDLENGULFING', 'CDLHARAMI', 'CDL3OUTSIDE', 'CDL3INSIDE']
        pattern_found = False  # Flag to check if any pattern is found
        if side == 'long':
            for pattern in bullish_patterns:
                pattern_value = current_candle.get(pattern)
                print(f'{pattern} Value:', pattern_value)
                if pattern_value is not None and pattern_value == 100:
                    # Bullish signal
                    base_leverage = max(1.0, min(base_leverage * 1.5, max_leverage))
                    print('Bullish Pattern:', pattern, 'Adjusted Leverage:', base_leverage)
                    pattern_found = True
                    break
        elif side == 'short':
            for pattern in bearish_patterns:
                pattern_value = current_candle.get(pattern)
                print(f'{pattern} Value:', pattern_value)
                if pattern_value is not None and pattern_value == 100:  # Bearish signal
                    base_leverage = max(1.0, min(base_leverage * 1.5, max_leverage))
                    print('Bearish Pattern:', pattern, 'Adjusted Leverage:', base_leverage)
                    pattern_found = True
                    break
        if not pattern_found and side == 'long':
            for pattern in bullish_bearish_patterns:
                pattern_value = current_candle.get(pattern)
                print(f'{pattern} Value:', pattern_value)
                if pattern_value is not None and pattern_value == 100:  # Bullish signal
                    base_leverage = max(1.0, min(base_leverage * 2, max_leverage))
                    print('Bullish Pattern:', pattern, 'Adjusted Leverage:', base_leverage)
                    pattern_found = True
                    break  # Exit the loop since a bullish pattern is found
        elif not pattern_found and side == 'short':
            for pattern in bullish_bearish_patterns:
                pattern_value = current_candle.get(pattern)
                print(f'{pattern} Value:', pattern_value)
                if pattern_value is not None and pattern_value == -100:  # Bearish signal
                    base_leverage = max(1.0, min(base_leverage * 2, max_leverage))
                    print('Bearish Pattern:', pattern, 'Adjusted Leverage:', base_leverage)
                    pattern_found = True
                    break  # Exit the loop since a bearish pattern is found
        # Apply maximum and minimum limits if any pattern is found
        if pattern_found:
            adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)  # Apply max and min limits
            print('Base Leverage:', proposed_leverage)
            print('Adjusted Leverage:', adjusted_leverage)
            return adjusted_leverage  # Return the adjusted leverage
        # If no pattern matches, return the proposed_leverage
        print('No Pattern Matched. Using Proposed Leverage:', proposed_leverage)
        return proposed_leverage

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if "short" in trade.enter_tag:
                num_shorts += 1
            elif "long" in trade.enter_tag:
                num_longs += 1

        if side == "long" and num_longs >= 5:
            return False

        if side == "short" and num_shorts >= 5:
            return False

        return True
