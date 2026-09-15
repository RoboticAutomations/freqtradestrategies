# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ImpulseMACD(IStrategy):
    '''    
    author@: Bryant Suen
    github@: https://github.com/BryantSuen

    Originally designed by @lazybear: https://www.tradingview.com/script/qt6xLfLi-Impulse-MACD-LazyBear/

    '''

    INTERFACE_VERSION = 3

    timeframe = '1h'

    can_short: bool = True

    order_types = {
    'entry': 'limit',
    'exit': 'limit',
    'stoploss': 'market',
    'stoploss_on_exchange': True
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }


    macd_ma_period = IntParameter(20, 50, default=20, space='buy', optimize=True, load=True)
    macd_signal_period = IntParameter(5, 15, default=14, space='buy', optimize=True, load=True)

    check_macd_position = BooleanParameter(default=False, space='buy', optimize=True, load=True)

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.8
    }

    stoploss = -0.5

    # trailing_stop = False
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.09
    trailing_only_offset_is_reached = True
    
    process_only_new_candles = False
    use_exit_signal = True

    startup_candle_count: int = 100

    def _cal_smma(self, series:pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    def _cal_zero_lag_ema(self, series:pd.Series, period: int) -> pd.Series:
        ema_1 = ta.EMA(series, timeperiod=period)
        ema_2 = ta.EMA(ema_1, timeperiod=period)
        return 2 * ema_1 - ema_2

    def impulsive_macd(self, dataframe: DataFrame, length_ma: int, length_signal: int) -> tuple:
        mean_hlc = dataframe[['high', 'low', 'close']].mean(axis=1)
        high_smma = self._cal_smma(dataframe['high'], length_ma)
        low_smma = self._cal_smma(dataframe['low'], length_ma)
        middle_zlema = self._cal_zero_lag_ema(mean_hlc, length_ma)

        impulse_macd = np.where(middle_zlema > high_smma, middle_zlema - high_smma, 0)
        impulse_macd = np.where(middle_zlema < low_smma, middle_zlema - low_smma, impulse_macd)

        impulse_macd_signal = ta.SMA(impulse_macd, timeperiod=length_signal)
        impulse_macd_hist = impulse_macd - impulse_macd_signal

        return impulse_macd, impulse_macd_signal, impulse_macd_hist

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for ma_period in self.macd_ma_period.range:
            for signal_period in self.macd_signal_period.range:
                macd, macdsignal, macdhist = self.impulsive_macd(dataframe, ma_period, signal_period)
                dataframe[f'impulse_macd_{ma_period}_{signal_period}'] = macd
                dataframe[f'impulse_macdsignal_{ma_period}_{signal_period}'] = macdsignal
                dataframe[f'impulse_macdhist_{ma_period}_{signal_period}'] = macdhist
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = []
        triggers_short = []

        guards_long = []
        guards_short = []

        triggers_long.append(qtpylib.crossed_above(dataframe[f'impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}'], dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}']))
        triggers_short.append(qtpylib.crossed_below(dataframe[f'impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}'], dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}']))

        if self.check_macd_position.value:
            guards_long.append(dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}'] < 0)
            guards_short.append(dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}'] > 0)

        # make sure volume is not 0
        guards_long.append(dataframe['volume'] > 0)
        guards_short.append(dataframe['volume'] > 0)

        if triggers_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_long) & reduce(lambda x, y: x & y, guards_long),
                'enter_long'] = 1
        
        if triggers_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_short) & reduce(lambda x, y: x & y, guards_short),
                'enter_short'] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = []
        triggers_short = []

        guards_long = []
        guards_short = []

        triggers_short.append(qtpylib.crossed_above(dataframe[f'impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}'], dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}']))
        triggers_long.append(qtpylib.crossed_below(dataframe[f'impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}'], dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}']))

        if self.check_macd_position.value:
            guards_short.append(dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}'] < 0)
            guards_long.append(dataframe[f'impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}'] > 0)
        
        # make sure volume is not 0
        guards_long.append(dataframe['volume'] > 0)
        guards_short.append(dataframe['volume'] > 0)

        if triggers_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_long) & reduce(lambda x, y: x & y, guards_long),
                'exit_long'] = 1
        
        if triggers_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_short) & reduce(lambda x, y: x & y, guards_short),
                'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return 5
    
    @property
    def plot_config(self):
        return {
            'main_plot': {},
            'subplots': {
                "IMPULSE_MACD": {
                    f"impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}": {'color': 'blue'},
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}": {'color': 'orange'},
                    f"impulse_macdhist_{self.macd_ma_period.value}_{self.macd_signal_period.value}": {'type': 'bar', 'plotly': {'opacity': 0.9}}
                }
            }
        }
    
