
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')
import numpy
import warnings
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from pandas import DataFrame, errors
from datetime import datetime
import numpy
from scipy.signal import argrelextrema
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)

warnings.simplefilter(action="ignore", category=errors.PerformanceWarning)



class el_extrema_rsiqui(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True

    entry_params = {
        'base_nb_candles_entry': 12,
        'ewo_high': 4.428,
        'ewo_low': -12.383,
        'low_offset': 0.915,
        'rsi_entry': 44,
    }

    exit_params = {
        'base_nb_candles_exit': 72,
        'high_offset': 1.008,
    }

    minimal_roi = {
        '0': 0.5,
        '60': 0.45,
        '120': 0.4,
        '240': 0.3,
        '360': 0.25,
        '720': 0.2,
        '1440': 0.15,
        '2880': 0.1,
        '3600': 0.05,
        '7200': 0.02,
    }

    stoploss = -0.05
    max_open_trades = 9
    timeframe = '5m'
    informative_timeframe = '1h'
    process_only_new_candles = True
    use_custom_stoploss = False
    
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True
    
    base_nb_candles_entry = IntParameter(5, 80, default=entry_params['base_nb_candles_entry'], space='buy', optimize=True)
    base_nb_candles_exit = IntParameter(5, 80, default=exit_params['base_nb_candles_exit'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=entry_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=exit_params['high_offset'], space='sell', optimize=True)
    
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=entry_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=entry_params['ewo_high'], space='buy', optimize=True)
    rsi_entry = IntParameter(30, 70, default=entry_params['rsi_entry'], space='buy', optimize=True)

    cooldown_lookback = IntParameter(2, 48, default=1, space='protection', optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space='protection', optimize=True)
    use_stop_protection = BooleanParameter(default=True, space='protection', optimize=True)

    @property
    def protections(self):
        prot = []
        prot.append(
            {
                'method': 'CooldownPeriod', 
                'stop_duration_candles': self.cooldown_lookback.value
            }
        )
        if self.use_stop_protection.value:
            prot.append(
                {
                    'method': 'StoplossGuard',
                    'lookback_period_candles': 24 * 3,
                    'trade_limit': 2,
                    'stop_duration_candles': self.stop_duration.value,
                    'only_per_pair': False,
                }
            )
        return prot
    
    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            'rsi': {},
            'sma' : {},
            'ewo' : {},
        }
        plot_config['subplots'] = {
            'RSI': {
                'rsi_gra' : {},
            },
            'SMA' : {
                'sma_gra' : {},
            },
            'ewo' : {
                'ewo_gra' : {},
            }
        }

        return plot_config
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.base_nb_candles_entry.range:
            dataframe[f'ma_entry_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_exit.range:
            dataframe[f'ma_exit_{val}'] = ta.EMA(dataframe, timeperiod=val)
    
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_gra'] = numpy.gradient(dataframe['rsi'], 60)
        
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['sma_gra'] = numpy.gradient(dataframe['sma'], 60)
        
        dataframe['ewo'] = (ta.SMA(dataframe, timeperiod=5)  - ta.SMA(dataframe, timeperiod=35)) / dataframe['close'] * 100
        dataframe['ewo_gra'] = numpy.gradient(dataframe['ewo'], 60)

        dataframe['&s-extrema'] = 0
        min_extrema_idx = argrelextrema(dataframe['close'].values, numpy.less, order=10)[0]
        max_extrema_idx = argrelextrema(dataframe['close'].values, numpy.greater, order=10)[0]
        dataframe.loc[min_extrema_idx, '&s-extrema'] = -1
        dataframe.loc[max_extrema_idx, '&s-extrema'] = 1
        dataframe['&s-extrema'] = dataframe['&s-extrema'].shift(1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)

            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['&s-extrema'] > 0) &
                (dataframe['close'] > dataframe[f'ma_exit_{self.base_nb_candles_exit.value}'] * self.high_offset.value) & 
                (dataframe['volume'] > 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 60) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'exit_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['&s-extrema'] < 0) &
                (dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & 
                (dataframe['ewo'] > self.ewo_high.value) & 
                (dataframe['rsi'] < self.rsi_entry.value) & 
                (dataframe['volume'] > 0)
            ),
        'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0