import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Tuple, Union
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)

class mk1(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = '5m'
    minimal_roi = {
        "0": 0.168,
        "16": 0.114,
        "31": 0.088,
        "41": 0.068,
        "61": 0.0422,
        
    }
    stoploss = -0.213

    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.02  # value loaded from strategy
    trailing_stop_positive_offset = 0.16  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    #use_exit_signal = True
    #exit_profit_only = False

    buy_params = {
        "rsi_entry_long": 41,
        "rsi_entry_short": 59,
        "window": 24,
    }

    sell_params = {
        "rsi_exit_long": 18,
        "rsi_exit_short": 78,
    }

    max_open_trades = 20

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 5


    rsi_entry_long  = IntParameter(0, 100, default=buy_params.get('rsi_entry_long'),  space='buy',  optimize=True)
    rsi_exit_long   = IntParameter(0, 100, default=buy_params.get('rsi_exit_long'),   space='sell', optimize=True)
    rsi_entry_short = IntParameter(0, 100, default=buy_params.get('rsi_entry_short'), space='buy',  optimize=True)
    rsi_exit_short  = IntParameter(0, 100, default=buy_params.get('rsi_exit_short'),  space='sell', optimize=True)
    window          = IntParameter(5, 100, default=buy_params.get('window'),          space='buy',  optimize=False)

    @property
    def protections(self):
        return [{
            "method": "CooldownPeriod",
            "stop_duration_candles": 12,
            "protection_per_coin": True
        }]
    
    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            'rsi_ema' : {}
        }
        plot_config['subplots'] = {
            'Misc': {
                'rsi': {},
                'rsi_gra' : {},
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ema'] = dataframe['rsi'].ewm(span=self.window.value).mean()
        dataframe['rsi_gra'] = np.gradient(dataframe['rsi_ema'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_entry_long.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_entry_short.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.rsi_exit_long.value) &
                qtpylib.crossed_below(dataframe['rsi_gra'], 0)
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_exit_short.value) &
                qtpylib.crossed_above(dataframe['rsi_gra'], 0)
            ),
        'exit_short'] = 1

        return dataframe