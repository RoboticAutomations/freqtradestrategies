import logging
import pandas as pd
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional
import talib.abstract as ta
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter, BooleanParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade


class ZaratustraV30(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '1m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = True
    
    # ROI table:
    minimal_roi = {
        "0": 0.032,
        "4": 0.023,
        "7": 0.005,
        "18": 0
    }

    # Stoploss:
    stoploss = -0.184

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.124
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = True
    
    # Max Open Trades:
    max_open_trades = 1

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
        }
        plot_config['subplots'] = {
            'Misc' : {
                'rsi_7' : { 'color' : 'yellow', },
                'rsi_14' : { 'color' : 'red' },
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_7'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['rsi_14'], 70)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'RSI Long')

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['rsi_14'], 30)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'RSI Short')
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['rsi_7'], dataframe['rsi_14'])
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'RSI Exit Long')

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['rsi_7'], dataframe['rsi_14'])
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'RSI Exit Short')
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 1