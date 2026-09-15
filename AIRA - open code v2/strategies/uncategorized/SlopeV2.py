import datetime
import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from scipy.stats import linregress
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy)
from freqtrade.persistence import Trade
from typing import Optional, Tuple, Union


class SlopeV2(IStrategy):
    INTERFACE_VERSION = 3
    
    can_short = True
    timeframe = '15m'
    use_exit_signal = True
    exit_profit_only = True

    # Buy hyperspace params:
    buy_params = {
        "threshold_di_buy": 34,
        "threshold_volume_buy": 1.559,
        "window": 24,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "threshold_di_sell": 52,
        "threshold_volume_sell": 63.506,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.207,
        "43": 0.126,
        "136": 0.054,
        "373": 0
    }

    # Stoploss:
    stoploss = -0.344

    # Trailing stop:
    trailing_stop = False

    # Max Open Trades:
    max_open_trades = -1

    # Hyperparameters
    window                =     IntParameter(  1,   120, space='buy',  default=buy_params['window'],                 optimize=False)
    threshold_di_buy      =     IntParameter(  1,   100, space='buy',  default=buy_params['threshold_di_buy'],       optimize=True)
    threshold_di_sell     =     IntParameter(  1,   100, space='sell', default=sell_params['threshold_di_sell'],     optimize=True)
    threshold_volume_buy  = DecimalParameter(0.0, 100.0, space='buy',  default=buy_params['threshold_volume_buy'],   optimize=True)
    threshold_volume_sell = DecimalParameter(0.0, 100.0, space='sell', default=sell_params['threshold_volume_sell'], optimize=True)

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
            },
            'subplots' : {
                'Directional Indicator' : {
                    'plus_di'    : { 'color' : 'red' },
                    'minus_di'   : { 'color' : 'blue' },
                },
                'Volume %' : {
                    'volume_pct' : { 'color' : 'black' },
                },
            }
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['plus_di']    = ta.PLUS_DI(dataframe)
        dataframe['minus_di']   = ta.MINUS_DI(dataframe)
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.window.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.threshold_volume_buy.value) &
                (dataframe['minus_di']   < self.threshold_di_buy.value)     &
                (dataframe['plus_di']    > self.threshold_di_buy.value)     &
                (dataframe['volume']     > 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.threshold_volume_buy.value) &
                (dataframe['minus_di']   > self.threshold_di_buy.value)     &
                (dataframe['plus_di']    < self.threshold_di_buy.value)     &
                (dataframe['volume']     > 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['volume_pct'] < self.threshold_volume_sell.value) &
                (dataframe['minus_di']   > self.threshold_di_sell.value)     &
                (dataframe['plus_di']    < self.threshold_di_sell.value)     &
                (dataframe['volume']     > 0)
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['volume_pct'] < self.threshold_volume_sell.value) &
                (dataframe['minus_di']   < self.threshold_di_sell.value)     &
                (dataframe['plus_di']    > self.threshold_di_sell.value)     &
                (dataframe['volume']     > 0)
            ),
        'exit_short'] = 1

        return dataframe