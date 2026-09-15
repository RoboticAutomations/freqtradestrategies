import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class DC_UT(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.12,
        "10": 0.1,
        "25": 0.08,
        "60" : 0.05
    }
    stoploss = -0.2
    trailing_stop = False
    timeframe = '1h'
    process_only_new_candles = False
    use_sell_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 600

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'emergency_exit':"market"
    }

    exit_pricing = {
        'price_side': 'other'  # Новое требование для рыночных ордеров выхода
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def informative_pairs(self):
        return []

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    buy_params = {
        "rsi": 28
    }

    sell_params = {
        "rsi": 68
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_12h'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['dollar_volume_12h'] = (dataframe['volume'] * dataframe['close']).shift(1)
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi_condition = (dataframe['rsi_12h'] > 49)

        if 'ma50_12h' in dataframe.columns:
            ma50_condition = dataframe['close'] > dataframe['ma50_12h']
        else:
            ma50_condition = True

        dollar_volume_condition = dataframe['dollar_volume_12h'] > 2000000

        combined_condition = rsi_condition & ma50_condition & dollar_volume_condition

        dataframe.loc[combined_condition, 'buy'] = 1

        # Добавим exit_pricing для buy
        dataframe.loc[combined_condition, 'exit_price'] = dataframe['close']

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsi_sell_condition = (dataframe['rsi_12h'] > self.sell_params["rsi"])
        dollar_volume_sell_condition = dataframe['dollar_volume_12h'] > 2000000

        combined_sell_condition = rsi_sell_condition & dollar_volume_sell_condition

        dataframe.loc[combined_sell_condition, 'sell'] = 1

        # Добавим exit_pricing для sell
        dataframe.loc[combined_sell_condition, 'exit_price'] = dataframe['close']

        return dataframe

    def hyperopt_space() -> dict:
        return {
            "space": {
                "buy_params": {
                    "rsi": {"default": 28, "space": "uniform(20, 40)"}
                },
                "sell_params": {
                    "rsi": {"default": 68, "space": "uniform(60, 80)"}
                },
                "stoploss": {"default": -0.15, "space": "uniform(-0.2, -0.1)"},
                "roi": {"0": 0.15, "25": 0.1, "space": {"0": "uniform(0.1, 0.5)", "25": "uniform(0.05, 0.2)"}}
            },
            "buy": {
                "rsi": {"default": 28, "space": "uniform(20, 40)"}
            },
            "sell": {
                "rsi": {"default": 68, "space": "uniform(60, 80)"}
            }
        }
