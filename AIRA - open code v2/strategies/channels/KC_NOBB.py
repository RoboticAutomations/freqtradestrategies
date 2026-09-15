from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List, Union
import numpy as np  # noqa
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (DecimalParameter,
                                IntParameter, IStrategy, stoploss_from_absolute, informative)
from freqtrade.optimize.space import SKDecimal, Dimension 
import freqtrade.vendor.qtpylib.indicators as qtpylib


class KC_NOBB(IStrategy):

    minimal_roi = {
        "0": 0.6,
        "4320": 0
    }

    order_types = {
            'entry': 'limit',
            'exit': 'limit',
            'stoploss': 'limit',
            'emergency_exit': 'market',
            'force_exit': 'market',
            'force_entry': 'market',
            'stoploss_on_exchange': True,
            'stoploss_on_exchange_interval': 60,
            'stoploss_on_exchange_limit_ratio': 0.99
    }

    stoploss = -0.6
    can_short = True

    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.013

    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 399

    abs_long = DecimalParameter(1.0, 8.0, decimals=1, default=3.0, space="buy")
    abs_short = DecimalParameter(1.0, 8.0, decimals=1, default=3.0, space="sell")
    abs_long_12h = DecimalParameter(1.0, 8.0, decimals=1, default=3.0, space="buy")
    abs_short_12h = DecimalParameter(1.0, 8.0, decimals=1, default=3.0, space="sell")
    linear_2h = IntParameter(5, 25, default=11, space="buy")
    linear_12h = IntParameter(5, 25, default=11, space="sell")
    dm = IntParameter(5, 35, default=14, space="sell")

    @informative('30m')
    def populate_indicators_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'])
        return dataframe 

    
    @informative('8h')
    def populate_indicators_8h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)
        for val in self.linear_12h.range:
            dataframe[f'linear_open_{val}'] = ta.LINEARREG(dataframe['open'], timeperiod=val)
            dataframe[f'linear_close_{val}'] = ta.LINEARREG(dataframe['close'], timeperiod=val)
        return dataframe

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = ta.BBANDS(dataframe,timeperiod=20)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe["bb_width"] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband'] * 100)

        return dataframe
    
    @informative('2h')
    def populate_indicators_2h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.linear_2h.range:
            dataframe[f'linear_open_{val}'] = ta.LINEARREG(dataframe['open'], timeperiod=val)
            dataframe[f'linear_close_{val}'] = ta.LINEARREG(dataframe['close'], timeperiod=val)
        for val in self.dm.range:
            dataframe[f'plus_dm_{val}'] = ta.PLUS_DI(dataframe, val)
            dataframe[f'minus_dm_{val}'] = ta.MINUS_DI(dataframe, val)
        for val in self.abs_long.range:
            dataframe[f'abs_long_{val}'] = val
        for val in self.abs_short.range:
            dataframe[f'abs_short_{val}'] = val

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        keltner = qtpylib.keltner_channel(dataframe,window=20, atrs=2.6)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe[f'linear_close_{self.linear_2h.value}_2h'] > dataframe[f'linear_open_{self.linear_2h.value}_2h']) &
                    (dataframe[f'linear_close_{self.linear_12h.value}_8h'] > dataframe[f'linear_open_{self.linear_12h.value}_8h']) &
                    ((dataframe[f'plus_dm_{self.dm.value}_2h'] - dataframe[f'minus_dm_{self.dm.value}_2h']).abs() > dataframe[f'abs_long_{self.abs_long.value}_2h']) &
                    (dataframe['rsi_8h'] < 72) &
                    (dataframe['rsi_30m'] > dataframe['rsi_ma_30m']) &
                    (dataframe['bb_width_4h'] > 7.0) &

                    (dataframe['low'] < dataframe['kc_lowerband']) 
                )
           
            ),
            'enter_long'] = 1
        

        dataframe.loc[
            (
                (
                    (dataframe[f'linear_close_{self.linear_2h.value}_2h'] < dataframe[f'linear_open_{self.linear_2h.value}_2h']) &
                    (dataframe[f'linear_close_{self.linear_12h.value}_8h'] < dataframe[f'linear_open_{self.linear_12h.value}_8h']) &
                    ((dataframe[f'plus_dm_{self.dm.value}_2h'] - dataframe[f'minus_dm_{self.dm.value}_2h']).abs() > dataframe[f'abs_short_{self.abs_short.value}_2h']) &
                    (dataframe['rsi_8h'] > 28) &
                    (dataframe['rsi_30m'] < dataframe['rsi_ma_30m']) &
                    (dataframe['bb_width_4h'] > 7.0) &

                    (dataframe['high'] > dataframe['kc_upperband']) 
                )

            ),
            'enter_short'] = 1

        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe['enter_long'] == 1)),
            'exit_short'] = 1
        dataframe.loc[
            ((dataframe['enter_short'] == 1)),
            'exit_long'] = 1
        
        return dataframe


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        return 15.0
