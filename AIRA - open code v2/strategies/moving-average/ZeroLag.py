from freqtrade.strategy.interface import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame
import pandas as pd
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import merge_informative_pair, informative
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

'''
      ______   __          __              __    __   ______   __    __        __     __    __             ______            
 /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
/$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
$$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
$$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
$$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
$$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
$$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
 $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                   $$$$$$/                                               
                                                                                                                         
'''     

class ZeroLag(IStrategy):
    INTERFACE_VERSION = 3
    
    # Strategy configurations
    stoploss = -0.10
    timeframe = '1h'
    informative_timeframes = ['4h', '1d']
    process_only_new_candles = True
    startup_candle_count = 200

    # Hyperparameters
    length = IntParameter(20, 100, default=21, space='buy', optimize=True)
    mult = DecimalParameter(1.0, 2.0, default=1.2, decimals=1, space='buy', optimize=True)
    trend_confirmation = IntParameter(1, 3, default=3, space='buy', optimize=True)

    def calculate_zero_lag(self, dataframe: DataFrame, length: int, mult: float, suffix: str = '') -> DataFrame:
        """Calculate Zero Lag EMA and volatility bands"""
        lag = int((length - 1) // 2)
        
        modified_src = dataframe['close'] + (dataframe['close'] - dataframe['close'].shift(lag))
        dataframe[f'zlema{suffix}'] = ta.EMA(modified_src, timeperiod=length)
        
        dataframe[f'atr{suffix}'] = ta.ATR(dataframe, timeperiod=length)
        dataframe[f'volatility{suffix}'] = dataframe[f'atr{suffix}'].rolling(length*3).max() * mult
        
        dataframe[f'upper_band{suffix}'] = dataframe[f'zlema{suffix}'] + dataframe[f'volatility{suffix}']
        dataframe[f'lower_band{suffix}'] = dataframe[f'zlema{suffix}'] - dataframe[f'volatility{suffix}']
        
        return dataframe

    def calculate_trend(self, dataframe: DataFrame, suffix: str = '') -> DataFrame:
        """Calculate trend based on price crossing volatility bands"""
        dataframe[f'trend{suffix}'] = 0
        dataframe.loc[dataframe['close'] > dataframe[f'upper_band{suffix}'].shift(1), f'trend{suffix}'] = 1
        dataframe.loc[dataframe['close'] < dataframe[f'lower_band{suffix}'].shift(1), f'trend{suffix}'] = -1
        
        dataframe[f'trend{suffix}'] = dataframe[f'trend{suffix}'].replace(0, method='ffill')
        return dataframe

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculate_zero_lag(dataframe, self.length.value, self.mult.value, '_4h')
        dataframe = self.calculate_trend(dataframe, '_4h')
        return dataframe

    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.calculate_zero_lag(dataframe, self.length.value, self.mult.value, '_1d')
        dataframe = self.calculate_trend(dataframe, '_1d')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Main timeframe indicators
        dataframe = self.calculate_zero_lag(dataframe, self.length.value, self.mult.value)
        dataframe = self.calculate_trend(dataframe)
        
        # Merge informative timeframes
        for tf in self.informative_timeframes:
            inf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)
            inf_df = self.calculate_zero_lag(inf_df, self.length.value, self.mult.value, f'_{tf}')
            inf_df = self.calculate_trend(inf_df, f'_{tf}')
            dataframe = merge_informative_pair(dataframe, inf_df, self.timeframe, tf, ffill=True)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['trend'] == 1) &
                (dataframe['trend'].shift() != 1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '1h')
        
        dataframe.loc[
            (
                (dataframe['trend_4h_4h_x'] == 1) &
                (dataframe['trend_4h_4h_x'].shift() != 1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '4h')

        dataframe.loc[
            (
                (dataframe['trend_1d_1d_x'] == 1) &
                (dataframe['trend_1d_1d_x'].shift() != 1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['enter_long', 'enter_tag']] = (1, '1d')
        
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['trend'] == -1) &
                (dataframe['trend'].shift() != -1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, '1h')
        
        dataframe.loc[
            (
                (dataframe['trend_4h_4h_x'] == -1) &
                (dataframe['trend_4h_4h_x'].shift() != -1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, '4h')

        dataframe.loc[
            (
                (dataframe['trend_1d_1d_x'] == -1) &
                (dataframe['trend_1d_1d_x'].shift() != -1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
            ),
            ['exit_long', 'exit_tag']] = (1, '1d')
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        # Dynamic stoploss based on volatility
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        return -(current_candle['volatility'] * 0.5) / current_rate