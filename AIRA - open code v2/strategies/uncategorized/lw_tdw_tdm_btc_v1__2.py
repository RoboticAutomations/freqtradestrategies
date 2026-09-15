# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.persistence import Order
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa
import pandas as pd
import datetime as dt
import numpy  as np

class lw_tdw_tdm_btc_v1(IStrategy):
    
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        # "0": 0.1
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # stoploss = -0.05
    #stoploss = -1 # unlimited
    stoploss = -1

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    # exit_profit_only = True
    ignore_roi_if_entry_signal = False
    
    # Optimal timeframe for the strategy
    timeframe = '15m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Convert 'date' column to datetime
        dataframe['datetime'] = pd.to_datetime(dataframe['date'], unit='ms')
        
        # Extract date components 
        dataframe['hour'] = dataframe['datetime'].dt.hour
        dataframe['minute'] = dataframe['datetime'].dt.minute
        dataframe['weekday'] = dataframe['datetime'].dt.weekday
        dataframe['monthday'] = dataframe['datetime'].dt.day

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # Entry signal at midnight (0 hour)
        dataframe.loc[
            (
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 4)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 6)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 8)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 14)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 18)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 25)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 26)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 29)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 16)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 22)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 29)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 31)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 1)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 3)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 4)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 5)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 6)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 9)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 10)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 12)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 13)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 15)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 16)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 20)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 21)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 25)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 26)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 27)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 29)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 4) & (dataframe['monthday'] == 6)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 4) & (dataframe['monthday'] == 13)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 4) & (dataframe['monthday'] == 19)) | 
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 4)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 10)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 19)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 20)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 28)) |
                ((dataframe['hour'] == 0) & (dataframe['weekday'] == 6) & (dataframe['monthday'] == 3))
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'lw_tdw_tdm')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # Exit signal at 23 hours 
        dataframe.loc[
            (
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 4)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 6)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 8)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 14)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 18)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 25)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 26)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 0) & (dataframe['monthday'] == 29)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 16)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 22)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 1) & (dataframe['monthday'] ==29)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 1) & (dataframe['monthday'] == 31)) |                
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 1)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 3)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 4)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 5)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 6)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 9)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 10)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 12)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 13)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 15)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 16)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 20)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 21)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 25)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 26)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 27)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 2) & (dataframe['monthday'] == 29)) |                
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 4) & (dataframe['monthday'] == 6)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 4) & (dataframe['monthday'] == 13)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 4) & (dataframe['monthday'] == 19)) |                
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 4)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 10)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 19)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 20)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 5) & (dataframe['monthday'] == 28)) |
                ((dataframe['hour'] == 23) & (dataframe['minute'] == 45) & (dataframe['weekday'] == 6) & (dataframe['monthday'] == 3))                
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'lw_tdw_tdm')

        return dataframe
