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

class lw_tdw_tdm_btc(IStrategy):
    
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

        dataframe['hour'] = 333
        dataframe['minute'] = 333
        dataframe['weekday'] = 333
        dataframe['monthday'] = 333
        
        dataframe['datetime'] = pd.to_datetime(dataframe.date, unit='ms')

        for i in range(len(dataframe)):
            dataframe['weekday'].values[i] = dt.datetime.weekday(dataframe['datetime'][i])
            dataframe['monthday'].values[i] = dataframe['datetime'][i].day

            if dataframe['datetime'][i].hour == 0:
                dataframe['hour'].values[i] = 0
            if dataframe['datetime'][i].hour == 23:
                dataframe['hour'].values[i] = 23
            if dataframe['datetime'][i].minute == 45:
                dataframe['minute'].values[i] = 45   
    
        """
        print(data['name'].values[0])
        print(data['id'].loc[data.index[0]]) 
        print(data['id'].iloc[0])
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

            dataframe.loc[
            (
                    (dataframe['hour'] ==  0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                    (dataframe['hour'] ==  23)
            ),
            'sell'] = 1
        return dataframe
