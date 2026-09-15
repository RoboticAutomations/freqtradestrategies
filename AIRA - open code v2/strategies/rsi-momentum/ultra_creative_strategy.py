# Required imports for FreqTrade strategy
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame

class UltraDynamicStrategy_1753139562(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '3m'
    
    minimal_roi = {"0": 0.08}
    stoploss = -0.22
    
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    process_only_new_candles = True
    startup_candle_count: int = 30
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Dynamic indicators generated with complexity: simple
        # MACD returns multiple columns
        macd_result = ta.macd(dataframe["close"])
        dataframe["indicator_0_macd"] = macd_result["MACD_12_26_9"]
        dataframe["indicator_0_signal"] = macd_result["MACDs_12_26_9"]
        dataframe["indicator_0_hist"] = macd_result["MACDh_12_26_9"]
        dataframe["indicator_1"] = dataframe["volume"].ewm(span=60).mean()
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["indicator_0_macd"].diff() > 0) & (dataframe["close"].diff() < 0).fillna(False)) & ((dataframe["indicator_0_signal"] > 0).fillna(False))
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["indicator_0_hist"] > dataframe["indicator_1"]).fillna(False)) & ((dataframe["indicator_1"] > 0).fillna(False))
            ),
            'exit_long'] = 1
        return dataframe
