# Required imports for FreqTrade strategy
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd
import pandas_ta as ta
from pandas import DataFrame

class UltraCreativeStrategy_20250721_210729_iter_2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '3m'
    
    minimal_roi = {"0": 0.09}
    stoploss = -0.1
    
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    
    process_only_new_candles = True
    startup_candle_count: int = 30
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Dynamic indicators generated with complexity: ultra_complex
        dataframe["indicator_0"] = ta.ema(dataframe["close"], length=50)
        dataframe["indicator_1"] = ta.sma(dataframe["close"], length=50)
        # Stochastic returns multiple columns
        stoch_result = ta.stoch(dataframe["high"], dataframe["low"], dataframe["close"])
        dataframe["indicator_2_k"] = stoch_result["STOCHk_14_3_3"]
        dataframe["indicator_2_d"] = stoch_result["STOCHd_14_3_3"]
        dataframe["indicator_3"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=12)
        # Keltner Channel returns multiple columns
        kc_result = ta.kc(dataframe["high"], dataframe["low"], dataframe["close"])
        dataframe["indicator_4_lower"] = kc_result["KCLe_20_2"]
        dataframe["indicator_4_basis"] = kc_result["KCBe_20_2"]
        dataframe["indicator_4_upper"] = kc_result["KCUe_20_2"]
        dataframe["indicator_5"] = dataframe["volume"].ewm(span=40).mean()
        dataframe["indicator_6"] = ta.obv(dataframe["close"], dataframe["volume"])
        dataframe["indicator_7"] = ta.roc(dataframe["close"], length=24)
        dataframe["indicator_8"] = ta.roc(dataframe["close"], length=8)
        dataframe["indicator_9"] = ta.roc(dataframe["close"], length=12)
        dataframe["indicator_10"] = ta.roc(dataframe["close"], length=5)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (((dataframe["indicator_0"] > dataframe["indicator_1"]).fillna(False)) & ((dataframe["indicator_1"] > 0).fillna(False))) & (((dataframe["indicator_2_k"].diff() > 0).fillna(False)) & ((dataframe["indicator_2_d"].diff() > 0).fillna(False)))
            ),
            'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (((dataframe["indicator_3"].diff() > 0).fillna(False)) & ((dataframe["indicator_4_lower"] > dataframe["indicator_4_basis"]).fillna(False))) & (((dataframe["indicator_4_basis"] > 0).fillna(False)) & ((dataframe["indicator_4_upper"].diff() > 0).fillna(False)))
            ),
            'exit_long'] = 1
        return dataframe
