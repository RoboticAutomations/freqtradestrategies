import talib.abstract as ta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy, informative


class RSIBB_V3(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    use_exit_signal = False
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.2

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "bbu": {"color": "blue"},
                "bbm": {"color": "orange"},
                "bbl": {"color": "blue"},
            },
            "subplots": {
                "RSI" : {
                    "rsi_fast" : {"color": "yellow"},
                    "rsi_slow" : {"color": "orange"},
                },
                "Volume" : {
                    "volume" : {"color": "red"},
                    "vbbu" : {"color": "blue"},
                    "vbbm" : {"color": "orange"},
                    "vbbl" : {"color": "blue"},
                }
            },
        }
        
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=6)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=12)
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bbl"] = bollinger["lower"]
        dataframe["bbm"] = bollinger["mid"]
        dataframe["bbu"] = bollinger["upper"]
        
        dataframe["vol_mean"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_std"] = dataframe["volume"].rolling(20).std()
        
        dataframe["vbbu"] = dataframe["vol_mean"] + (dataframe["vol_std"] * 2)
        dataframe["vbbm"] = dataframe["vol_mean"]
        dataframe["vbbl"] = dataframe["vol_mean"] - (dataframe["vol_std"] * 2)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["close"], dataframe["bbu"]) &
                qtpylib.crossed_above(dataframe["volume"], dataframe["vbbu"])
            ),
            "enter_long"
        ] = 1
        
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["close"], dataframe["bbl"]) &
                qtpylib.crossed_below(dataframe["volume"], dataframe["vbbl"])
            ),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["volume"], dataframe["vbbm"])
            ),
            "exit_short"
        ] = 1
        
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["volume"], dataframe["vbbm"])
            ),
            "exit_short"
        ] = 1
        
        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 1