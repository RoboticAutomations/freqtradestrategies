import talib.abstract as ta
from functools import reduce
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import (CategoricalParameter, DecimalParameter, IntParameter, IStrategy)


class RSIBB_V4(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    set_leverage = 3
    use_exit_signal = True  # Changed to True to use exit signals
    startup_candle_count = 20
    
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
        
    # --- Parameter definitions ---
    rsi_long_min = IntParameter(40, 60, default=50, space="buy", optimize=True)
    rsi_short_max = IntParameter(40, 60, default=50, space="sell", optimize=True)
    bars_above_bb = IntParameter(1, 3, default=1, space="buy", optimize=True)
    bars_below_bb = IntParameter(1, 3, default=1, space="sell", optimize=True)
    squeeze_threshold = DecimalParameter(0.02, 0.10, decimals=3, default=0.05, space="buy", optimize=True)
    expansion_factor = DecimalParameter(1.05, 1.50, decimals=2, default=1.20, space="buy", optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = dataframe.copy()
        # RSI
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=6)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=12)
        
        # Price Bollinger Bands
        typical_price = qtpylib.typical_price(dataframe)
        bollinger = qtpylib.bollinger_bands(typical_price, window=20, stds=2)
        dataframe["bbl"] = bollinger["lower"]
        dataframe["bbm"] = bollinger["mid"]
        dataframe["bbu"] = bollinger["upper"]
        
        # Volume Bollinger Bands
        dataframe["vol_mean"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_std"] = dataframe["volume"].rolling(20).std()
        dataframe["vbbu"] = dataframe["vol_mean"] + (dataframe["vol_std"] * 2)
        dataframe["vbbm"] = dataframe["vol_mean"]
        dataframe["vbbl"] = dataframe["vol_mean"] - (dataframe["vol_std"] * 2)
        
        # Bollinger Band Width (BBW)
        dataframe["bbw"] = (dataframe["bbu"] - dataframe["bbl"]) / dataframe["bbm"]

        # Detect squeeze: BBW < threshold
        dataframe["squeeze"] = dataframe["bbw"] < self.squeeze_threshold.value

        # Expansion: current BBW > N% above last N BBW min
        rolling_min_bbw = dataframe["bbw"].rolling(5).min()
        dataframe["expansion"] = dataframe["bbw"] > (rolling_min_bbw * self.expansion_factor.value)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> DataFrame:
        # Initialize columns (safety)
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0
        
        # Long entry
        long_conditions = (
            (dataframe["close"] > dataframe["bbu"]).rolling(self.bars_above_bb.value).sum() == self.bars_above_bb.value,
            dataframe["expansion"],
            dataframe["squeeze"].shift(1),
            (dataframe["volume"] > dataframe["vbbu"]),
            (dataframe["rsi_fast"] > self.rsi_long_min.value),
            (dataframe["rsi_slow"] > self.rsi_long_min.value)
        )
        dataframe.loc[reduce(lambda x, y: x & y, long_conditions), "enter_long"] = 1

        # Short entry
        if self.can_short:
            short_conditions = (
                (dataframe["close"] < dataframe["bbl"]).rolling(self.bars_below_bb.value).sum() == self.bars_below_bb.value,
                dataframe["expansion"],
                dataframe["squeeze"].shift(1),
                (dataframe["volume"] > dataframe["vbbu"]),
                (dataframe["rsi_fast"] < self.rsi_short_max.value),
                (dataframe["rsi_slow"] < self.rsi_short_max.value)
            )
            dataframe.loc[reduce(lambda x, y: x & y, short_conditions), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> DataFrame:
        # Initialize columns (critical!)
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        
        # Long exit
        dataframe.loc[
            (dataframe["close"] < dataframe["bbm"]) | (dataframe["volume"] < dataframe["vbbm"]),
            "exit_long"
        ] = 1

        # Short exit
        if self.can_short:
            dataframe.loc[
                (dataframe["close"] > dataframe["bbm"]) | (dataframe["volume"] < dataframe["vbbm"]),
                "exit_short"
            ] = 1

        return dataframe  

    def leverage(self, pair: str, current_time: datetime, current_rate: float, 
                proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return self.set_leverage
