import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, merge_informative_pair


class GnF_V2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "Volume": {
                    "volume": {"color": "yellow"},
                    "volume_mean": {"color": "orange"},
                },
                "GreedFearIndex": {
                    "threshold_upper": {"color": "red", "type" : "scatter"},
                    "greed_fear_index": {"color": "blue"},
                    "threshold_lower": {"color": "red", "type" : "scatter"},
                },
            },
        }
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ===== GFI: 1. Price Momentum (normalized) =====
        ma = ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['momentum'] = (dataframe['close'] - ma) / ma

        # ===== GFI: 2. Volatility via ATR (normalized) =====
        dataframe['volatility'] = ta.NATR(dataframe, timeperiod=14)

        # ===== GFI: 3. Volume Sentiment (Up/Down volume ratio) =====        
        dataframe["volume_mean"] = dataframe["volume"].rolling(50).mean()
        dataframe['volume_up'] = dataframe['volume'] * (dataframe['close'] > dataframe['open'])
        dataframe['volume_down'] = dataframe['volume'] * (dataframe['close'] < dataframe['open'])
        volume_total = dataframe['volume_up'] + dataframe['volume_down']
        dataframe['volume_sentiment'] = (dataframe['volume_up'] - dataframe['volume_down']) / volume_total.replace(0, 1)

        # ===== GFI: Final Greed & Fear Index =====
        # Normalize all to 0–1
        dataframe['momentum_norm'] = (dataframe['momentum'] - dataframe['momentum'].rolling(20).min()) / (dataframe['momentum'].rolling(20).max() - dataframe['momentum'].rolling(20).min() + 1e-9)
        dataframe['volatility_norm'] = (dataframe['volatility'] - dataframe['volatility'].rolling(20).min()) / (dataframe['volatility'].rolling(20).max() - dataframe['volatility'].rolling(20).min() + 1e-9)
        dataframe['volume_sentiment_norm'] = (dataframe['volume_sentiment'] - dataframe['volume_sentiment'].rolling(20).min()) / (dataframe['volume_sentiment'].rolling(20).max() - dataframe['volume_sentiment'].rolling(20).min() + 1e-9)

        # GFI Score (0–1), higher = more greed
        dataframe['greed_fear_index'] = (
            dataframe['momentum_norm'] +
            dataframe['volatility_norm'] +
            dataframe['volume_sentiment_norm']
        ) / 3
        
        dataframe['greed_fear_mean'] = dataframe['greed_fear_index'].rolling(14).mean()
        
        dataframe["threshold_upper"] = 0.7
        dataframe["threshold_lower"] = 0.3

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['greed_fear_index'] > 0.7) &
                (dataframe['volume'] > dataframe['volume_mean'])
            ),
            "enter_long"
        ] = 1
        
        dataframe.loc[
            (
                (dataframe['greed_fear_index'] < 0.3) &
                (dataframe['volume'] > dataframe['volume_mean'])
            ),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['greed_fear_index'] < 0.3)
            ),
            "exit_long"
        ] = 1
        
        dataframe.loc[
            (
                (dataframe['greed_fear_index'] > 0.7)
            ),
            "exit_short"
        ] = 1
        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 1
