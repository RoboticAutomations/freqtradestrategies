import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, merge_informative_pair


class GnF(IStrategy):
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
                "Momentum" : {
                    "momentum_norm" : {}
                },
                "Volatility" : {
                    "volatility_norm" : {}
                },
                "Sentiment" : {
                    "volume_sentiment_norm" : {}
                },
                "GreedFearIndex": {
                    "greed_fear_index": {"color": "blue"},
                },
            },
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ===== GFI: 1. Price Momentum (normalized) =====
        ma = ta.SMA(dataframe['close'], timeperiod=20)
        dataframe['momentum'] = (dataframe['close'] - ma) / ma

        # ===== GFI: 2. Volatility via ATR (normalized) =====
        atr = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = atr / dataframe['close']

        # ===== GFI: 3. Volume Sentiment (Up/Down volume ratio) =====
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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['greed_fear_index'] > 0.7)
            ),
            "enter_long"
        ] = 1
        
        dataframe.loc[
            (
                (dataframe['greed_fear_index'] < 0.3)
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
