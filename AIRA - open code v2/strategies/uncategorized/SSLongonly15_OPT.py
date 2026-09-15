from freqtrade.strategy import IStrategy
from freqtrade.strategy import DecimalParameter, IntParameter
from pandas import DataFrame
import numpy as np
import pandas as pd


class SSLongonly15_OPT(IStrategy):
    """
    Scalping strategy for FreqTrade
    """
    INTERFACE_VERSION: int = 3

    # Core strategy parameters
    timeframe = "15m"
    startup_candle_count = 30

    # Strategy parameters
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = False

    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize any variables here
        self.buy_params = {}
        self.sell_params = {}

    # ROI table:
    minimal_roi = {
        "0": 0.005  # 0.5% profit
    }

    # Stoploss:
    stoploss = -0.5

    # Trading parameters
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate all indicators used by the strategy"""
        
        # Calculate candle characteristics
        dataframe['candle_height'] = dataframe['high'] - dataframe['low']
        dataframe['body_height'] = abs(dataframe['open'] - dataframe['close'])
        dataframe['is_green'] = dataframe['close'] > dataframe['open']
        
        # Calculate upper and lower shadows
        dataframe['upper_shadow'] = np.where(
            dataframe['is_green'],
            dataframe['high'] - dataframe['close'],
            dataframe['high'] - dataframe['open']
        )
        dataframe['lower_shadow'] = np.where(
            dataframe['is_green'],
            dataframe['open'] - dataframe['low'],
            dataframe['close'] - dataframe['low']
        )

        # Calculate half price levels using vectorized operations
        dataframe['half_price'] = np.where(
            dataframe['is_green'],
            np.where(
                dataframe['upper_shadow'] >= (dataframe['body_height'] + dataframe['lower_shadow']),
                dataframe['close'] - ((dataframe['close'] - dataframe['low']) * 0.8),
                np.where(
                    dataframe['candle_height'] / dataframe['low'] > 0.03,
                    dataframe['close'] - ((dataframe['close'] - dataframe['low']) * 0.2),
                    dataframe['close'] - ((dataframe['close'] - dataframe['low']) * 0.45)
                )
            ),
            np.where(
                dataframe['lower_shadow'] <= (dataframe['body_height'] + dataframe['upper_shadow']),
                dataframe['open'] - ((dataframe['open'] - dataframe['low']) * 0.8),
                dataframe['open'] - ((dataframe['open'] - dataframe['low']) * 0.45)
            )
        )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signal logic"""
        dataframe.loc[
            (
                # Entry conditions for green candles
                (dataframe['is_green']) &
                (dataframe['close'] >= dataframe['high'].shift(1))
            ) |
            (
                # Entry conditions for red candles
                (~dataframe['is_green']) &
                (dataframe['close'] >= dataframe['open'].shift(1) - 0.2 * dataframe['body_height'].shift(1))
            ) |
            (
                # Entry on half price level
                (dataframe['close'] <= dataframe['half_price'].shift(1))
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit signal logic"""
        dataframe.loc[
            (
                # Exit conditions based on time and profit
                (dataframe.index % 45 >= 30) &  # After 30 minutes in hour
                (
                    # Exit if price is below 40% of candle range in middle period
                    ((dataframe.index % 45 < 45) & 
                     ((dataframe['close'] - dataframe['low']) / 
                      (dataframe['high'] - dataframe['low']) < 0.4)) |
                    # Exit if price is below 90% of candle range in end period
                    ((dataframe.index % 45 >= 45) &
                     ((dataframe['close'] - dataframe['low']) / 
                      (dataframe['high'] - dataframe['low']) < 0.9))
                )
            ),
            'exit_long'
        ] = 1

        return dataframe
