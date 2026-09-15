from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy.interface import IStrategy
from datetime import datetime
from pandas import DataFrame
from typing import Dict, List
import talib.abstract as ta
import pywt

class SupportResistance(IStrategy):
    INTERFACE_VERSION = 3

    # Parameter strategi
    timeframe = '5m'
    window = 20
    std_dev = 2
    
    stoploss = -0.3
    minimal_roi = {
        "0": 0.612
    }
    use_exit_signal = True
    can_short = True
    
    trailing_stop = False
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.080
    trailing_only_offset_is_reached = True
    
    
    def detect_double_top_bottom(self, df: DataFrame, threshold=0.05) -> DataFrame:
        # Define the rolling window
        roll_window = self.window
        # Define a threshold to check for the range of pattern
        range_threshold = threshold
    
        # Create a rolling window for high and low
        df['high_roll_max'] = df['high'].rolling(window=roll_window).max()
        df['low_roll_min'] = df['low'].rolling(window=roll_window).min()
    
        # Create a boolean mask for Double Top pattern
        df["mask_double_top"] = (df['high_roll_max'] >= df['high'].shift(1)) & (df['high_roll_max'] >= df['high'].shift(-1)) & (df['high'] < df['high'].shift(1)) & (df['high'] < df['high'].shift(-1)) & ((df['high'].shift(1) - df['low'].shift(1)) <= range_threshold * (df['high'].shift(1) + df['low'].shift(1))/2) & ((df['high'].shift(-1) - df['low'].shift(-1)) <= range_threshold * (df['high'].shift(-1) + df['low'].shift(-1))/2)
        # Create a boolean mask for Double Bottom pattern
        df["mask_double_bottom"] = (df['low_roll_min'] <= df['low'].shift(1)) & (df['low_roll_min'] <= df['low'].shift(-1)) & (df['low'] > df['low'].shift(1)) & (df['low'] > df['low'].shift(-1)) & ((df['high'].shift(1) - df['low'].shift(1)) <= range_threshold * (df['high'].shift(1) + df['low'].shift(1))/2) & ((df['high'].shift(-1) - df['low'].shift(-1)) <= range_threshold * (df['high'].shift(-1) + df['low'].shift(-1))/2)
    
        # Create a new column for Double Top and Double Bottom pattern and populate it using the boolean masks
        df["double_top"] = df["mask_double_top"]
        df["double_bottom"] = df["mask_double_bottom"]
    
        return df
    
    
    def calculate_support_resistance(self, dataframe: DataFrame) -> DataFrame:
        """Hitung support dan resistance"""
        dataframe['high_roll_max'] = dataframe['high'].rolling(window=self.window).max()
        dataframe['low_roll_min'] = dataframe['low'].rolling(window=self.window).min()
        
        dataframe['mean_high'] = dataframe['high'].rolling(window=self.window).mean()
        dataframe['std_high'] = dataframe['high'].rolling(window=self.window).std()
        dataframe['mean_low'] = dataframe['low'].rolling(window=self.window).mean()
        dataframe['std_low'] = dataframe['low'].rolling(window=self.window).std()
        
        dataframe['support'] = dataframe['mean_low'] - self.std_dev * dataframe['std_low']
        dataframe['resistance'] = dataframe['mean_high'] + self.std_dev * dataframe['std_high']
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indikator"""
        dataframe = self.calculate_support_resistance(dataframe)
        dataframe = self.detect_double_top_bottom(dataframe)
        dataframe["rsi"] = ta.RSI(dataframe)
        return dataframe

    # Sinyal masuk
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Sinyal masuk"""
        dataframe.loc[
            (
                (dataframe['close_5m'] < dataframe['support_5m']) &
                (dataframe['rsi_5m'] < 35) &
                (dataframe['close_1h'] < dataframe['support_1h']) &
                (dataframe['rsi_1h'] < 30)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long support')
        
        dataframe.loc[
            (
                (dataframe["close_5m"] > dataframe['double_top_5m']) &
                (dataframe['rsi_5m'] > 70) &
                (dataframe["close_1h"] > dataframe['double_top_1h']) &
                (dataframe['rsi_1h'] > 70)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short double top')
        
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10