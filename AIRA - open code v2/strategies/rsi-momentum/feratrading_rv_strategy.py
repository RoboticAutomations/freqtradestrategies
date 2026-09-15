# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta

class FeraRelativeVolumeStrategy(IStrategy):
    """
    FeraTrading Relative Volume Strategy
    
    Converted from Pine Script indicator that uses volume pressure analysis
    to generate buy/sell signals based on relative volume movements.
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.10,    # 10% profit
        "30": 0.05,   # 5% after 30 minutes
        "60": 0.02,   # 2% after 1 hour
        "120": 0.01   # 1% after 2 hours
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.05  # 5% stoploss

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Strategy parameters - equivalent to Pine Script inputs
    pressure_weight = IntParameter(0, 100, default=0, space="buy", optimize=True)
    pressure_length = IntParameter(5, 30, default=14, space="buy", optimize=True)
    sma1_length = IntParameter(20, 100, default=50, space="buy", optimize=True)
    sma2_length = IntParameter(20, 100, default=50, space="sell", optimize=True)
    
    # Optional parameters for fine-tuning
    bottom_line_weight = IntParameter(100, 300, default=200, space="buy", optimize=False)
    bottom_line_length = IntParameter(10, 20, default=14, space="buy", optimize=False)
    top_line_weight = IntParameter(100, 300, default=200, space="buy", optimize=False)
    top_line_length = IntParameter(10, 20, default=14, space="buy", optimize=False)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        
        # Calculate buy and sell pressure (equivalent to Pine Script logic)
        high_low_diff = dataframe['high'] - dataframe['low']
        # Avoid division by zero
        high_low_diff = high_low_diff.replace(0, 0.0001)
        
        # Buy pressure: volume weighted by how close the close is to the high
        buy_pressure = dataframe['volume'] * (dataframe['close'] - dataframe['low']) / high_low_diff
        
        # Sell pressure: volume weighted by how close the close is to the low  
        sell_pressure = dataframe['volume'] * (dataframe['high'] - dataframe['close']) / high_low_diff
        
        # Net pressure with weight adjustment
        pressure_net = (buy_pressure - sell_pressure) * (1 - (self.pressure_weight.value / 100))
        
        # Smooth the pressure with SMA
        dataframe['pressure_smoothed'] = ta.SMA(pressure_net, timeperiod=self.pressure_length.value)
        
        # Calculate bottom and top lines (volume-based)
        volume_sma_bottom = ta.SMA(dataframe['volume'], timeperiod=self.bottom_line_length.value)
        dataframe['bottom_line'] = volume_sma_bottom / (1000 / self.bottom_line_weight.value) * -1
        
        volume_sma_top = ta.SMA(dataframe['volume'], timeperiod=self.top_line_length.value)
        dataframe['top_line'] = volume_sma_top / (1000 / self.top_line_weight.value)
        
        # ATR weighted calculation
        dataframe['atr_weighted'] = dataframe['pressure_smoothed'] + abs(dataframe['bottom_line'])
        
        # Calculate SMAs for the ATR weighted values (these generate the signals)
        dataframe['atr_sma1'] = ta.SMA(dataframe['atr_weighted'], timeperiod=self.sma1_length.value)
        dataframe['atr_sma2'] = ta.SMA(dataframe['atr_weighted'], timeperiod=self.sma2_length.value)
        
        # Additional indicators for context
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        Buy signal: when atr_weighted crosses above its SMA (similar to Pine Script logic)
        """
        dataframe.loc[
            (
                # Main signal: atr_weighted crosses above its SMA
                (dataframe['atr_weighted'] > dataframe['atr_sma1']) &
                (dataframe['atr_weighted'].shift(1) <= dataframe['atr_sma1'].shift(1)) &
                
                # Additional filters for better signal quality
                (dataframe['volume'] > dataframe['volume_sma']) &  # Above average volume
                (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70) &  # Not in extreme RSI territory
                
                # Ensure we have valid data
                (dataframe['atr_weighted'].notna()) &
                (dataframe['atr_sma1'].notna())
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        Sell signal: when atr_weighted crosses below its SMA2
        """
        dataframe.loc[
            (
                # Main signal: atr_weighted crosses below its SMA2
                (dataframe['atr_weighted'] < dataframe['atr_sma2']) &
                (dataframe['atr_weighted'].shift(1) >= dataframe['atr_sma2'].shift(1)) &
                
                # Ensure we have valid data
                (dataframe['atr_weighted'].notna()) &
                (dataframe['atr_sma2'].notna())
            ) |
            (
                # Alternative exit: RSI overbought
                (dataframe['rsi'] > 80)
            ),
            'exit_long'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returns the new stoploss relative to current_rate.
        """
        # Use the default stoploss for this strategy
        return self.stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing is critical here, so avoid doing heavy computations.
        """
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing is critical here, so avoid doing heavy computations.
        """
        return True