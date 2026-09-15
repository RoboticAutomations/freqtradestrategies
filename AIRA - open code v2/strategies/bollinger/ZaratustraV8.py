# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from freqtrade.persistence import Trade
from typing import Dict, Optional, Union, Tuple
from freqtrade.strategy import IStrategy, merge_informative_pair 
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class ZaratustraV8(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    use_exit_signal = False
    exit_profit_only = True
    position_adjustment_enable = True
    inf_times = ["5m", "15m",]

    # ROI table:
    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    # Stoploss:
    stoploss = -0.296

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.071
    trailing_only_offset_is_reached = True
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 10.0
    
    def adjust_trade_position( self, trade: Trade, current_rate: float, current_profit: float, **kwargs, ) -> Optional[float]:
        return trade.stake_amount * (1 + current_profit)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for inf_time in self.inf_times:
            for pair in pairs:
                informative_pairs.append((pair, inf_time))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe

        for inf_time in self.inf_times:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_time)
            
            # Directional Indicator
            informative['pdi'] = ta.PLUS_DI(informative)
            informative['mdi'] = ta.MINUS_DI(informative)

            # Bollinger Bands
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            informative['bbu'] = bollinger['upper']
            informative['bbm'] = bollinger['mid']
            informative['bbl'] = bollinger['lower']

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_time, ffill=True)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['signal'] = macd['macdsignal']

        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Directional Indicator
                (dataframe['pdi_15m'] > dataframe['mdi_15m'] ) &
                (dataframe['pdi_5m']  > dataframe['mdi_5m'] ) &
                (dataframe['pdi_15m'] > 25) &
                (dataframe['pdi_5m']  > 25) &
                # Bollinger Bands
                (dataframe['close_15m'] > dataframe['bbm_15m']) &
                (dataframe['close_5m']  > dataframe['bbm_5m']) &
                # MACD
                (dataframe['macd']   > dataframe['signal']) & 
                (dataframe['signal'] > 0) & 
                (dataframe['macd']   > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                # Directional Indicator
                (dataframe['mdi_15m'] > dataframe['pdi_15m'] ) &
                (dataframe['mdi_5m']  > dataframe['pdi_5m'] ) &
                (dataframe['mdi_15m'] > 25) &
                (dataframe['mdi_5m']  > 25) &
                # Bollinger Bands
                (dataframe['close_15m'] < dataframe['bbm_15m']) &
                (dataframe['close_5m']  < dataframe['bbm_5m']) &
                # MACD
                (dataframe['macd']   < dataframe['signal']) & 
                (dataframe['signal'] < 0) & 
                (dataframe['macd']   < 0)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe