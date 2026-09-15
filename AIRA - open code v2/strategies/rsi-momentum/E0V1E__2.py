# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement, trailing-whitespace, unused-import, wrong-import-order, line-too-long, logging-fstring-interpolation
# flake8: noqa: F401
# isort: skip_file
# from freqtrade.strategy.interface import IStrategy ####
import pandas as pd
from pandas import DataFrame
import numpy as np
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative)
from freqtrade.exchange import timeframe_to_prev_date
# from technical.util import resample_to_interval, resampled_merge

# --------------------------------
# Add your lib to import here
import logging
import talib.abstract as ta
import pandas_ta as pta
from datetime import datetime, timedelta
from typing import List, Optional
from technical.indicators import PMAX, SSLChannels, VIDYA
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib


logger = logging.getLogger(__name__)

TMP_HOLD = []
TMP_HOLD1 = []


class E0V1E(IStrategy):
    INTERFACE_VERSION = 3
    STRATEGY_VERSION = 1
    
    startup_candle_count: int = 200
    
    minimal_roi = {
        "0": 1
    }
    
    timeframe = '5m'
    strategy_leverage = 5
    stoploss = -0.25 # disabled
    use_custom_stoploss = False
    max_open_trades = 10
    can_short = False
    
    # Trailing stop:
    trailing_stop = False
    
    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': True,
    }

    @property
    def protections(self):
        
        return [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 18
        }
        ]

    is_optimize_32 = True
    buy_rsi_fast_32 = IntParameter(20, 70, default=40, space='buy', optimize=is_optimize_32)
    buy_rsi_32 = IntParameter(15, 50, default=42, space='buy', optimize=is_optimize_32)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.973, decimals=3, space='buy', optimize=is_optimize_32)
    buy_cti_32 = DecimalParameter(-1, 1, default=0.69, decimals=2, space='buy', optimize=is_optimize_32)

    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)
    # rsi_exit = IntParameter(low=49, high=80, default=70, space='sell', optimize=True)

    cci_opt = True
    sell_loss_cci = IntParameter(low=0, high=600, default=120, space='sell', optimize=cci_opt)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0, default=-0.15, decimals=2, space='sell', optimize=cci_opt)
    

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()


        min_profit = trade.calc_profit_ratio(trade.min_rate)

        if current_candle['close'] > current_candle["ma120"] or current_candle['close'] > current_candle["ma240"]:
            if trade.id not in TMP_HOLD:
                TMP_HOLD.append(trade.id)
        else:
            if trade.id not in TMP_HOLD1:
                TMP_HOLD1.append(trade.id)

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"

        if current_candle["cci"] > 80:
            if current_candle["high"] >= trade.open_rate:
                return "cci_high_sell"
            
        if min_profit <= -0.15:
            if current_profit > self.sell_loss_cci_profit.value:
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell"

        if trade.id in TMP_HOLD and current_candle["close"] < current_candle["ma120"] and current_candle["close"] < \
                current_candle["ma240"]:
            if current_time - timedelta(minutes=12) > trade.open_date_utc:
                TMP_HOLD.remove(trade.id)
                return "ma120_sell"

        if trade.id in TMP_HOLD1:
            if current_candle["high"] > current_candle["ma120"] or current_candle["high"] > current_candle["ma240"]:
                if -0.1<= min_profit <= -0.05:
                    TMP_HOLD1.remove(trade.id)
                    return "cross_120_or_240_sell"

        # if (current_candle['rsi'] >= self.rsi_exit.value) and (previous_candle['rsi'] >= current_candle['rsi']):
        #     return "rsi_reversal"

        return None
    
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # buy_1 indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)
        dataframe['ma240'] = ta.MA(dataframe, timeperiod=240)

        return dataframe
    
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.
    
        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return self.strategy_leverage


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value) &
                (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_1, ['enter_long', 'enter_tag']] = (1, 'buy_1')
        

        buy_new = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 34) &
                (dataframe['rsi'] > 28) &
                (dataframe['close'] < dataframe['sma_15'] * 0.96) &
                (dataframe['cti'] < self.buy_cti_32.value) &
                (dataframe['volume'] > 0)
        )
        dataframe.loc[buy_new, ['enter_long', 'enter_tag']] = (1, 'buy_new')
        
        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe
    