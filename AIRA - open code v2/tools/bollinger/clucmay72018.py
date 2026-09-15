# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.persistence import Order
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa
import pandas as pd

class ClucMay72018(IStrategy):
    
    pair_list = pd.read_csv("/root/freqtrade/user_data/strategies/ClucMay72018.csv",header=None, names=['pair', 'amount'], sep=',')

    total_profit_usdt = 0
    stake_usdt = 100
    """

    author@: Gert Wohlgemuth

    works on new objectify branch!

    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        # "0": 0.1
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    # stoploss = -0.05
    #stoploss = -1 # unlimited
    stoploss = -1

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    # exit_profit_only = True
    ignore_roi_if_entry_signal = False
    
    # Optimal timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['ema100']) &
                    (dataframe['close'] < 0.985 * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume'].rolling(window=30).mean().shift(1) * 99))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband'])
            ),
            'sell'] = 1
        return dataframe


    # This is called when placing the initial order (opening trade)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                             leverage: float, entry_tag: str | None, side: str,
                             **kwargs) -> float:
       
         for i in range(len(self.pair_list)):
             if self.pair_list['pair'].iloc[i] == pair:
                 proposed_stake = self.pair_list['amount'].iloc[i]
                 break

         # We need to leave most of the funds for possible further DCA orders
         # This also applies to fixed stakes
         return proposed_stake


    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Called right after an order fills. 
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if trade.nr_of_successful_exits  == 1:

            buy_usdt = trade.orders[0].price * trade.orders[0].amount
            sell_usdt = trade.orders[1].price * trade.orders[1].amount
            buy_commission_usdt = buy_usdt * 0.001
            sell_commission_usdt = sell_usdt * 0.001
            close_profit_usdt = sell_usdt - buy_usdt
            realized_profit_usdt = close_profit_usdt - (buy_commission_usdt + sell_commission_usdt)

            for i in range(len(self.pair_list)):
                if self.pair_list['pair'].iloc[i] == pair:
                    self.pair_list['amount'].iloc[i] = self.pair_list['amount'].iloc[i] + realized_profit_usdt
                    self.pair_list.to_csv("/root/freqtrade/user_data/strategies/ClucMay72018.csv",header=None, index = None, sep=',')
                    break
        
        return None