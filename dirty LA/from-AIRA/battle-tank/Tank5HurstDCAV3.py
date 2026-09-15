import logging
from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import logging
import os
import numpy as np
import pandas as pd
import warnings
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from technical import qtpylib
from typing import List, Tuple, Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, RealParameter, merge_informative_pair)
from typing import Optional
from functools import reduce
import warnings
import math
pd.options.mode.chained_assignment = None
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from scipy.signal import find_peaks, butter, filtfilt, hilbert


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class Tank5HurstDCAV3(IStrategy):

    '''
          ______   __          __              __    __   ______   __    __        __     __    __             ______            
     /      \ /  |       _/  |            /  |  /  | /      \ /  \  /  |      /  |   /  |  /  |           /      \           
    /$$$$$$  |$$ |____  / $$ |    _______ $$ | /$$/ /$$$$$$  |$$  \ $$ |     _$$ |_  $$ |  $$ |  _______ /$$$$$$  |  _______ 
    $$ |  $$/ $$      \ $$$$ |   /       |$$ |/$$/  $$ ___$$ |$$$  \$$ |    / $$   | $$ |__$$ | /       |$$$  \$$ | /       |
    $$ |      $$$$$$$  |  $$ |  /$$$$$$$/ $$  $$<     /   $$< $$$$  $$ |    $$$$$$/  $$    $$ |/$$$$$$$/ $$$$  $$ |/$$$$$$$/ 
    $$ |   __ $$ |  $$ |  $$ |  $$ |      $$$$$  \   _$$$$$  |$$ $$ $$ |      $$ | __$$$$$$$$ |$$ |      $$ $$ $$ |$$      \ 
    $$ \__/  |$$ |  $$ | _$$ |_ $$ \_____ $$ |$$  \ /  \__$$ |$$ |$$$$ |      $$ |/  |     $$ |$$ \_____ $$ \$$$$ | $$$$$$  |
    $$    $$/ $$ |  $$ |/ $$   |$$       |$$ | $$  |$$    $$/ $$ | $$$ |______$$  $$/      $$ |$$       |$$   $$$/ /     $$/ 
     $$$$$$/  $$/   $$/ $$$$$$/  $$$$$$$/ $$/   $$/  $$$$$$/  $$/   $$//      |$$$$/       $$/  $$$$$$$/  $$$$$$/  $$$$$$$/  
                                                                       $$$$$$/                                               
                                                                                                                             
    '''          

    exit_profit_only = True ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count: int = 200
    stoploss = -0.99
    locked_stoploss = {}
    timeframe = '5m'

    # DCA
    position_adjustment_enable = True
    useDca = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    max_epa = IntParameter(0, 3, default = 1 ,space='buy', optimize=True, load=True) # of additional buys.
    max_dca_multiplier = DecimalParameter(low=1.0, high=1.5, default=1.1, decimals=2 ,space='buy', optimize=True, load=True)
    safety_order_reserve = IntParameter(2, 4, default=2, space='buy', optimize=True)
    filldelay = IntParameter(120, 360, default = 283 ,space='buy', optimize=True, load=True)
    max_entry_position_adjustment = max_epa.value
    ### Custom Functions
    # HA Market Bias
    ha_len = IntParameter(10, 100, default=49, space='buy', optimize=True)
    ha_len2 = IntParameter(10, 100, default=41, space='buy', optimize=True)
    osc_len = IntParameter(5, 21, default=17, space='buy', optimize=True)
    window_size = IntParameter(250, 500, default=266, space='buy', optimize=True)

    # Logic Selection
    use0 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use1 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use5 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use6 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use8 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use9 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use14 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use15 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use16 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use17 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use18 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use19 = BooleanParameter(default=True, space="sell", optimize=True, load=True)


    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # protections
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True, load=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True, load=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)

    locked_stoploss = {}

    minimal_roi = {
    }


    plot_config = {
    "main_plot": {
        "upper_envelope": {
          "color": "#a9c108",
          "type": "line"
        },
        "lower_envelope": {
          "color": "#7264c4"
        },
        "enter_tag": {
          "color": "#97c774"
        },
        "exit_tag": {
          "color": "#f57d6f"
        },
        "dominant_cycle": {
          "color": "#474e30"
        },
        "upper_envelope_h0": {
          "color": "#ad8d7b"
        },
        "lower_envelope_h0": {
          "color": "#ad8d7b",
          "type": "line"
        },
        "upper_envelope_h1": {
          "color": "#db230c"
        },
        "lower_envelope_h1": {
          "color": "#653565",
          "type": "line"
        },
        "upper_envelope_h2": {
          "color": "#b614be",
          "type": "line"
        },
        "lower_envelope_h2": {
          "color": "#af9913"
        }
    },
    "subplots": {
        "Dominant Cycle": {
            "signal": {
            "color": "#4fd4f1",
            "type": "line"
            },
            "signal_MEAN_UP": {
            "color": "#68f90f",
            "type": "line"
            },
            "signal_MEAN_DN": {
            "color": "#f6b3e6",
            "type": "line"
            }
        },
        "move": {
            "cycle_move_mean": {
            "color": "#f11bb1",
            "type": "line"
            },
            "h0_move_mean": {
            "color": "#7b877f"
            },
            "h1_move_mean": {
            "color": "#c48501"
            },
            "h2_move": {
            "color": "#f10257"
            },
            "h2_move_mean": {
            "color": "#57635b"
            }
        }
    }
    }

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot


    ### Custom Functions ###
    # This is called when placing the initial order (opening trade)
    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        dom1 = current_candle['dominant_cycle'] 
        dom2 = previous_candle['dominant_cycle']
        # We need to leave most of the funds for possible further DCA orders
        if dom1 < dom2 and self.useDca.value == True:
            calculated_stake = proposed_stake / (self.max_dca_multiplier.value + self.safety_order_reserve.value) 
            self.dp.send_msg(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
            logger.info(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
        else:
            # increase stake size in bullish enviroments
            calculated_stake = proposed_stake / (self.max_dca_multiplier.value)

        return calculated_stake 


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        last_fill = (current_time - trade.date_last_filled_utc).seconds / 60 
        # print(trade.pair, last_fill, current_time, count_of_entries)

        current_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        dom1 = current_candle['dominant_cycle'] 
        dom2 = previous_candle['dominant_cycle']

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean'] 
        TP2 = current_candle['h0_move_mean'] 
        TP3 = current_candle['cycle_move_mean']
        display_profit = current_profit * 100
        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}% # of Entries: {trade.nr_of_successful_entries}")

        # Profit Based DCA   
        if trade.nr_of_successful_entries == self.max_epa.value + 1:
            return None 
        if current_profit > -TP1:
            return None

        try:
            # This returns first order stake size 
            # Modify the following parameters to enable more levels or different buy size:
            # max_entry_position_adjustment = 3 
            # max_dca_multiplier = 3.5 

            stake_amount = filled_entries[0].cost
            # This then calculates current safety order size
            if (last_fill > self.filldelay.value):
                if (signal == 1 and current_profit < -TP1):
                    if count_of_entries <= 1: 
                        stake_amount = stake_amount * count_of_entries
                    else:
                        stake_amount = stake_amount

                    return stake_amount
        except Exception as exception:
            return None

        return None
    

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        dom1 = current_candle['dominant_cycle'] 
        dom2 = previous_candle['dominant_cycle']
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        SLT0 = current_candle['h2_move_mean'] 
        SLT1 = current_candle['h1_move_mean'] 
        SLT2 = current_candle['h0_move_mean'] 
        SLT3 = current_candle['cycle_move_mean']

        if trade_duration > 720 and trade_duration < 1080: 
            SL1 = SLT1 - SLT0
        else: 
            SL1 = SLT1 - SLT0

        SL2 = SLT2 - SLT1
        SL3 = SLT2 - SLT1
        display_profit = current_profit * 100
        slt0 = SLT0 * 100
        sl0 = SL1 * 100        
        slt1 = SLT1 * 100
        sl1 = SL1 * 100
        slt2 = SLT2 * 100
        sl2 = SL2 * 100
        slt3 = SLT3 * 100
        sl3 = SL3 * 100

        if pair not in self.locked_stoploss:  # No locked stoploss for this pair yet
            if SLT3 is not None and current_profit > SLT3:
                self.locked_stoploss[pair] = SL3
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                return SL2
            elif SLT2 is not None and current_profit > SLT2:
                self.locked_stoploss[pair] = SL2
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                return SL2
            elif SLT1 is not None and current_profit > SLT1:
                self.locked_stoploss[pair] = SL1
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                return SL1
            # Add Lower Trailing during a down trend
            elif SLT0 is not None and current_profit > SLT0 and dom1 < dom2:
                self.locked_stoploss[pair] = SL1
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                return SL1
            else:
                return self.stoploss
        elif pair in self.locked_stoploss:  # Stoploss setting for each pair
            if SLT3 is not None and current_profit > SLT3:
                self.locked_stoploss[pair] = SL3
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                return SL2
            elif SLT2 is not None and current_profit > SLT2:
                self.locked_stoploss[pair] = SL2
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                return SL2
            elif SLT1 is not None and current_profit > SLT1:
                self.locked_stoploss[pair] = SL1
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                return SL1
            # Add Lower Trailing during a down trend
            elif SLT0 is not None and current_profit > SLT0 and dom1 < dom2:
                self.locked_stoploss[pair] = SL1
                self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                logger.info(f'*** {pair} *** Profit {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                return SL1
        else: # Stoploss has been locked for this pair
            self.dp.send_msg(f'*** {pair} *** Profit {display_profit:.3f}% stoploss locked at {self.locked_stoploss[pair]:.4f}')
            logger.info(f'*** {pair} *** Profit {display_profit:.3f}% stoploss locked at {self.locked_stoploss[pair]:.4f}')
            return self.locked_stoploss[pair]
        if current_profit < -.01:
            if pair in self.locked_stoploss:
                del self.locked_stoploss[pair]
                self.dp.send_msg(f'*** {pair} *** Stoploss reset.')
                logger.info(f'*** {pair} *** Stoploss reset.')

        return self.stoploss



    def custom_entry_price(self, pair: str, trade: Optional['Trade'], current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:

        dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                timeframe=self.timeframe)

        entry_price = (dataframe['close'].iat[-1] + dataframe['open'].iat[-1] + proposed_rate + proposed_rate) / 4
        logger.info(f"{pair} Using Entry Price: {entry_price} | close: {dataframe['close'].iat[-1]} open: {dataframe['open'].iat[-1]} proposed_rate: {proposed_rate}") 

        # Check if there is a stored last entry price and if it matches the proposed entry price
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0001:  # Tolerance for floating-point comparison
            entry_price *= self.increment.value # Increment by 0.2%
            logger.info(f"{pair} Incremented entry price: {entry_price} based on previous entry price : {self.last_entry_price}.")

        # Update the last entry price
        self.last_entry_price = entry_price

        return entry_price


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if exit_reason == 'roi' and trade.enter_tag == 'Dump Ending 8':
            return False

        if exit_reason == 'roi' and trade.enter_tag == 'Dump Ending 9':
            return False

        if exit_reason == 'trailing_stop_loss' and last_candle['bull_check'] is not None:
            logger.info(f"{trade.pair} trailing stop temporarily released")
            self.dp.send_msg(f'{trade.pair} trailing stop temporarily released')
            return False

        # Handle freak events

        if exit_reason == 'roi' and trade.calc_profit_ratio(rate) < 0.003:
            logger.info(f"{trade.pair} ROI is below 0")
            # self.dp.send_msg(f'{trade.pair} ROI is below 0')
            return False

        if exit_reason == 'partial_exit' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} partial exit is below 0")
            # self.dp.send_msg(f'{trade.pair} partial exit is below 0')
            return False

        if exit_reason == 'trailing_stop_loss' and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{trade.pair} trailing stop price is below 0")
            # self.dp.send_msg(f'{trade.pair} trailing stop price is below 0')
            return False

        return True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
    
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        dataframe['ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())
        dataframe['ha_tail'] = (heikinashi['close'] - heikinashi['low'])
        dataframe['ha_wick'] = (heikinashi['high'] - heikinashi['close'])

        dataframe['HLC3'] = (heikinashi['high'] + heikinashi['low'] + heikinashi['close'])/3

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        ### HA Market Bias
        # OHLC Smoothing
        dataframe['o_ema'] = ta.EMA(dataframe['open'], timeperiod = self.ha_len.value)
        dataframe['h_ema'] = ta.EMA(dataframe['high'], timeperiod = self.ha_len.value)
        dataframe['l_ema'] = ta.EMA(dataframe['low'], timeperiod = self.ha_len.value)
        dataframe['c_ema'] = ta.EMA(dataframe['close'], timeperiod = self.ha_len.value)

        # Heikin Ashi Calculations
        dataframe['ha_close'] = (dataframe['o_ema'] + dataframe['h_ema'] + dataframe['l_ema'] + dataframe['c_ema']) / 4
        dataframe['xha_open'] = (dataframe['o_ema'] + dataframe['c_ema']) / 2
        dataframe['ha_open'] = (dataframe['xha_open'].shift(1) + dataframe['ha_close'].shift(1)) / 2
        dataframe['ha_high'] = dataframe[['h_ema', 'ha_open', 'ha_close']].max(axis=1)
        dataframe['ha_low'] = dataframe[['l_ema', 'ha_open', 'ha_close']].min(axis=1)

        # Smooth Heikin Ashi Candles
        dataframe['o2'] = ta.EMA(dataframe['ha_open'], timeperiod = self.ha_len2.value)
        dataframe['c2'] = ta.EMA(dataframe['ha_close'], timeperiod = self.ha_len2.value)
        dataframe['h2'] = ta.EMA(dataframe['ha_high'], timeperiod = self.ha_len2.value)
        dataframe['l2'] = ta.EMA(dataframe['ha_low'], timeperiod = self.ha_len2.value)

        # Bias Average
        dataframe['ha_avg'] = (dataframe['h2'] + dataframe['l2']) / 2

        # Oscillator
        dataframe['osc_bias'] = 100 * (dataframe['c2'] - dataframe['o2'])
        dataframe['osc_smooth'] = ta.EMA(dataframe['osc_bias'], timeperiod = self.osc_len.value)
        dataframe['osc'] = dataframe['osc_bias'] - dataframe['osc_smooth']
        dataframe.loc[dataframe['osc_smooth'] > 0, "osc_UP"] = dataframe['osc_smooth']
        dataframe.loc[dataframe['osc_smooth'] < 0, "osc_DN"] = dataframe['osc_smooth']
        dataframe['osc_UP'].ffill()
        dataframe['osc_DN'].ffill()
        dataframe['osc_MEAN_UP'] = dataframe['osc_UP'].mean() * 1.618
        dataframe['osc_MEAN_DN'] = dataframe['osc_DN'].mean() * 1.618

        dataframe.loc[((dataframe['osc_bias'] > dataframe['osc_bias'].shift())), "bull"] = 1
        dataframe.loc[((dataframe['osc_bias'] > 0) & (dataframe['osc'] > dataframe['osc'].shift())), "bullStrengthens"] = 2
        dataframe.loc[((dataframe['osc_bias'] > 0) & (dataframe['osc'] < dataframe['osc'].shift())), "bullWeakens"] = 1
        dataframe.loc[((dataframe['osc_bias'].shift(2) < dataframe['osc_bias'].shift(1)) & (dataframe['osc_bias'] > dataframe['osc_bias'].shift())), "bullChange"] = 1

        dataframe.loc[(dataframe['osc_bias'] < dataframe['osc_bias'].shift()), "bear"] = -1
        dataframe.loc[((dataframe['osc_bias'] < 0) & (dataframe['osc'] < dataframe['osc'].shift())), "bearStrengthens"] = -2
        dataframe.loc[((dataframe['osc_bias'] < 0) & (dataframe['osc'] > dataframe['osc'].shift())), "bearWeakens"] = -1
        dataframe.loc[((dataframe['osc_bias'].shift(2) > dataframe['osc_bias'].shift(1)) & (dataframe['osc_bias'] < dataframe['osc_bias'].shift())), "bearChange"] = -1

        dataframe['bull'] = dataframe['bull'].fillna(0)
        dataframe['bear'] = dataframe['bear'].fillna(0)
        dataframe['bullStrengthens'] = dataframe['bullStrengthens'].fillna(0)
        dataframe['bullWeakens'] = dataframe['bullWeakens'].fillna(0)
        dataframe['bullChange'] = dataframe['bullChange'].fillna(0)
        dataframe['bearStrengthens'] = dataframe['bearStrengthens'].fillna(0)
        dataframe['bearWeakens'] = dataframe['bearWeakens'].fillna(0)
        dataframe['bearChange'] = dataframe['bearChange'].fillna(0)

        # Calculate marketbias
        dataframe['marketbias'] = (
            dataframe['bull'] + dataframe['bullChange'] + dataframe['bullStrengthens'] + 
            dataframe['bullWeakens'] + dataframe['bear'] + dataframe['bearChange'] + 
            dataframe['bearStrengthens'] + dataframe['bearWeakens']
        )

        dataframe['marketbias_sma'] = ta.SMA(dataframe['marketbias'], timeperiod=5)
        dataframe['marketbias_sig'] = ta.SMA(dataframe['marketbias_sma'], timeperiod=21)

        if self.dp.runmode.value in ('dry_run'):
            window_size = self.window_size.value  # Adjust this value as appropriate
        else:
            window_size = None

        if len(dataframe) < self.window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.window_size.value} data points.")


        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(dataframe['OHLC4'], window_size=self.window_size.value)

        if len(freq) == 0 or len(power) == 0:
            raise ValueError("FFT resulted in zero or invalid frequencies. Check the data or the FFT implementation.")

        # Filter out the zero-frequency component and limit the frequency to below 500
        positive_mask = (freq > 0) & (1 / freq < self.window_size.value)
        positive_freqs = freq[positive_mask]
        positive_power = power[positive_mask]

        # Convert frequencies to periods
        cycle_periods = 1 / positive_freqs

        # Set a threshold to filter out insignificant cycles based on power
        power_threshold = 0.01 * np.max(positive_power)
        significant_indices = positive_power > power_threshold
        significant_periods = cycle_periods[significant_indices]
        significant_power = positive_power[significant_indices]

        # Identify the dominant cycle
        dominant_freq_index = np.argmax(significant_power)
        dominant_freq = positive_freqs[dominant_freq_index]
        cycle_period = int(np.abs(1 / dominant_freq)) if dominant_freq != 0 else np.inf

        if cycle_period == np.inf:
            raise ValueError("No dominant frequency found. Check the data or the method used.")

        # Calculate the half-span of the dominant cycle period
        half_span_period = cycle_period // 2

        # Apply an inverse half-span average to the dominant cycle
        dataframe['inverse_half_span_avg'] = 1 / dataframe['OHLC4'].ewm(span=half_span_period).mean()

        # Calculate harmonics for the dominant cycle
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        dataframe['dominant_cycle'] = ta.SMA(dataframe['OHLC4'], timeperiod=cycle_period)
        dataframe['harmonic_1/2'] = ta.SMA(dataframe['OHLC4'], timeperiod=int(harmonics[0]))
        dataframe['harmonic_1/3'] = ta.SMA(dataframe['OHLC4'], timeperiod=int(harmonics[1]))
        dataframe['harmonic_1/4'] = ta.SMA(dataframe['OHLC4'], timeperiod=int(harmonics[2]))
        dataframe['dc_EWM'] = dataframe['OHLC4'].ewm(span=int(cycle_period)).mean()
        dataframe['dc_1/2'] = dataframe['OHLC4'].ewm(span=int(harmonics[0])).mean()
        dataframe['dc_1/3'] = dataframe['OHLC4'].ewm(span=int(harmonics[1])).mean()
        dataframe['dc_1/4'] = dataframe['OHLC4'].ewm(span=int(harmonics[2])).mean()

        if cycle_period > 0:
            dataframe.loc[:, "period_dc"] = cycle_period
            dataframe.loc[:, "period_1/2"] = harmonics[0] 
            dataframe.loc[:, "period_1/3"] = harmonics[1]
            dataframe.loc[:, "period_1/4"] = harmonics[2]

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = dataframe['OHLC4'].rolling(cycle_period) 
        rolling_maxc = rolling_windowc.max()
        rolling_minc = rolling_windowc.min()
        rolling_windowh0 = dataframe['OHLC4'].rolling(int(harmonics[0]))
        rolling_maxh0 = rolling_windowh0.max()
        rolling_minh0 = rolling_windowh0.min()
        rolling_windowh1 = dataframe['OHLC4'].rolling(int(harmonics[1])) 
        rolling_maxh1 = rolling_windowh1.max()
        rolling_minh1 = rolling_windowh1.min()
        rolling_windowh2 = dataframe['OHLC4'].rolling(int(harmonics[2])) 
        rolling_maxh2 = rolling_windowh2.max()
        rolling_minh2 = rolling_windowh2.min()

        dataframe['cycle_avg'] = ((rolling_maxc - rolling_minc) / 2) + rolling_minc 
        dataframe['h0_avg'] = ((rolling_maxh0 - rolling_minh0) / 2) + rolling_minh0 
        dataframe['h1_avg'] = ((rolling_maxh1 - rolling_minh1) / 2) + rolling_minh1 
        dataframe['h2_avg'] = ((rolling_maxh2 - rolling_minh2) / 2) + rolling_minh2 
        dataframe['data_avg'] = ((dataframe['OHLC4'].max() - dataframe['OHLC4'].min()) / 2) + dataframe['OHLC4'].min()

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['cycle_move'] = ptp_valuec / dataframe['OHLC4']
        dataframe['cycle_move_mean'] = dataframe['cycle_move'].mean() 
        dataframe['h0_move'] = ptp_valueh0 / dataframe['OHLC4']
        dataframe['h0_move_mean'] = dataframe['h0_move'].mean()
        dataframe['h1_move'] = ptp_valueh1 / dataframe['OHLC4']
        dataframe['h1_move_mean'] = dataframe['h1_move'].mean() 
        dataframe['h2_move'] = ptp_valueh2 / dataframe['OHLC4']
        dataframe['h2_move_mean'] = dataframe['h2_move'].mean()
        dataframe['h2_move_ema'] = dataframe['h2_move'].ewm(span=9).mean()

        # Add envelopes for the dominant cycle
        dataframe['upper_envelope'] = dataframe['dc_EWM'] * (1 + dataframe['cycle_move_mean'])
        dataframe['lower_envelope'] = dataframe['dc_EWM'] * (1 - dataframe['cycle_move_mean'])
        dataframe['upper_envelope_h0'] = dataframe['dc_1/2'] * (1 + dataframe['h0_move_mean'])
        dataframe['lower_envelope_h0'] = dataframe['dc_1/2'] * (1 - dataframe['h0_move_mean'])
        dataframe['upper_envelope_h1'] = dataframe['dc_1/3'] * (1 + dataframe['h1_move_mean'])
        dataframe['lower_envelope_h1'] = dataframe['dc_1/3'] * (1 - dataframe['h1_move_mean'])
        dataframe['upper_envelope_h2'] = dataframe['dc_1/4'] * (1 + dataframe['h2_move_mean'])
        dataframe['lower_envelope_h2'] = dataframe['dc_1/4'] * (1 - dataframe['h2_move_mean'])

        dataframe['lowerspan'] = ((dataframe['lower_envelope_h2'] - dataframe['lower_envelope_h0']) / dataframe['lower_envelope_h0']) * 100
        dataframe['upperspan'] = ((dataframe['upper_envelope_h0'] - dataframe['upper_envelope_h2']) / dataframe['upper_envelope_h0']) * 100
        dataframe['lowerspan_mean'] = dataframe['lowerspan'].rolling(cycle_period).mean()
        dataframe['upperspan_mean'] = dataframe['upperspan'].rolling(cycle_period).mean()
        dataframe['span ratio'] = dataframe['lowerspan'] / dataframe['upperspan']
        dataframe['span_h2_limit'] = dataframe['h2_move_mean'] * 100

        # Initialize 'signal' column
        dataframe['signal'] = 0
        dataframe['power_lvl'] = 0

        # Generate signals based on each significant period and harmonics
        for period, power_value in zip(significant_periods, significant_power):
            rolling_avg = dataframe['OHLC4'].rolling(window=int(period)).mean()
            deviation = dataframe['OHLC4'] - rolling_avg
            
            # Define thresholds for signals
            upper_threshold = deviation.std() * 1.618
            lower_threshold = -deviation.std() * 1.618
            
            # Generate buy and sell signals
            dataframe.loc[deviation < lower_threshold, 'signal'] += 1
            dataframe.loc[deviation > upper_threshold, 'signal'] -= 1
            dataframe['power_lvl'] += power_value

        # Normalize the 'signal' column for clarity
        dataframe['signal'] = dataframe['signal'] / dataframe['signal'].abs().max()

        # Filter based on power levels
        power_threshold = np.percentile(dataframe['power_lvl'], 75)  # Using 75th percentile as threshold
        dataframe = dataframe[dataframe['power_lvl'] >= power_threshold]

        # Additional calculations for signal mean and EMA
        dataframe['signal_UP'] = np.where(dataframe['signal'] > 0, dataframe['signal'], np.nan)
        dataframe['signal_DN'] = np.where(dataframe['signal'] < 0, dataframe['signal'], np.nan)
        dataframe['signal_UP'] = dataframe['signal_UP'].ffill()
        dataframe['signal_DN'] = dataframe['signal_DN'].ffill()
        dataframe['signal_MEAN_UP'] = dataframe['signal_UP'].mean() * 1.618
        dataframe['signal_MEAN_DN'] = dataframe['signal_DN'].mean() * 1.618
        dataframe['signal_ma'] = ta.EMA(dataframe['signal'], timeperiod=3)

        dataframe['is_zero'] = dataframe['signal'] == 0
        dataframe['group'] = (dataframe['is_zero'] != dataframe['is_zero'].shift()).cumsum()
        dataframe['zero_group_size'] = dataframe.groupby('group')['is_zero'].transform('sum')
        dataframe['dominant_move_soon'] = dataframe['zero_group_size'] >= cycle_period
        dataframe['h1/2_move_soon'] = dataframe['zero_group_size'] >= harmonics[0]
        dataframe['h1/3_move_soon'] = dataframe['zero_group_size'] >= harmonics[1]
        dataframe['h1/4_move_soon'] = dataframe['zero_group_size'] >= harmonics[2]

        dataframe['max'] = dataframe["OHLC4"].max()
        dataframe['min'] = dataframe["OHLC4"].min()
        dataframe['entry_max'] = dataframe['max'] * (1 - dataframe['cycle_move_mean'])
        dataframe['exit_min'] = dataframe['min'] * (1 + dataframe['cycle_move_mean'])
        dataframe["mfi"] = (ta.MFI(dataframe, timeperiod=cycle_period) - 50) * 2
        
        # dataframe['homodyne_dc'] = homodyne_discriminator(dataframe['OHLC4'].values, period=cycle_period)
        # dataframe['homodyne_dc'] = dataframe['OHLC4'].rolling(cycle_period).apply(lambda x: homodyne_discriminator(x.values, cycle_period))
        # dataframe['homodyne_1/2'] = dataframe['OHLC4'].rolling(int(harmonics[0])).apply(lambda x: homodyne_discriminator(x.values, int(harmonics[0])))
        # dataframe['homodyne_1/3'] = dataframe['OHLC4'].rolling(int(harmonics[1])).apply(lambda x: homodyne_discriminator(x.values, int(harmonics[1])))
        # dataframe['homodyne_1/4'] = dataframe['OHLC4'].rolling(int(harmonics[2])).apply(lambda x: homodyne_discriminator(x.values, int(harmonics[2])))

        # WaveTrend using OHLC4 or HA close - 9/12
        ap = (0.333 * (heikinashi['high'] + heikinashi['low'] + heikinashi["close"]))

        dataframe['esa'] = ta.EMA(ap, timeperiod = 10)
        dataframe['d'] = ta.EMA(abs(ap - dataframe['esa']), timeperiod = 10)
        dataframe['wave_ci'] = (ap-dataframe['esa']) / (0.015 * dataframe['d'])
        dataframe['wave_t1'] = ta.EMA(dataframe['wave_ci'], timeperiod = 21)
        dataframe['wave_t2'] = ta.SMA(dataframe['wave_t1'], timeperiod = 4)

        dataframe.loc[dataframe['wave_t1'] > 0, "wave_t1_UP"] = dataframe['wave_t1']
        dataframe.loc[dataframe['wave_t1'] < 0, "wave_t1_DN"] = dataframe['wave_t1']
        dataframe['wave_t1_UP'].ffill()
        dataframe['wave_t1_DN'].ffill()
        dataframe['wave_t1_MEAN_UP'] = dataframe['wave_t1_UP'].mean()
        dataframe['wave_t1_MEAN_DN'] = dataframe['wave_t1_DN'].mean()
        dataframe['wave_t1_UP_FIB'] = dataframe['wave_t1_MEAN_UP'] * 1.618
        dataframe['wave_t1_DN_FIB'] = dataframe['wave_t1_MEAN_DN'] * 1.618
        
        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        full_send1 = (
                (self.use0.value == True) &
                (df['upper_envelope_h2'] > df['cycle_avg']) & 
                (df['upper_envelope'] > df['upper_envelope_h0']) & 
                (df['lower_envelope'] > df['lower_envelope_h2']) & 
                (df['signal'] > df['signal_MEAN_UP']) &
                (df['lower_envelope_h1'] > df['close']) & 
                (df['volume'] > 0)   # Make sure Volume is not 0
        )
            
        df.loc[full_send1, 'enter_long'] = 1
        df.loc[full_send1, 'enter_tag'] = 'Full Send 1'

        full_send2 = (
                (self.use1.value == True) &
                (df['upper_envelope'] > df['upper_envelope_h2']) & 
                (df['cycle_avg'] < df['lower_envelope_h2']) & 
                (df['dominant_cycle'] < df['close']) & 
                (df['signal'] > 0) &
                # (df['lower_envelope_h1'] > df['close']) & 
                (df['volume'] > 0)   # Make sure Volume is not 0
        )
        df.loc[full_send2, 'enter_long'] = 1
        df.loc[full_send2, 'enter_tag'] = 'Full Send 2'

        full_send3 = (
                (self.use2.value == True) &
                (df['dominant_cycle'] < df['lower_envelope_h1']) & 
                (df['h0_move_mean'] < df['h2_move']) & 
                (df['signal'] > df['signal_MEAN_UP']) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send3, 'enter_long'] = 1
        df.loc[full_send3, 'enter_tag'] = 'Full Send 3'

        full_send4 = (
                (self.use3.value == True) &
                (df['upper_envelope_h2'] > df['cycle_avg']) & 
                (df['upper_envelope'] > df['upper_envelope_h0']) & 
                (df['lower_envelope'] > df['lower_envelope_h2']) & 
                (df['signal'] > df['signal_MEAN_UP']) &
                (df['lower_envelope'] > df['close']) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send4, 'enter_long'] = 1
        df.loc[full_send4, 'enter_tag'] = 'Full Send 4'

        full_send5 = (
                (self.use4.value == True) &
                (df['data_avg'].shift() > df['cycle_avg'].shift()) &
                (df['data_avg'] <  df['cycle_avg']) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send5, 'enter_long'] = 1
        df.loc[full_send5, 'enter_tag'] = 'Dominant Avg < Data Avg'

        full_send6 = (
                (self.use5.value == True) &
                (df['OHLC4'] < df['lower_envelope']) & 
                (df['h0_move_mean'] < df['h2_move']) & 
                (df['signal'] > df['signal_MEAN_UP']) &
                (df['dominant_cycle'] < df['dominant_cycle'].shift()) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send6, 'enter_long'] = 1
        df.loc[full_send6, 'enter_tag'] = 'Full Send 6'

        dump_over7 = (              
                (self.use6.value == True) &                
                (df["lowerspan"] < df['lowerspan_mean']) &
                (df['lowerspan'].shift() > df['lowerspan']) &
                (df['cycle_avg'] < df['h0_avg']) &
                (df['h2_move_mean'] < df['h2_move']) & 
                (df['h0_avg'].shift(2) <= df['h0_avg']) & 
                (df['low'] < df['lower_envelope_h1']) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[dump_over7, 'enter_long'] = 1
        df.loc[dump_over7, 'enter_tag'] = 'Bull Below Span Avg'


        up_trend8 = (           
                (self.use7.value == True) &    
                (df['signal'] > df['signal_MEAN_UP']) &
                (df["wave_t1"] < df["wave_t1_DN_FIB"]) &               
                (df['lowerspan'].shift(1) < 0.2) &
                (df['lowerspan'].shift(1) < df['lowerspan'].shift(2)) &
                (df['lowerspan'].shift(1) < df['lowerspan']) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[up_trend8, 'enter_long'] = 1
        df.loc[up_trend8, 'enter_tag'] = 'Dump Ending 8'

        full_send9 = (           
                (self.use8.value == True) &                   
                (df['lowerspan'].shift(1) < 0) &
                (df['lowerspan'].shift(1) < df['lowerspan']) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[full_send9, 'enter_long'] = 1
        df.loc[full_send9, 'enter_tag'] = 'Dump Ending 9'


        df['bull_check'] = None
        up_trend8_idx = df.index[up_trend8]
        for idx in up_trend8_idx:
            period = int(df['period_dc'].loc[idx])
            df.loc[idx:idx+period, 'bull_check'] = df['min'].loc[idx]


        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        # profit_taker1 = (
        #         (self.use10.value == True) &
        #         (df["lowerspan"] < df['lowerspan_mean']) &
        #         (df['lowerspan'].shift() > df['lowerspan_mean'].shift()) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        # )
        # df.loc[profit_taker1, 'exit_long'] = 1
        # df.loc[profit_taker1, 'exit_tag'] = 'Profit Taker 1'

        profit_taker2 = (
                (self.use11.value == True) &
                (df['signal'] < df['signal_MEAN_DN']) &
                (df['harmonic_1/4'] > df['upper_envelope']) &
                (df['upper_envelope'] > df['harmonic_1/3']) &  
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker2, 'exit_long'] = 1
        df.loc[profit_taker2, 'exit_tag'] = 'Profit Taker 2'

        profit_taker3 = (
                (self.use12.value == True) &
                (df['signal'] < df['signal_MEAN_DN']) &
                (df['signal'].iloc[-3] > df['signal_MEAN_DN'].iloc[-3]) &
                (df['harmonic_1/2'] > df['dominant_cycle']) &
                (df['harmonic_1/2'].shift() < df['dominant_cycle'].shift()) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker3, 'exit_long'] = 1
        df.loc[profit_taker3, 'exit_tag'] = 'Profit Taker 3'

        profit_taker4 = (
                (self.use13.value == True) &
                (df['upper_envelope_h1'] > df['upper_envelope']) & 
                (df['upper_envelope_h0'] > df['upper_envelope_h1']) & 
                (df['upper_envelope_h0'] > df['upper_envelope']) &
                (df['upper_envelope_h0'].shift() < df['upper_envelope_h1'].shift()) & 
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker4, 'exit_long'] = 1
        df.loc[profit_taker4, 'exit_tag'] = 'Profit Taker 4'

        profit_taker5 = (
                (self.use14.value == True) &
                (df['cycle_move_mean'] < df['h2_move']) & 
                (df['close'] > df['upper_envelope']) & 
                (df['upperspan'] > df['upperspan'].shift()) & 

                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker5, 'exit_long'] = 1
        df.loc[profit_taker5, 'exit_tag'] = 'Profit Taker 5'

        # profit_taker6 = (
        #         (self.use15.value == True) &
        #         (df['signal'] < df['signal_MEAN_UP']) &               
        #         (df['upperspan'].shift(1) > df['upperspan']) &
        #         (df['upperspan'].shift(1) > df['upperspan'].shift(2)) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        # )
        # df.loc[profit_taker6, 'exit_long'] = 1
        # df.loc[profit_taker6, 'exit_tag'] = 'Profit Taker 6'

        # profit_taker6 = (
        #         (self.use16.value == True) &
        #         (df["signal"] < df['signal_MEAN_DN']) &
        #         (df["move"] > df['move_mean']) &
        #         # (df['total_bullish_divergences'].shift() > 0) &
        #         (df["willr14"] > -15) &
        #         (df["OHLC4"] > df['200sma']) &
        #         (df['volume'] > 0)   # Make sure Volume is not 0

        # )
        # df.loc[profit_taker6, 'exit_long'] = 1
        # df.loc[profit_taker6, 'exit_tag'] = 'Profit Taker 6'

        profit_taker7 = (
                (self.use17.value == True) &
                (df['signal'] < df['signal_MEAN_DN']) &
                (df["wave_t1"] > df["wave_t1_UP_FIB"]) &               
                (df['upperspan'].shift(1) < 0.2) &
                (df['upperspan'].shift(1) < df['upperspan'].shift(2)) &
                (df['volume'] > 0)   # Make sure Volume is not 0

        )
        df.loc[profit_taker7, 'exit_long'] = 1
        df.loc[profit_taker7, 'exit_tag'] = 'Profit Taker 7'

        # is_exiting = ( 
        #     (df['exiting'].shift() == 1) &
        #     (df['exiting'] !=1 )
        #     )
        # df.loc[is_exiting, 'exit_long'] = 1

        df['bear_check'] = None
        profit_taker7_idx = df.index[profit_taker7]
        for idx in profit_taker7_idx:
            period = int(df['period_dc'].loc[idx])
            df.loc[idx:idx+period, 'bear_check'] = df['max'].loc[idx]


        return df


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def get_distance(p1, p2):
    return (p1) - (p2)

def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc

def perform_fft(price_data, window_size=None):
    if window_size is not None:
        # Apply rolling window to smooth the data
        price_data = price_data.rolling(window=window_size, center=True).mean().dropna()

    normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
    n = len(normalized_data)
    fft_data = np.fft.fft(normalized_data)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_data) ** 2
    power[np.isinf(power)] = 0
    return freq, power

# Function to calculate upper and lower envelopes
def calculate_envelopes(data, period, percent):
    rolling_mean = ta.SMA(data, timeperiod=period)
    envelope_upper = rolling_mean * (1 + percent / 100)
    envelope_lower = rolling_mean * (1 - percent / 100)
    return envelope_upper, envelope_lower

def homodyne_discriminator(time_series, period):
    analytic_signal = hilbert(time_series)
    phase = np.angle(analytic_signal)
    phase_unwrapped = np.unwrap(phase)
    phase_trend = np.polyfit(np.arange(len(time_series)), phase_unwrapped, 1)[0]
    return np.sin(2 * np.pi * period * phase_trend)