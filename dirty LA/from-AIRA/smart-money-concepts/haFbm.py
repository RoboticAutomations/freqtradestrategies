import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy
from freqtrade.exchange import timeframe_to_minutes
from datetime import datetime
import scipy as sp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import talib.abstract as ta
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
import time
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
from scipy.signal import find_peaks, butter, filtfilt
# from smartmoneyconcepts import smc 


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class haFbm(IStrategy):
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

    exit_profit_only = False ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False
    ignore_roi_if_entry_signal = True
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True
    startup_candle_count: int = 200
    stoploss = -0.20
    locked_stoploss = {}
    timeframe = '5m'

    # DCA
    position_adjustment_enable = True
    max_epa = IntParameter(0, 3, default = 3 ,space='buy', optimize=True, load=True) # of additional buys.
    max_dca_multiplier = DecimalParameter(low=1.0, high=1.5, default=1.1, decimals=1 ,space='buy', optimize=True, load=True)
    use_static = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    filldelay = IntParameter(5, 300, default = 30 ,space='buy', optimize=True, load=True)
    max_entry_position_adjustment = max_epa.value

    # Stake size adjustments
    stake0 = DecimalParameter(low=0.33, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake1 = DecimalParameter(low=0.33, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake2 = DecimalParameter(low=0.33, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake3 = DecimalParameter(low=0.33, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)

    ### Custom Functions
    # Threshold and Limits
    dc_x = DecimalParameter(low=3.0, high=5.0, default=4.5, decimals=1 ,space='buy', optimize=True, load=True)
    dc_0 = DecimalParameter(low=3.0, high=15.0, default=6.5, decimals=1 ,space='buy', optimize=True, load=True)
    dc_1 = DecimalParameter(low=3.0, high=15.0, default=7.5, decimals=1 ,space='buy', optimize=True, load=True)
    dc_2 = DecimalParameter(low=3.0, high=15.0, default=8.5, decimals=1 ,space='buy', optimize=True, load=True)

    fs1 = DecimalParameter(low=1.0, high=10.0, default=2.0, decimals=1 ,space='buy', optimize=True, load=True)

    pt1 = DecimalParameter(low=0.1, high=10.0, default=2.0, decimals=1 ,space='sell', optimize=True, load=True)
    pt2 = DecimalParameter(low=0.1, high=10.0, default=10.0, decimals=1 ,space='sell', optimize=True, load=True)
    pt3 = DecimalParameter(low=0.1, high=0.50, default=0.20, decimals=2 ,space='sell', optimize=True, load=True)
    pt4 = DecimalParameter(low=0.05, high=0.20, default=0.10, decimals=2 ,space='sell', optimize=True, load=True)

    # Logic Selection
    use0 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use1 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use2 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    use3 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use4 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use5 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use6 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use7 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # use8 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    # use9 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use10 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use11 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use12 = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    use13 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

    # Custom Entry
    increment = DecimalParameter(low=1.0005, high=1.002, default=1.001, decimals=4 ,space='buy', optimize=True, load=True)
    last_entry_price = None

    # protections
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True, load=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True, load=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True, load=True)
    use_stop1 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop2 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop3 = BooleanParameter(default=False, space="protection", optimize=True, load=True)
    use_stop4 = BooleanParameter(default=False, space="protection", optimize=True, load=True)

    locked_stoploss = {}
    minimal_roi = {}

    plot_config = {
    "main_plot": {
        "enter_tag": {
          "color": "#97c774"
        },
        "exit_tag": {
          "color": "#f57d6f"
        },
        "upper_envelope_h0": {
          "color": "#ad8d7b"
        },
        "lower_envelope_h0": {
          "color": "#ad8d7b",
          "type": "line"
        },
    },
    "subplots": {
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
                "only_per_pair": True
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
        EP0 = current_candle['lower_envelope'] 
        EP1 = current_candle['lower_envelope_h0'] 
        EP2 = current_candle['lower_envelope_h1'] 
        EP3 = current_candle['lower_envelope_h2']

        if self.use_static.value == 'True':
            return (self.wallets.get_total_stake_amount() / self.config["max_open_trades"])

        # We need to leave most of the funds for possible further DCA orders
        if current_rate < EP0:
            # increase stake size in bullish enviroments
            calculated_stake = proposed_stake / self.max_dca_multiplier.value
        elif current_rate < EP1 and current_rate > EP0:     
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake3.value
            self.dp.send_msg(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
            logger.info(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
        elif current_rate < EP2 and current_rate > EP1:     
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake2.value
            self.dp.send_msg(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
            logger.info(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
        elif current_rate < EP2 and current_rate > EP3:     
            calculated_stake = (proposed_stake / self.max_dca_multiplier.value) * self.stake1.value
            self.dp.send_msg(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
            logger.info(f'*** {pair} *** DCA MODE!!! Stake Amount: ${proposed_stake} reduced to {calculated_stake}')
        else:
            # increase stake size in bullish enviroments
            calculated_stake = proposed_stake / (self.max_dca_multiplier.value) * self.stake0.value

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

        current_candle = dataframe.iloc[-1].squeeze()

        TP0 = current_candle['h2_move_mean'] 
        TP1 = current_candle['h1_move_mean'] 
        TP2 = current_candle['h0_move_mean'] 
        TP3 = current_candle['cycle_move_mean']
        display_profit = current_profit * 100
        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}% # of Entries: {trade.nr_of_successful_entries}")
        # Take Profit if m00n
        if current_profit > TP2 and trade.nr_of_successful_exits == 0:
            # Take quarter of the profit at next fib%
            return -(trade.stake_amount / 4)
        if current_profit > TP3 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at last fib%
            return -(trade.stake_amount / 2)
            

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
                    if count_of_entries >= 1: 
                        stake_amount = stake_amount * 2
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
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        SLT0 = current_candle['h2_move_mean'] 
        SLT1 = current_candle['h1_move_mean'] 
        SLT2 = current_candle['h0_move_mean'] 
        SLT3 = current_candle['cycle_move_mean']
        bias = current_candle['market_dir']

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

        if bias < 1:
            if pair not in self.locked_stoploss:  # No locked stoploss for this pair yet
                if SLT3 is not None and current_profit > SLT3:
                    self.locked_stoploss[pair] = SL3
                    self.dp.send_msg(f'*** {pair} *** Profit 4 {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 4 {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                    return SL2
                elif SLT2 is not None and current_profit > SLT2:
                    self.locked_stoploss[pair] = SL2
                    self.dp.send_msg(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    return SL2
                elif SLT1 is not None and current_profit > SLT1:
                    self.locked_stoploss[pair] = SL1
                    self.dp.send_msg(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    return SL1
                elif SLT0 is not None and current_profit > SLT0:
                    self.locked_stoploss[pair] = SL1
                    self.dp.send_msg(f'*** {pair} *** Profit 1 {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 1 {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                    return SL1
                else:
                    return self.stoploss
            elif pair in self.locked_stoploss:  # Stoploss setting for each pair
                if SLT3 is not None and current_profit > SLT3:
                    self.locked_stoploss[pair] = SL3
                    self.dp.send_msg(f'*** {pair} *** Profit 4 {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 4 {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                    return SL2
                elif SLT2 is not None and current_profit > SLT2:
                    self.locked_stoploss[pair] = SL2
                    self.dp.send_msg(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                    return SL2
                elif SLT1 is not None and current_profit > SLT1:
                    self.locked_stoploss[pair] = SL1
                    self.dp.send_msg(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                    return SL1
                elif SLT0 is not None and current_profit > SLT0:
                    self.locked_stoploss[pair] = SL1
                    self.dp.send_msg(f'*** {pair} *** Profit 1 {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
                    logger.info(f'*** {pair} *** Profit 1 {display_profit:.3f}% - {slt0:.3f}%/{sl0:.3f}% activated')
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
    window_size = IntParameter(250, 500, default=266, space='buy', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        start_time = time.time()
        pair = metadata['pair']
        ha_df = dataframe.copy()
        ha_df['zero'] = 0
        ha_df['ha_close'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        ha_df['ha_open'] = ha_df['ha_close'].shift(1)

        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        ha_df['ha_trend'] = (ha_df['ha_high'] > ha_df['ha_low']) & (ha_df['ha_close'] > ha_df['ha_open'])

        if self.dp.runmode.value in ('dry_run'):
            window_size = self.window_size.value  # Adjust this value as appropriate
        else:
            window_size = None

        if len(dataframe) < self.window_size.value:
            raise ValueError(f"Insufficient data points for FFT: {len(dataframe)}. Need at least {self.window_size.value} data points.")


        # Perform FFT to identify cycles with a rolling window
        freq, power = perform_fft(ha_df['ha_close'], window_size=self.window_size.value)

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

        # Calculate harmonics for the dominant cycle
        harmonics = [cycle_period / (i + 1) for i in range(1, 4)]
        ha_df['dc_EWM'] = ha_df['ha_close'].ewm(span=int(cycle_period)).mean()
        ha_df['dc_1/2'] = ha_df['ha_close'].ewm(span=int(harmonics[0])).mean()
        ha_df['dc_1/3'] = ha_df['ha_close'].ewm(span=int(harmonics[1])).mean()
        ha_df['dc_1/4'] = ha_df['ha_close'].ewm(span=int(harmonics[2])).mean()

        # Calculate the rolling difference (ΔY)
        ha_df['diff'] = ha_df['dc_EWM'].diff(periods=2)
        delta_x = 2
        
        # Calculate the slope
        ha_df['slope'] = ha_df['diff'] / delta_x
        
        # Convert slope to angle (in radians), then convert to degrees
        ha_df['angle'] = np.degrees(np.arctan(ha_df['slope']))

        # Fractional Brownian Motion (fBm)
        n = len(ha_df['dc_EWM'])
        h = 0.5
        t = 1
        dt = t / n
        fBm = np.zeros(n)
        for i in range(1, n):
            fBm[i] = fBm[i-1] + np.sqrt(dt) * np.random.normal(0, 1)
        ha_df['fBm'] = fBm * (dt ** h)

        # Fractional differentiation
        ha_df['frac_diff'] = np.gradient(ha_df['ha_close'])
        ha_df['fBm_mean'] = np.mean(ha_df['fBm']) + 2 * np.std(ha_df['fBm'])
        ha_df['frac_sma_dc'] = ha_df['frac_diff'].ewm(span=int(cycle_period)).mean()
        ha_df['signal_UP_dc'] = np.where(ha_df['frac_sma_dc'] > 0, ha_df['frac_sma_dc'], np.nan)
        ha_df['signal_DN_dc'] = np.where(ha_df['frac_sma_dc'] < 0, ha_df['frac_sma_dc'], np.nan)
        ha_df['signal_UP_dc'] = ha_df['signal_UP_dc'].ffill()
        ha_df['signal_DN_dc'] = ha_df['signal_DN_dc'].ffill()
        if self.dp.runmode.value in ('live', 'dry_run'):
            ha_df['signal_MEAN_UP_dc'] = ha_df['signal_UP_dc'].rolling(int(cycle_period)).mean() * self.dc_x.value
            ha_df['signal_MEAN_DN_dc'] = ha_df['signal_DN_dc'].rolling(int(cycle_period)).mean() * self.dc_x.value
        else:
            # this step is for hyperopting and backtesting to simulate the rolling window of the exchange in live mode.
            # this needs to be a value specific for YOUR exchange and/or timeframe.
            ha_df['signal_MEAN_UP_dc'] = ha_df['signal_UP_dc'].rolling(1700).mean() * self.dc_x.value
            ha_df['signal_MEAN_DN_dc'] = ha_df['signal_DN_dc'].rolling(1700).mean() * self.dc_x.value

        ha_df['frac_diff_0'] = np.gradient(ha_df['ha_close'])
        ha_df['fBm_mean'] = np.mean(ha_df['fBm']) + 2 * np.std(ha_df['fBm'])
        ha_df['frac_sma_0'] = ha_df['frac_diff_0'].ewm(span=int(harmonics[0])).mean()
        ha_df['signal_UP_0'] = np.where(ha_df['frac_sma_0'] > 0, ha_df['frac_sma_0'], np.nan)
        ha_df['signal_DN_0'] = np.where(ha_df['frac_sma_0'] < 0, ha_df['frac_sma_0'], np.nan)
        ha_df['signal_UP_0'] = ha_df['signal_UP_0'].ffill()
        ha_df['signal_DN_0'] = ha_df['signal_DN_0'].ffill()
        if self.dp.runmode.value in ('live', 'dry_run'):
            ha_df['signal_MEAN_UP_0'] = ha_df['signal_UP_0'].mean() * self.dc_0.value
            ha_df['signal_MEAN_DN_0'] = ha_df['signal_DN_0'].mean() * self.dc_0.value
        else:
            # this step is for hyperopting and backtesting to simulate the rolling window of the exchange in live mode.
            # this needs to be a value specific for YOUR exchange and/or timeframe.
            ha_df['signal_MEAN_UP_0'] = ha_df['signal_UP_0'].rolling(1700).mean() * self.dc_0.value
            ha_df['signal_MEAN_DN_0'] = ha_df['signal_DN_0'].rolling(1700).mean() * self.dc_0.value

        ha_df['frac_diff_1'] = np.gradient(ha_df['ha_close'])
        ha_df['fBm_mean'] = np.mean(ha_df['fBm']) + 2 * np.std(ha_df['fBm'])
        ha_df['frac_sma_1'] = ha_df['frac_diff_1'].ewm(span=int(harmonics[1])).mean()
        ha_df['signal_UP_1'] = np.where(ha_df['frac_sma_1'] > 0, ha_df['frac_sma_1'], np.nan)
        ha_df['signal_DN_1'] = np.where(ha_df['frac_sma_1'] < 0, ha_df['frac_sma_1'], np.nan)
        ha_df['signal_UP_1'] = ha_df['signal_UP_1'].ffill()
        ha_df['signal_DN_1'] = ha_df['signal_DN_1'].ffill()
        if self.dp.runmode.value in ('live', 'dry_run'):
            ha_df['signal_MEAN_UP_1'] = ha_df['signal_UP_1'].mean() * self.dc_1.value
            ha_df['signal_MEAN_DN_1'] = ha_df['signal_DN_1'].mean() * self.dc_1.value
        else:
            # this step is for hyperopting and backtesting to simulate the rolling window of the exchange in live mode.
            # this needs to be a value specific for YOUR exchange and/or timeframe.
            ha_df['signal_MEAN_UP_1'] = ha_df['signal_UP_1'].rolling(1700).mean() * self.dc_1.value
            ha_df['signal_MEAN_DN_1'] = ha_df['signal_DN_1'].rolling(1700).mean() * self.dc_1.value

        ha_df['frac_diff_2'] = np.gradient(ha_df['ha_close'])
        ha_df['fBm_mean'] = np.mean(ha_df['fBm']) + 2 * np.std(ha_df['fBm'])
        ha_df['frac_sma_2'] = ha_df['frac_diff_2'].ewm(span=int(harmonics[2])).mean()
        ha_df['signal_UP_2'] = np.where(ha_df['frac_sma_2'] > 0, ha_df['frac_sma_2'], np.nan)
        ha_df['signal_DN_2'] = np.where(ha_df['frac_sma_2'] < 0, ha_df['frac_sma_2'], np.nan)
        ha_df['signal_UP_2'] = ha_df['signal_UP_2'].ffill()
        ha_df['signal_DN_2'] = ha_df['signal_DN_2'].ffill()
        if self.dp.runmode.value in ('live', 'dry_run'):
            ha_df['signal_MEAN_UP_2'] = ha_df['signal_UP_2'].mean() * self.dc_2.value
            ha_df['signal_MEAN_DN_2'] = ha_df['signal_DN_2'].mean() * self.dc_2.value
        else:
            # this step is for hyperopting and backtesting to simulate the rolling window of the exchange in live mode.
            # this needs to be a value specific for YOUR exchange and/or timeframe.
            ha_df['signal_MEAN_UP_2'] = ha_df['signal_UP_2'].rolling(1700).mean() * self.dc_2.value
            ha_df['signal_MEAN_DN_2'] = ha_df['signal_DN_2'].rolling(1700).mean() * self.dc_2.value

        # Apply rolling window operation to the 'OHLC4' column
        rolling_windowc = ha_df['ha_close'].rolling(cycle_period) 
        rolling_windowh0 = ha_df['ha_close'].rolling(int(harmonics[0]))
        rolling_windowh1 = ha_df['ha_close'].rolling(int(harmonics[1])) 
        rolling_windowh2 = ha_df['ha_close'].rolling(int(harmonics[2])) 

        # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_valuec = rolling_windowc.apply(lambda x: np.ptp(x))
        ptp_valueh0 = rolling_windowh0.apply(lambda x: np.ptp(x))
        ptp_valueh1 = rolling_windowh1.apply(lambda x: np.ptp(x))
        ptp_valueh2 = rolling_windowh2.apply(lambda x: np.ptp(x))

        # Assign the calculated peak-to-peak value to the DataFrame column
        ha_df['cycle_move'] = ptp_valuec / ha_df['ha_close']
        ha_df['h0_move'] = ptp_valueh0 / ha_df['ha_close']
        ha_df['h1_move'] = ptp_valueh1 / ha_df['ha_close']
        ha_df['h2_move'] = ptp_valueh2 / ha_df['ha_close']

        if self.dp.runmode.value in ('live', 'dry_run'):
            ha_df['cycle_move_mean'] = ha_df['cycle_move'].mean()        
            ha_df['h0_move_mean'] = ha_df['h0_move'].mean()
            ha_df['h1_move_mean'] = ha_df['h1_move'].mean() 
            ha_df['h2_move_mean'] = ha_df['h2_move'].mean()
        else:
            # this step is for hyperopting and backtesting to simulate the rolling window of the exchange in live mode.
            # this needs to be a value specific for YOUR exchange and/or timeframe.
            ha_df['cycle_move_mean'] = ha_df['cycle_move'].rolling(1700).mean()        
            ha_df['h0_move_mean'] = ha_df['h0_move'].rolling(1700).mean()
            ha_df['h1_move_mean'] = ha_df['h1_move'].rolling(1700).mean() 
            ha_df['h2_move_mean'] = ha_df['h2_move'].rolling(1700).mean()

        # Add envelopes for the dominant cycle
        ha_df['upper_envelope'] = ha_df['dc_EWM'] * (1 + ha_df['cycle_move_mean'])
        ha_df['lower_envelope'] = ha_df['dc_EWM'] * (1 - ha_df['cycle_move_mean'])
        ha_df['upper_envelope_h0'] = ha_df['dc_EWM'] * (1 + ha_df['h0_move_mean'])
        ha_df['lower_envelope_h0'] = ha_df['dc_EWM'] * (1 - ha_df['h0_move_mean'])
        ha_df['upper_envelope_h1'] = ha_df['dc_EWM'] * (1 + ha_df['h1_move_mean'])
        ha_df['lower_envelope_h1'] = ha_df['dc_EWM'] * (1 - ha_df['h1_move_mean'])
        ha_df['upper_envelope_h2'] = ha_df['dc_EWM'] * (1 + ha_df['h2_move_mean'])
        ha_df['lower_envelope_h2'] = ha_df['dc_EWM'] * (1 - ha_df['h2_move_mean'])

        ha_df['market_bias'] = (ha_df['signal_MEAN_UP_dc'] / abs(ha_df['signal_MEAN_DN_dc']))
        ha_df['market_dir'] = np.where((ha_df['market_bias'] > ha_df['market_bias'].shift()), 1, np.where((ha_df['market_bias'] < ha_df['market_bias'].shift()), -1, 0))  
        ha_df['market_dir'] = np.where((ha_df['market_bias'] < 1), 0, ha_df['market_dir'])      
        market_bias = ha_df['market_bias'].iloc[-1]

        if ha_df['market_dir'].iloc[-1] is not None:
            if ha_df['market_dir'].iloc[-1] == 1:
                market_dir = 'Uptrend'
            if ha_df['market_dir'].iloc[-1] == 0:
                market_dir = 'Neutral'
            if ha_df['market_dir'].iloc[-1] == -1:
                market_dir = 'Downtrend'

        # # Calculate Fair Value Gap (FVG)
        # fvg_results = smc.fvg(ha_df)
        # # print(fvg_results)
        # ha_df['FVG'] = fvg_results['FVG']
        # ha_df['FVG_Top'] = fvg_results['Top']
        # ha_df['FVG_Bottom'] = fvg_results['Bottom']
        # ha_df['FVG_MitigatedIndex'] = fvg_results['MitigatedIndex']

        # # Calculate Swing Highs and Lows
        # swing_results = smc.swing_highs_lows(ohlc=ha_df, swing_length=5)
        # # print(swing_results)
        # ha_df['HighLow'] = swing_results['HighLow']
        # ha_df['Swing_Level'] = swing_results['Level']

        # # Calculate Break of Structure (BOS) and Change of Character (CHOCH)
        # bos_choch_results = smc.bos_choch(ha_df, swing_results)
        # # print(bos_choch_results)
        # ha_df['BOS'] = bos_choch_results['BOS']
        # ha_df['CHOCH'] = bos_choch_results['CHOCH']
        # ha_df['BOS_Level'] = bos_choch_results['Level']
        # ha_df['BOS_BrokenIndex'] = bos_choch_results['BrokenIndex']

        # # Calculate Order Blocks (OB)
        # ob_results = smc.ob(ha_df, swing_results, close_mitigation=False)
        # # print(ob_results)
        # ha_df['OB'] = ob_results['Order Block']
        # ha_df['OB_Top'] = ob_results['Top']
        # ha_df['OB_Bottom'] = ob_results['Bottom']
        # ha_df['OB_Volume'] = ob_results['Volume']
        # ha_df['OB_Percentage'] = ob_results['Percentage']
        # ha_df['OB_MitigatedIndex'] = ob_results['Mitigated Index']
        # ha_df['OB_Exit'] = np.where(((ha_df['OB'] == -1) & (ha_df['OB'].shift(-1).fillna(0) == 0)), -1, 0)
        # ha_df['OB_Entry'] = np.where(((ha_df['OB'] == 1) & (ha_df['OB'].shift(-1).fillna(0) == 0)), 1, 0)

        # # Calculate Liquidity
        # liquidity_results = smc.liquidity(ha_df, ob_results)
        # ha_df['Liquidity'] = liquidity_results['Liquidity']
        # ha_df['Liquidity_Change'] = np.where(((ha_df['Liquidity'] == 0) & (ha_df['Liquidity'].shift(-1) > 0)), 1, 0)
        # ha_df['_Entry'] = ha_df['Liquidity_Change'] + ha_df['OB_Entry']
        # ha_df['_Exit'] = -(ha_df['Liquidity_Change']) + ha_df['OB_Exit']

        # # Calculate Liquidity
        # liquidity_results = smc.liquidity(ha_df, ob_results)
        # # print(liquidity_results)
        # ha_df['Liquidity'] = liquidity_results['Liquidity']

        logger.info(f'{pair} - Bias: {market_bias:.2f} - {market_dir} | DC: {cycle_period:.2f} | 1/2: {harmonics[0]:.2f} | 1/3: {harmonics[1]:.2f} | 1/4: {harmonics[2]:.2f}')
        end_time = time.time()
        logger.info(f"Indicators done for {pair} in {end_time - start_time:.2f} secs")


        return ha_df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        market_bias = dataframe['market_bias'].iloc[-1]

        full_send1 = (
                (self.use0.value == True) &
                (dataframe['frac_sma_dc'].shift() < dataframe['signal_MEAN_DN_dc']) &
                (dataframe['frac_sma_dc'] > dataframe['signal_MEAN_DN_dc']) &
                (dataframe['market_bias'] < self.fs1.value) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send1, 'enter_long'] = 1
        dataframe.loc[full_send1, 'enter_tag'] = f'Full Send 1 - {market_bias:.3f}'

        full_send2 = (
                (self.use1.value == True) &
                (dataframe['frac_sma_0'].shift() < dataframe['signal_MEAN_DN_dc']) &
                (dataframe['frac_sma_0'] > dataframe['signal_MEAN_DN_dc']) &   
                (dataframe['h2_move'] > dataframe['h2_move_mean']) &                
                (dataframe['market_bias'] > dataframe['market_bias'].shift()) &    
                (dataframe['market_dir'] == 1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send2, 'enter_long'] = 1
        dataframe.loc[full_send2, 'enter_tag'] = f'Full Send 2 - {market_bias:.3f}'

        # full_send3 = (
        #         (self.use2.value == True) &
        #         (dataframe['frac_sma_1'].shift() < dataframe['signal_MEAN_DN_dc']) &
        #         (dataframe['frac_sma_1'] > dataframe['signal_MEAN_DN_dc']) &
        #         (dataframe['market_bias'] < 3.0) &
        #         (dataframe['market_bias'] > 1.5) &
        #         (dataframe['volume'] > 0)   # Make sure Volume is not 0
        # )
            
        # dataframe.loc[full_send3, 'enter_long'] = 1
        # dataframe.loc[full_send3, 'enter_tag'] = f'Full Send 3 - {market_bias:.3f}'

        # full_send4 = (
        #         (self.use3.value == True) &
        #         (dataframe['frac_sma_2'].shift() < dataframe['signal_MEAN_DN_dc']) &
        #         (dataframe['frac_sma_2'] > dataframe['signal_MEAN_DN_dc']) &
        #         # (dataframe['market_bias'] < 3.0) &
        #         (dataframe['market_bias'] > 2.5) &
        #         (dataframe['volume'] > 0)   # Make sure Volume is not 0
        # )
            
        # dataframe.loc[full_send4, 'enter_long'] = 1
        # dataframe.loc[full_send4, 'enter_tag'] = f'Full Send 4 - {market_bias:.3f}'


        # dataframe.loc[
        #     (
        #         (dataframe['fBm'] < np.mean(dataframe['fBm']) - 2 * np.std(dataframe['fBm'])) &
        #         (dataframe['frac_diff'] < 0)
        #     ),
        #     'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        market_bias = dataframe['market_bias'].iloc[-1]

        profit1 = (
                (self.use10.value == True) &
                (dataframe['frac_sma_dc'].shift() > dataframe['signal_MEAN_UP_dc']) &
                (dataframe['frac_sma_dc'] < dataframe['signal_MEAN_UP_dc']) &
                (dataframe['h2_move'] > dataframe['h0_move_mean']) &  
                (dataframe['market_bias'] > self.pt1.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit1, 'exit_long'] = 1
        dataframe.loc[profit1, 'exit_tag'] = f'Profit 1 - {market_bias:.3f}'

        profit2 = (
                (self.use11.value == True) &
                (dataframe['frac_sma_0'].shift() > dataframe['signal_MEAN_UP_dc']) &
                (dataframe['frac_sma_0'] < dataframe['signal_MEAN_UP_dc']) &
                (dataframe['h2_move'] > dataframe['h0_move_mean']) & 
                (dataframe['market_bias'] > self.pt2.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit2, 'exit_long'] = 1
        dataframe.loc[profit2, 'exit_tag'] = f'Profit 2 - {market_bias:.3f}'

        profit3 = (
                (self.use12.value == True) &
                (dataframe['frac_sma_1'].shift() > dataframe['signal_MEAN_UP_dc']) &
                (dataframe['frac_sma_1'] < dataframe['signal_MEAN_UP_dc']) &
                (dataframe['market_bias'] < self.pt3.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit3, 'exit_long'] = 1
        dataframe.loc[profit3, 'exit_tag'] = f'Profit 3 - {market_bias:.3f}'

        profit4 = (
                (self.use13.value == True) &
                (dataframe['frac_sma_2'].shift() > dataframe['signal_MEAN_UP_dc']) &
                (dataframe['frac_sma_2'] < dataframe['signal_MEAN_UP_dc']) &
                (dataframe['market_bias'] < self.pt4.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit4, 'exit_long'] = 1
        dataframe.loc[profit4, 'exit_tag'] = f'Profit 4 - {market_bias:.3f}'


        # dataframe.loc[
        #     (
        #         (dataframe['fBm'] > np.mean(dataframe['fBm']) + 2 * np.std(dataframe['fBm'])) |
        #         (dataframe['frac_diff'] > 0)
        #     ),
        #     'exit_short'] = 1
        # df['bull_check'] = None
        # profit1_idx = df.index[profit1]
        # for idx in profit1_idx:
        #     period = int(df['period_dc'].loc[idx])
        #     df.loc[idx:idx+period, 'bull_check'] = df['max'].loc[idx]

        return dataframe

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