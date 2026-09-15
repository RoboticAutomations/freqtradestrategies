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


class haFbmVVRPlev(IStrategy):
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
    stoploss = -0.30
    locked_stoploss = {}
    timeframe = '5m'
    leverage = 10.0

    # Stake size adjustments
    stake0 = DecimalParameter(low=0.5, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake1 = DecimalParameter(low=0.5, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake2 = DecimalParameter(low=0.5, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)
    stake3 = DecimalParameter(low=0.5, high=1.0, default=1.0, decimals=1 ,space='buy', optimize=True, load=True)

    ### Custom Functions
    window_size = IntParameter(250, 500, default=266, space='buy', optimize=True)
    # Threshold and Limits
    dc_x = DecimalParameter(low=3.0, high=5.0, default=4.5, decimals=1 ,space='buy', optimize=True, load=True)
    fs1 = DecimalParameter(low=0.5, high=5.0, default=2.0, decimals=1 ,space='buy', optimize=True, load=True)
    fs2 = DecimalParameter(low=0.5, high=5.0, default=1.5, decimals=1 ,space='buy', optimize=True, load=True)
    pt1 = DecimalParameter(low=0.1, high=10.0, default=2.0, decimals=1 ,space='sell', optimize=True, load=True)

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
    # use14 = BooleanParameter(default=True, space="sell", optimize=True, load=True)

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
        "dc_EWM": {
          "color": "#d2a51b",
          "type": "line"
        },
        "upper_envelope_h2": {
          "color": "#e99882"
        },
        "lower_envelope_h2": {
          "color": "#56bb6f"
        }
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
        },
        "S/R": {
          "zero": {
            "color": "#94d1e6"
          },
          "SR_Ratio_Smooth": {
            "color": "#ab4173",
            "type": "line"
          }
        },
        "market bias": {
          "market_bias": {
            "color": "#2bcd6"
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

        TP0 = current_candle['h2_move_mean'] * self.leverage
        TP1 = current_candle['h1_move_mean'] * self.leverage 
        TP2 = current_candle['h0_move_mean'] * self.leverage 
        TP3 = current_candle['cycle_move_mean'] * self.leverage
        bias = current_candle['SR_Ratio_Smooth']
        display_profit = current_profit * 100

        if current_candle['enter_long'] is not None:
            signal = current_candle['enter_long']

        if current_profit is not None:
            logger.info(f"{trade.pair} - Current Profit: {display_profit:.3}%")
        # Take Profit if m00n
        if current_profit > TP2 and trade.nr_of_successful_exits == 0:
            # Take quarter of the profit at next fib%
            return -(trade.stake_amount / 4)
        if current_profit > TP3 and trade.nr_of_successful_exits == 1:
            # Take half of the profit at last fib%
            return -(trade.stake_amount / 4)
        if current_profit > (TP3 * 1.5) and trade.nr_of_successful_exits == 2:
            # Take half of the profit at last fib%
            return -(trade.stake_amount / 4)
        if current_profit > (TP3 * 2.0) and trade.nr_of_successful_exits == 3:
            # Take half of the profit at last fib%
            return -(trade.stake_amount / 4)

        return None
    

    ### Trailing Stop ###
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        trade_duration = (current_time - trade.open_date_utc).seconds / 60
        SLT0 = current_candle['h2_move_mean'] * self.leverage
        SLT1 = current_candle['h1_move_mean'] * self.leverage 
        SLT2 = current_candle['h0_move_mean'] * self.leverage 
        SLT3 = current_candle['cycle_move_mean'] * self.leverage
        bias = current_candle['SR_Ratio_Smooth']
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
                self.dp.send_msg(f'*** {pair} *** Profit 4 {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                logger.info(f'*** {pair} *** Profit 4 {display_profit:.3f}% - {slt3:.3f}%/{sl3:.3f}% activated')
                return SL2
            elif SLT2 is not None and current_profit > SLT2 and bias < 0.5:
                self.locked_stoploss[pair] = SL2
                self.dp.send_msg(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                logger.info(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                return SL2
            elif SLT1 is not None and current_profit > SLT1 and bias < 0:
                self.locked_stoploss[pair] = SL1
                self.dp.send_msg(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                logger.info(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                return SL1
            elif SLT0 is not None and current_profit > SLT0 and bias < 0:
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
            elif SLT2 is not None and current_profit > SLT2 and bias < 0.5:
                self.locked_stoploss[pair] = SL2
                self.dp.send_msg(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                logger.info(f'*** {pair} *** Profit 3 {display_profit:.3f}% - {slt2:.3f}%/{sl2:.3f}% activated')
                return SL2
            elif SLT1 is not None and current_profit > SLT1 and bias < 0:
                self.locked_stoploss[pair] = SL1
                self.dp.send_msg(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                logger.info(f'*** {pair} *** Profit 2 {display_profit:.3f}% - {slt1:.3f}%/{sl1:.3f}% activated')
                return SL1
            elif SLT0 is not None and current_profit > SLT0 and bias < 0:
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

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

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

        ha_df['frac_sma_0'] = ha_df['frac_diff'].ewm(span=int(harmonics[0])).mean()
        ha_df['frac_sma_1'] = ha_df['frac_diff'].ewm(span=int(harmonics[1])).mean()
        ha_df['frac_sma_2'] = ha_df['frac_diff'].ewm(span=int(harmonics[2])).mean()

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
            ha_df['move_mean_bottom'] = ha_df['h2_move_mean']/2
        else:
            # this step is for hyperopting and backtesting to simulate the rolling window of the exchange in live mode.
            # this needs to be a value specific for YOUR exchange and/or timeframe.
            ha_df['cycle_move_mean'] = ha_df['cycle_move'].rolling(1700).mean()        
            ha_df['h0_move_mean'] = ha_df['h0_move'].rolling(1700).mean()
            ha_df['h1_move_mean'] = ha_df['h1_move'].rolling(1700).mean() 
            ha_df['h2_move_mean'] = ha_df['h2_move'].rolling(1700).mean()
            ha_df['move_mean_bottom'] = ha_df['h2_move_mean']/2

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

        ### Visual Volume Range Profile 
        # Calculate the average price 
        ha_df['Average_Price'] = (ha_df['high'] + ha_df['low']) / 2 
        hi = ha_df['high'].max() 
        lo = ha_df['low'].min() 
        width = hi - lo 
 
        # Create bins and labels 
        num_bins = 40
        bin_width = width / num_bins 
        bin_labels = range(1, num_bins + 1) 
 
        # Assign each row to a price bin 
        ha_df['Price Bins'] = pd.cut(ha_df['Average_Price'], bins=num_bins, labels=bin_labels) 
 
        # Calculate the total volume 
        total_volume = ha_df['volume'].sum() 
 
        # Create new columns for the volume bins 
        for bin_label in bin_labels: 
            ha_df[f'Volume Bin {bin_label}'] = 0.00  # Initialize all volume bins to 0 
 
        # Reset starting_point 
        starting_point = lo + bin_width / 2 
 
        # Allocate volume to the corresponding volume bin for each row 
        for index, row in ha_df.iterrows(): 
            if pd.notnull(row['Price Bins']): 
                volume_bin_label = f'Volume Bin {int(row["Price Bins"])}' 
                ha_df.at[index, volume_bin_label] = row['volume'] 
 
        # Create new columns for the mid price of each bin 
        for bin_label in bin_labels: 
            mid_price = starting_point + bin_width * (bin_label - 0.5) 
            ha_df[f'Mid Price Bin {bin_label}'] = mid_price 
 
 
        # Print the DataFrame with the adjusted volume bin lengths 
        pd.set_option('display.float_format', lambda x: '%.4f' % x) 
        volume_bins_sums = ha_df.filter(like='Volume Bin').sum(axis=0) 
 
        # Add the aggregated volume bins to the DataFrame 
        ha_df['Total Volume Bins'] = volume_bins_sums 
 
        # Calculate the percentage of total volume in each volume bin 
        for bin_label in bin_labels: 
            percentage_col = f'Percentage Bin {bin_label}' 
            volume_col = f'Volume Bin {bin_label}' 
            ha_df[percentage_col] = ha_df[volume_col] / total_volume * 100

 
        perc_bins_sums = ha_df.filter(like='Percentage Bin').sum(axis=0) 
        # Calculate the mean of the percentage of volume bins 
        mean_percentage = perc_bins_sums.median() 
 
        ha_df['SR_Ratio'] = 0
        srCount = 0

        # Determine which bins are above the mean 
        for bin_label in bin_labels:
            percentage_col = f'Percentage Bin {bin_label}'
            above_mean_col = f'VRVP_{bin_label}'

            # Check if the percentage bin is above the mean percentage
            ha_df[above_mean_col] = perc_bins_sums.iloc[bin_label-1] > (mean_percentage)
            # Calculate the mid price of the current bin
            mid_price = starting_point + bin_width * (bin_label - 0.5)

            # Assign the mid price to the new column name
            ha_df[above_mean_col] = ha_df[above_mean_col].apply(lambda x: mid_price if x else np.nan)  # Use np.nan instead of None

            mask = ha_df[above_mean_col].notnull()  # Create a mask to select rows where the value is not null

            if mask.any():  # Only proceed if there are non-null values in the mask
                srCount += 1
                ha_df.loc[mask & (ha_df['ha_close'] > ha_df[above_mean_col]), 'SR_Ratio'] += 1 #uptrend
                ha_df.loc[mask & (ha_df['ha_close'] < ha_df[above_mean_col]), 'SR_Ratio'] += -1 #downtrend

        ha_df['SR_Ratio_Smooth'] = ha_df['SR_Ratio'] / srCount # 

        ha_df['SR_Ratio'] = 0
        for i in range(1, 21):
            col = f'VRVP_{i}'
            if col in ha_df.columns:
                ha_df['SR_Ratio'] += np.where(ha_df[col].notnull(),
                                              np.where(ha_df['ha_close'] > ha_df[col], -1, 1),
                                              0)
        sR_zone = ha_df['SR_Ratio_Smooth'].iloc[-1]
        if sR_zone <= -0.8:
            sR = '***Bottom Soon!!!*** '
        elif sR_zone >= 0.8:
            sR = '***Top Soon!!!*** '
        else:
            sR = ''

        logger.info(f'{pair} - Total Vol: {total_volume:.1f} Bin Width: {width:.5f} Mean: {mean_percentage:.2f}% {sR}')
        logger.info(f'{pair} - Bias: {market_bias:.2f} - {market_dir} | DC: {cycle_period:.2f} | 1/2: {harmonics[0]:.2f} | 1/3: {harmonics[1]:.2f} | 1/4: {harmonics[2]:.2f}')
        end_time = time.time()
        logger.info(f"{pair} Indicators calculated in {end_time - start_time:.2f} secs")


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
                (dataframe['market_bias'] < self.fs2.value) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send2, 'enter_long'] = 1
        dataframe.loc[full_send2, 'enter_tag'] = f'Full Send 2 - {market_bias:.3f}'

        full_send3 = (
                (self.use2.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == -1) &
                (dataframe['SR_Ratio_Smooth'] <= -1) &
                (dataframe['h2_move'] > dataframe['h2_move_mean']) &  
                (dataframe['ha_close'] < dataframe['dc_EWM']) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send3, 'enter_long'] = 1
        dataframe.loc[full_send3, 'enter_tag'] = f'Full Send 3 - {market_bias:.3f}'

        full_send4 = (
                (self.use3.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == -0.8) &
                (dataframe['SR_Ratio_Smooth'] <= -0.8) &
                (dataframe['h2_move'] > dataframe['h0_move_mean']) & 
                (dataframe['ha_close'] < dataframe['dc_EWM']) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send4, 'enter_long'] = 1
        dataframe.loc[full_send4, 'enter_tag'] = f'Full Send 4 - {market_bias:.3f}'

        full_send5 = (
                (self.use4.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == -0.6) &
                (dataframe['SR_Ratio_Smooth'] <= -0.6) &
                (dataframe['h2_move'] > dataframe['cycle_move_mean']) & 
                (dataframe['ha_close'] < dataframe['dc_EWM']) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send5, 'enter_long'] = 1
        dataframe.loc[full_send5, 'enter_tag'] = f'Full Send 5 - {market_bias:.3f}'

        full_send6 = (
                (self.use5.value == True) &
                (dataframe['frac_sma_dc'].shift() > dataframe['signal_MEAN_UP_dc']) &
                (dataframe['frac_sma_dc'] < dataframe['signal_MEAN_UP_dc']) &
                (dataframe['h2_move'] > dataframe['h0_move_mean']) &  
                (dataframe['market_bias'] > self.pt1.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send6, 'enter_short'] = 1
        dataframe.loc[full_send1, 'enter_tag'] = f'Full Send 6 - {market_bias:.3f}'

        full_send7 = (
                (self.use6.value == True) &
                (dataframe['frac_sma_dc'].shift() > dataframe['signal_MEAN_UP_dc']) &
                (dataframe['frac_sma_dc'] < dataframe['signal_MEAN_UP_dc']) & 
                (dataframe['h2_move'] > dataframe['h2_move_mean']) &                
                (dataframe['market_bias'] < dataframe['market_bias'].shift()) &    
                (dataframe['market_dir'] == -1) &
                (dataframe['market_bias'] > self.fs2.value) & 
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send7, 'enter_short'] = 1
        dataframe.loc[full_send7, 'enter_tag'] = f'Full Send 7 - {market_bias:.3f}'

        full_send8 = (
                (self.use7.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == 1) &
                (dataframe['SR_Ratio_Smooth'] <= 1) &
                (dataframe['ha_close'] > dataframe['dc_EWM']) &
                (dataframe['h2_move'] > dataframe['h2_move_mean']) &  
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send8, 'enter_short'] = 1
        dataframe.loc[full_send8, 'enter_tag'] = f'Full Send 8 - {market_bias:.3f}'

        full_send9 = (
                (self.use8.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == 0.8) &
                (dataframe['SR_Ratio_Smooth'] <= 0.8) &
                (dataframe['h2_move'] > dataframe['h0_move_mean']) & 
                (dataframe['ha_close'] > dataframe['dc_EWM']) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send9, 'enter_short'] = 1
        dataframe.loc[full_send9, 'enter_tag'] = f'Full Send 9 - {market_bias:.3f}'

        full_send10 = (
                (self.use9.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == 0.6) &
                (dataframe['SR_Ratio_Smooth'] <= 0.6) &
                (dataframe['h2_move'] > dataframe['cycle_move_mean']) & 
                (dataframe['ha_close'] > dataframe['dc_EWM']) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[full_send10, 'enter_short'] = 1
        dataframe.loc[full_send10, 'enter_tag'] = f'Full Send 10 - {market_bias:.3f}'



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
                (dataframe['SR_Ratio_Smooth'].shift() == 1) &
                (dataframe['SR_Ratio_Smooth'] <= 1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit2, 'exit_long'] = 1
        dataframe.loc[profit2, 'exit_tag'] = f'Profit 2 - {market_bias:.3f}'

        profit3 = (
                (self.use12.value == True) &
                (dataframe['frac_sma_dc'].shift() < dataframe['signal_MEAN_DN_dc']) &
                (dataframe['frac_sma_dc'] > dataframe['signal_MEAN_DN_dc']) &
                (dataframe['h2_move'] > dataframe['h0_move_mean']) &  
                (dataframe['market_bias'] < self.pt1.value) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit3, 'exit_short'] = 1
        dataframe.loc[profit3, 'exit_tag'] = f'Profit 3 - {market_bias:.3f}'

        profit4 = (
                (self.use13.value == True) &
                (dataframe['SR_Ratio_Smooth'].shift() == -1) &
                (dataframe['SR_Ratio_Smooth'] >= -1) &
                (dataframe['volume'] > 0)   # Make sure Volume is not 0
        )
            
        dataframe.loc[profit3, 'exit_long'] = 1
        dataframe.loc[profit3, 'exit_tag'] = f'Profit 3 - {market_bias:.3f}'

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
